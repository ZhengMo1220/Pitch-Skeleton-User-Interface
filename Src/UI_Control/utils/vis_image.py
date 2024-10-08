import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from .analyze import PoseAnalyzer
from skeleton.detect_skeleton import PoseEstimater
import os
from scipy.signal import savgol_filter
import sys
try:
    colors = np.round(
        np.array(plt.get_cmap('gist_rainbow').colors) * 255
    ).astype(np.uint8)[:, ::-1].tolist()
except AttributeError:  # if palette has not pre-defined colors
    colors = np.round(
        np.array(plt.get_cmap('gist_rainbow')(np.linspace(0, 1, 10))) * 255
    ).astype(np.uint8)[:, -2::-1].tolist()

if getattr(sys, 'frozen', False):
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(__file__)

class ImageDrawer():
    def __init__(self, pose_estimater: PoseEstimater=None, pose_analyzer:PoseAnalyzer=None, angle_name:str = "右手肘"):
        self.font_path = os.path.join(application_path, 'R-PMingLiU-TW-2.ttf')
        self.fontStyle = ImageFont.truetype(self.font_path, 20)
        self.pose_estimater = pose_estimater
        self.pose_analyzer = pose_analyzer
        self.angle_name = angle_name
        self.show_grid = False
        self.show_bbox = False
        self.show_skeleton = False
        self.show_traj = False
        self.show_region = False
        self.show_angle_info = False
        self.angle_info_pos = (0,0)
        self.region = [(100, 250), (450, 600)]
    
    def draw_info(self, img:np.ndarray, frame_num:int=None, kpt_buffer:list = None):
        if img is None:
            return
        image = img.copy()
        curr_person_df = self.pose_estimater.getPersonDf(frame_num = frame_num, is_select=True)
        if self.show_region:
            image = self.draw_region(image)

        if self.show_grid :
            image = self.draw_grid(image)
        
        if self.show_bbox:
            image = self.draw_bbox(image, curr_person_df)
        
        if self.show_skeleton:
            image = self.draw_points_and_skeleton(image, curr_person_df, self.pose_estimater.joints['haple']['skeleton_links'], 
                                                points_palette_samples=10)
        
        if self.show_traj:
            image = self.draw_traj(image, kpt_buffer)

        if self.show_angle_info:
            image = self.draw_angle_info(image, frame_num)
        
        return image

    def draw_cross(self, image, x, y, length=5, color=(0, 0, 255), thickness=2):
        cv2.line(image, (x, y - length), (x, y + length), color, thickness)
        cv2.line(image, (x - length, y), (x + length, y), color, thickness)

    def draw_grid(self, image:np.ndarray):
        #return image:np.ndarray
        
        height, width = image.shape[:2]
        self.draw_cross(image,int(width/2),int(height/2),length=20,color=(0,0,255),thickness = 3)
        # 計算垂直線的位置
        vertical_interval = width // 5
        vertical_lines = [vertical_interval * i for i in range(1, 5)]

        # 計算水平線的位置
        horizontal_interval = height // 5
        horizontal_lines = [horizontal_interval * i for i in range(1, 5)]

        # 畫垂直線
        for x in vertical_lines:
            cv2.line(image, (x, 0), (x, height), (0, 255, 0), 2)

        # 畫水平線
        for y in horizontal_lines:
            cv2.line(image, (0, y), (width, y), (0, 255, 0), 2)

        return image

    def draw_bbox(self, image:np.ndarray, person_df:pd.DataFrame):
        if person_df.empty:
            return image
        person_ids = person_df['person_id']
        person_bbox = person_df['bbox']
        for id, bbox in zip(person_ids, person_bbox):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            color = tuple(colors[id % len(colors)])
            color = (0,255,0)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
            image = cv2.putText(image, str(id), (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 1.5, color, 2)
        return image

    def draw_traj(self, img: np.ndarray, kpt_buffer: list):
        if not kpt_buffer or len(kpt_buffer) < 2:
            return img

        # 將座標轉換為整數
        int_kpt_buffer = [tuple(map(int, kpt)) for kpt in kpt_buffer]
        
        # 迭代相鄰的兩個點，並畫出軌跡線
        for (f_kptx, f_kpty), (s_kptx, s_kpty) in zip(int_kpt_buffer[:-1], int_kpt_buffer[1:]):
            cv2.line(img, (f_kptx, f_kpty), (s_kptx, s_kpty), (0, 255, 0), 5)

        return img

    def draw_angle_info(self, img: np.ndarray, frame_num: int) -> np.ndarray:
        # 从 pose_analyzer 获取角度数据
        _, angle_info = self.pose_analyzer.get_frame_angle_data(frame_num, self.angle_name)
        # 提取角度值，并将其转换为整数
        angle_value = int(angle_info[0])
        
        # 提取坐标并将其转换为整数元组
        p = tuple(map(int, angle_info[1][1]))
        # 使用 cv2.line 绘制线条
        cv2.line(img, p, (self.angle_info_pos[0] + 20, self.angle_info_pos[1] + 320), (0, 0, 0), 2)
        
        # 使用 cv2.putText 绘制角度值
        img = cv2.putText(img, str(angle_value), (self.angle_info_pos[0] - 50, self.angle_info_pos[1] + 420), 
                        cv2.FONT_HERSHEY_COMPLEX, 3.5, (0, 255, 0), 3)
        
        return img

    def draw_region(self, img:np.ndarray):
        cv2.rectangle(img, self.region[0], self.region[1], (0, 255, 0), -1)
        return img

    def draw_points(self, image, points, person_idx, color_palette='gist_rainbow', palette_samples=10, confidence_threshold=0.3):
        """
        Draws `points` on `image`.

        Args:
            image: image in opencv format
            points: list of points to be drawn.
                Shape: (nof_points, 3)
                Format: each point should contain (y, x, confidence)
            color_palette: name of a matplotlib color palette
                Default: 'tab20'
            palette_samples: number of different colors sampled from the `color_palette`
                Default: 16
            confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
                Default: 0.5

        Returns:
            A new image with overlaid points

        """
        try:
            colors = np.round(
                np.array(plt.get_cmap(color_palette).colors) * 255
            ).astype(np.uint8)[:, ::-1].tolist()
        except AttributeError:  # if palette has not pre-defined colors
            colors = np.round(
                np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
            ).astype(np.uint8)[:, -2::-1].tolist()

        circle_size = max(1, min(image.shape[:2]) // 160)  # ToDo Shape it taking into account the size of the detection
        # circle_size = max(2, int(np.sqrt(np.max(np.max(points, axis=0) - np.min(points, axis=0)) // 16)))
        for i, pt in enumerate(points):
        
            unlabel = False if pt[0] != 0 and pt[1] != 0 else True
            if pt[2] > confidence_threshold and not unlabel:
                image = cv2.circle(image, (int(pt[1]), int(pt[0])), circle_size, tuple(colors[person_idx % len(colors)]), -1)

        return image

    def draw_skeleton(self, image, points, skeleton, color_palette='Set2', palette_samples='jet', person_index=0,
                    confidence_threshold=0.5):
        """
        Draws a `skeleton` on `image`.

        Args:
            image: image in opencv format
            points: list of points to be drawn.
                Shape: (nof_points, 3)
                Format: each point should contain (y, x, confidence)
            skeleton: list of joints to be drawn
                Shape: (nof_joints, 2)
                Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
            color_palette: name of a matplotlib color palette
                Default: 'Set2'
            palette_samples: number of different colors sampled from the `color_palette`
                Default: 8
            person_index: index of the person in `image`
                Default: 0
            confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
                Default: 0.5

        Returns:
            A new image with overlaid joints

        """
        try:
            colors = np.round(
                np.array(plt.get_cmap(color_palette).colors) * 255
            ).astype(np.uint8)[:, ::-1].tolist()
        except AttributeError:  # if palette has not pre-defined colors
            colors = np.round(
                np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
            ).astype(np.uint8)[:, -2::-1].tolist()
        right_skeleton = self.pose_estimater.joints['haple']['right_points_indices']
        left_skeleton = self.pose_estimater.joints['haple']['left_points_indices']
        
        for i, joint in enumerate(skeleton):
            pt1, pt2 = points[joint]
            pt1_unlabel = False if pt1[0] != 0 and pt1[1] != 0 else True
            pt2_unlabel = False if pt2[0] != 0 and pt2[1] != 0 else True
            skeleton_color = tuple(colors[person_index % len(colors)])
            skeleton_color = (0, 165, 255)
            if joint in right_skeleton:
                skeleton_color = (240, 176, 0)
            elif joint in left_skeleton:
                skeleton_color = (0, 0, 255)
            if pt1[2] > confidence_threshold and not pt1_unlabel and pt2[2] > confidence_threshold and not pt2_unlabel:
                image = cv2.line(
                    image, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])),
                    skeleton_color , 6
                )
        return image

    def draw_points_and_skeleton(self, image, person_df, skeleton, points_color_palette='gist_rainbow', points_palette_samples=10,
                                skeleton_color_palette='Set2', skeleton_palette_samples='jet', confidence_threshold=0.3):
        """
        Draws `points` and `skeleton` on `image`.

        Args:
            image: image in opencv format
            points: list of points to be drawn.
                Shape: (nof_points, 3)
                Format: each point should contain (y, x, confidence)
            skeleton: list of joints to be drawn
                Shape: (nof_joints, 2)
                Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
            points_color_palette: name of a matplotlib color palette
                Default: 'tab20'
            points_palette_samples: number of different colors sampled from the `color_palette`
                Default: 16
            skeleton_color_palette: name of a matplotlib color palette
                Default: 'Set2'
            skeleton_palette_samples: number of different colors sampled from the `color_palette`
                Default: 8
            person_index: index of the person in `image`
                Default: 0
            confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
                Default: 0.5

        Returns:
            A new image with overlaid joints

        """
        if person_df is None:
            return image
        if person_df.empty:
            return image
        person_data = self.df_to_points(person_df)
        for person_id, points in person_data.items(): 
            image = self.draw_skeleton(image, points, skeleton,person_index=person_id)
            image = self.draw_points(image, points,person_idx=person_id)
        return image

    def df_to_points(self, person_df):
        person_data = {}
        person_ids = person_df['person_id']
        person_kpts = person_df['keypoints']
        for id, kpts in zip(person_ids, person_kpts):
            person_data[id] = np.array(self.swap_values(kpts))
        return person_data

    def swap_values(self, kpts):
        return [[item[1], item[0], item[2]] for item in kpts]

    def setShowBbox(self, status:bool):
        self.show_bbox = status
    
    def setShowSkeleton(self, status:bool):
        self.show_skeleton = status
    
    def setShowGrid(self, status:bool):
        self.show_grid = status
        
    def set_show_region(self, status:bool):
        self.show_region = status

    def setShowTraj(self, status:bool):
        self.show_traj = status

    def setShowAngleInfo(self, status:bool):
        self.show_angle_info = status
        self.setAngleInfoPos()

    def setAngleInfoPos(self):
        person_df = self.pose_estimater.getPersonDf(is_select=True)
        if person_df is None:
            return
        self.angle_info_pos = person_df.iloc[0]['keypoints'][19]
        self.angle_info_pos = tuple(map(int,self.angle_info_pos))

    def reset(self):
        self.show_grid = False
        self.show_bbox = False
        self.show_skeleton = False
        self.show_traj = False
        self.show_angle_info = False
        self.angle_info_pos = (0,0)