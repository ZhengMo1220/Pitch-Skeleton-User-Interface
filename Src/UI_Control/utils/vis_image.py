import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os
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

font_path = os.path.join(application_path, 'R-PMingLiU-TW-2.ttf')
print("Font path:", font_path)  # 打印以确认路径
fontStyle = ImageFont.truetype(font_path, 20)

def draw_cross(image, x, y, length=5, color=(0, 0, 255), thickness=2):
    cv2.line(image, (x, y - length), (x, y + length), color, thickness)
    cv2.line(image, (x - length, y), (x + length, y), color, thickness)

def draw_grid(image:np.ndarray):
    #return image:np.ndarray
    
    height, width = image.shape[:2]


    draw_cross(image,int(width/2),int(height/2),length=20,color=(0,0,255),thickness = 3)


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


def draw_bbox(person_data, image):
    person_ids = person_data['person_id']
    person_bbox = person_data['bbox']
    for id, bbox in zip(person_ids, person_bbox):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        color = tuple(colors[id % len(colors)])
        color = (0,255,0)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
        image = cv2.putText(image, str(id), (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 1.5, color, 2)
    return image

def draw_inverted_triangle(image, center, size, color, border_color=(0, 0, 0), border_thickness=2):
    """
    在图像上绘制一个倒三角形，并添加黑色边框。

    参数:
    - image: 输入图像。
    - center: 三角形的底部顶点（x, y）。
    - size: 三角形的边长。
    - color: 三角形的颜色（B, G, R）。
    - border_color: 边框的颜色（默认黑色）。
    - border_thickness: 边框的粗细（默认2）。
    """
    height = int(size * (np.sqrt(3) / 2))  # 计算三角形的高度
    vertices = np.array([
        [center[0], center[1]],  # 底部顶点
        [center[0] - size // 2, center[1] - height],  # 左上顶点
        [center[0] + size // 2, center[1] - height]   # 右上顶点
    ], np.int32)
    vertices = vertices.reshape((-1, 1, 2))
    
    # 先画边框
    cv2.polylines(image, [vertices], isClosed=True, color=border_color, thickness=border_thickness)
    # 填充三角形
    cv2.fillPoly(image, [vertices], color)

def draw_traj(kpt_buffer, img):
    if len(kpt_buffer) == 0 or len(kpt_buffer) == 1:
        return img
    image = img.copy()
    
    for i in range(1, len(kpt_buffer)):
        f_kptx, f_kpty = map(int, kpt_buffer[i-1])
        s_kptx, s_kpty = map(int, kpt_buffer[i])
        cv2.line(image, (f_kptx, f_kpty), (s_kptx, s_kpty), (0, 255, 0), 5)

    return image

def draw_video_traj(img, person_df, person_id, kpt_id, frame_num):
    if person_df.empty:
        return img
    
    image = img.copy()


    for i in range(0, frame_num):
        pre_person_data = person_df.loc[(person_df['frame_number'] == i-1) &
                    (person_df['person_id'] == person_id)]
        
        curr_person_data = person_df.loc[(person_df['frame_number'] == i) &
                    (person_df['person_id'] == person_id)]
        
        if pre_person_data.empty:
            continue
        if curr_person_data.empty:
            continue
        pre_kptx, pre_kpty, _, _ = pre_person_data['keypoints'].iloc[0][kpt_id]
        curr_kptx, curr_kpty, _, _ = curr_person_data['keypoints'].iloc[0][kpt_id]

        cv2.line(image, (int(pre_kptx), int(pre_kpty)), (int(curr_kptx), int(curr_kpty)), (0, 255, 0), 5)

    return image

def draw_angle_info(img: np.ndarray, angle_info: pd.DataFrame, frame_num:int):
    if angle_info is None:
        return image
    
    image = img.copy()
    data = angle_info.loc[angle_info['frame_number'] == (frame_num-1)]
    data = data['angle'].iloc[0]

    for _, info in data.items(): 
        angle = int(info[0])
        pt1, pt2, pt3 = [tuple(map(int, point)) for point in info[1]]
        image = cv2.putText(image, str(angle), (pt2[0] - 10, pt2[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
    
    return image

# def obtain_curr_info(angle_info: pd.DataFrame, frame_num:int):
#     print(angle_info)
#     print(frame_num)
#     data = angle_info.loc[angle_info['frame_number'] == (frame_num-1)]

#     return data

def draw_region(img:np.ndarray):
    image = img.copy()
    cv2.rectangle(image, (100, 250), (450, 600), (0, 255, 0), -1)
    return image