from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import numpy as np
import sys
import cv2
import os
from camera_ui import Ui_camera_ui
import pandas as pd
import queue
from utils.cv_thread import VideoCaptureThread, VideoWriter
from utils.one_euro_filter import OneEuroFilter
from utils.timer import FPS_Timer
from utils.vis_image import draw_grid, draw_bbox, draw_traj
from utils.vis_pose import draw_points_and_skeleton, joints_dict
from datetime import datetime
from topdown_demo_with_mmdet import process_one_image
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class PoseCameraTabControl(QWidget):
    def __init__(self, model, parent = None):
        super(PoseCameraTabControl, self).__init__(parent)
        self.ui = Ui_camera_ui()
        self.ui.setupUi(self)
        self.init_var()
        self.bind_ui()
        self.video_writer = None
        self.model = model
        self.fps_timer = FPS_Timer()
        self.smooth_filter = OneEuroFilter()

    def bind_ui(self):
        self.ui.record_checkBox.setDisabled(True)
        self.ui.camera_checkBox.clicked.connect(self.toggle_camera)
        self.ui.show_skeleton_checkBox.clicked.connect(self.toggle_analyze)
        self.ui.record_checkBox.clicked.connect(self.toggle_record)
        self.ui.select_checkBox.clicked.connect(self.toggle_select)
        self.ui.select_keypoint_checkbox.clicked.connect(self.toggle_select_kpt)

    def init_var(self):
        self.db_path = f"../../Db"
        self.is_opened = False
        self.is_analyze = self.ui.show_skeleton_checkBox.isChecked()
        self.is_record = False
        self.is_select_kpt = False
        self.select_person_id = None
        self.select_kpt_id = -1
        self.select_kpt_buffer = []
        self.buffer_len = 15
        self.fps_control = 1
        self.pre_person_df = pd.DataFrame()
        self.camera_scene = QGraphicsScene()
        self.person_df = pd.DataFrame()
        self.frame_buffer = queue.Queue()
        self.frame_count = 0
        self.kpts_dict = joints_dict()['haple']['keypoints']

    def toggle_camera(self):
        if self.is_opened:
            self.close_camera()
            self.ui.image_resolution_label.setText(f"(0, 0) - ")
            self.ui.record_checkBox.setDisabled(True)
        else:
            self.open_camera()
            frame_width = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.video_thread.cap.get(cv2.CAP_PROP_FPS))
            self.ui.image_resolution_label.setText(f"(0, 0) - ({frame_width} x {frame_height}), FPS: {fps}")
            self.ui.record_checkBox.setEnabled(True)
    
    def toggle_record(self):
        if self.is_record:
            self.stop_recording()
        else:
            self.start_recording()

    def toggle_analyze(self):
        if self.is_analyze:
            self.is_analyze = False
        else:
            self.is_analyze = True

    def toggle_select(self):
        if not self.ui.select_checkBox.isChecked():
            self.select_person_id = None
    
    def toggle_select_kpt(self):
        if not self.ui.select_keypoint_checkbox.isChecked():
            self.select_kpt_id = -1
            self.select_kpt_buffer = []
        
    def open_camera(self):
        self.video_thread = VideoCaptureThread(camera_index=self.ui.camera_id_input.value())
        self.video_thread.frame_ready.connect(self.buffer_frame)
        self.video_thread.start_capture()
        self.is_opened = True

    def close_camera(self):
        self.video_thread.stop_capture()
        self.camera_scene.clear()
        self.is_opened = False

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def buffer_frame(self, frame:np.ndarray):
        self.frame_count += 1
        if self.is_analyze:
            if self.is_record:
                self.fps_control = 30
            else:
                self.fps_control = 15
    
        if not self.frame_buffer.full() and self.frame_count % self.fps_control ==0:     
            self.frame_buffer.put(frame)
            self.analyze_frame()


        if self.video_writer is not None:
            self.video_writer.write(frame)
    
    def start_recording(self):
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = f'../../Db/Record/C{self.ui.camera_id_input.value()}_Fps120_{current_time}'
        os.makedirs(output_dir, exist_ok=True)
        video_filename = os.path.join(output_dir, f'C{self.ui.camera_id_input.value()}_Fps120_{current_time}.mp4')
        frame_width = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.video_thread.cap.get(cv2.CAP_PROP_FPS))
        self.video_writer = VideoWriter(video_filename, frame_width, frame_height,fps=fps)
        self.is_record = True

    def stop_recording(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.is_record = False

    def show_image(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView):
        scene.clear()
        image = cv2.circle(image, (0, 0), 10, (0, 0, 255), -1)
        h, w = image.shape[:2]
        qImg = QImage(image, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        scene.addPixmap(pixmap)
        GraphicsView.setScene(scene)
        GraphicsView.setAlignment(Qt.AlignLeft)
        GraphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def merge_keypoint_datas(self, pred_instances):
        return [
            np.hstack(
                (
                    np.round(person['keypoints'][0], 2),
                    np.round(person['keypoint_scores'][0], 2).reshape(-1, 1),
                    np.full((len(person['keypoints'][0]), 1), False, dtype=bool)
                )
            )
            for person in pred_instances
        ]

    def merge_person_datas(self, person_ids:list, person_bboxes:np.ndarray, person_kpts:np.ndarray):
        person_data = []
        for pid, bbox, kpts in zip(person_ids, person_bboxes, person_kpts):
            new_kpts = np.zeros((len(self.kpts_dict), kpts.shape[1]))
            new_kpts[:26] = kpts
            new_kpts[26:, 2] = 0.9
            person_data.append({
                'person_id': pid,
                'bbox': bbox,
                'keypoints': new_kpts
            })
        if person_data:
            self.person_df = pd.DataFrame(person_data)

    def analyze_frame(self):
        if not self.frame_buffer.empty():
            frame = self.frame_buffer.get()
            img = frame.copy()
            if self.is_analyze:
                self.fps_timer.tic()
                pred_instances, person_ids = process_one_image(self.model, img, select_id=self.select_person_id)
                average_time = self.fps_timer.toc()
                fps= int(1/max(average_time,0.00001))
                if fps <10:
                    self.ui.fps_info_label.setText(f"0{fps}")
                else:
                    self.ui.fps_info_label.setText(f"{fps}")
                person_kpts = self.merge_keypoint_datas(pred_instances)
                person_bboxes = pred_instances['bboxes']
                self.merge_person_datas(person_ids, person_bboxes, person_kpts)
                self.smooth_kpt(person_ids)
                if self.select_kpt_id != -1:
                    self.buffer_kpt_info(person_kpts[0])
            self.update_frame(img)

    def update_frame(self, image:np.ndarray):
        if not self.person_df.empty:
            if self.ui.show_skeleton_checkBox.isChecked():
                image = draw_points_and_skeleton(image, self.person_df, joints_dict()['haple']['skeleton_links'],
                                                points_color_palette='gist_rainbow', skeleton_palette_samples='jet',
                                                points_palette_samples=10, confidence_threshold=0.3)
            if self.ui.show_bbox_checkbox.isChecked():
                image = draw_bbox(self.person_df, image)
                
            if self.ui.select_keypoint_checkbox.isChecked():
                image = draw_traj(self.select_kpt_buffer,image)

        if self.ui.show_line_checkBox.isChecked():
            image = draw_grid(image)
        self.show_image(image, self.camera_scene, self.ui.camer_frame_view)
        self.person_df = pd.DataFrame()

    def smooth_kpt(self, person_ids:list):
        if self.pre_person_df.empty :
            self.pre_person_df = self.person_df.copy()

        if self.person_df.empty:
            return  # 跳过当前 frame
        
        for person_id in person_ids: 
            pre_person_data = self.pre_person_df.loc[self.pre_person_df['person_id'] == person_id]
            curr_person_data = self.person_df.loc[self.person_df['person_id'] == person_id]
            
            # 确保前一帧和当前帧的数据都不为空
            if curr_person_data.empty or pre_person_data.empty:
                continue  # 跳过当前 ID
            
            pre_kpts = pre_person_data.iloc[0]['keypoints'] if not pre_person_data.empty else None
            curr_kpts = curr_person_data.iloc[0]['keypoints'] if not curr_person_data.empty else None
            
            smoothed_kpts = []
            
            if curr_kpts is not None and pre_kpts is not None:
                for pre_kpt, curr_kpt in zip(pre_kpts, curr_kpts): 
                    pre_kptx, pre_kpty = pre_kpt[0], pre_kpt[1]
                    curr_kptx, curr_kpty, curr_conf, curr_label = curr_kpt[0], curr_kpt[1], curr_kpt[2], curr_kpt[3]
                    
                    # 仅当所有关键点都有有效坐标时进行平滑处理
                    if all([pre_kptx != 0, pre_kpty != 0, curr_kptx != 0, curr_kpty != 0]):
                        curr_kptx = self.smooth_filter(curr_kptx, pre_kptx)
                        curr_kpty = self.smooth_filter(curr_kpty, pre_kpty)
                    
                    smoothed_kpts.append([curr_kptx, curr_kpty, curr_conf, curr_label])
            
                # 更新当前帧的关键点数据
                self.person_df.at[curr_person_data.index[0], 'keypoints'] = smoothed_kpts
        
        # 更新前一帧数据
        self.pre_person_df = self.person_df.copy()  # 确保拷贝数据

    def mousePressEvent(self, event):
        if self.ui.select_checkBox.isChecked():
            pos = event.pos()
            scene_pos = self.ui.camer_frame_view.mapToScene(pos)
            x, y = scene_pos.x(), scene_pos.y()
            if event.button() == Qt.LeftButton:
                self.person_id_selector(x, y)

        if self.ui.select_keypoint_checkbox.isChecked():
            pos = event.pos()
            scene_pos = self.ui.camer_frame_view.mapToScene(pos)
            x, y = scene_pos.x(), scene_pos.y()
            if event.button() == Qt.LeftButton:
                self.kpt_id_selector(x, y)

    def person_id_selector(self, x, y):
        if self.pre_person_df.empty:
            return
        
        selected_id = None
        max_area = -1

        for _, row in self.pre_person_df.iterrows():
            person_id = row['person_id']
            bbox = row['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            if x1 <= x <= x2 and y1 <= y <= y2:
                w = x2 - x1
                h = y2 - y1
                area = w * h

                if area > max_area:
                    max_area = area
                    selected_id = person_id

        self.select_person_id = selected_id

    def kpt_id_selector(self, x, y):
        def calculate_distance(point1, point2):
            return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        
        if self.pre_person_df.empty :
            return

        selected_id = None
        min_distance = 10000

        for _, row in self.pre_person_df.iterrows():
            person_kpts = row['keypoints']
            for kpt_id, kpt in enumerate(person_kpts):
                
                kptx, kpty, kpt_score, _= map(int, kpt)
                
                # if kpt_score
                distance = calculate_distance([kptx, kpty], [x, y])
                if distance < min_distance:
                    min_distance = distance
                    selected_id = kpt_id

        self.select_kpt_id = selected_id

    def buffer_kpt_info(self, person_kpts):
        if len(self.select_kpt_buffer) > self.buffer_len:
            _ = self.select_kpt_buffer.pop(0)
        self.select_kpt_buffer.append(person_kpts[self.select_kpt_id][:2])
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PoseCameraTabControl()
    window.show()
    sys.exit(app.exec_())
