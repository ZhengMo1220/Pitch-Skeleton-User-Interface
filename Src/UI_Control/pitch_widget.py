from PyQt5.QtWidgets import *
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtCore import Qt
import numpy as np
import sys
import cv2
import os
from pitch_ui import Ui_pitch_ui
import pandas as pd
import queue
from utils.cv_thread import VideoCaptureThread, VideoWriter
from datetime import datetime
from utils.timer import FPS_Timer, Timer
import time
from utils.vis_image import draw_grid, draw_bbox, draw_traj, draw_region
from utils.vis_pose import draw_points_and_skeleton, joints_dict
from utils.store import save_video
from topdown_demo_with_mmdet import process_one_image
import sys
from utils.one_euro_filter import OneEuroFilter


class PosePitchTabControl(QWidget):
    def __init__(self, model,parent= None):
        super(PosePitchTabControl, self).__init__(parent)
        self.ui = Ui_pitch_ui()
        self.ui.setupUi(self)
        self.init_var()
        self.bind_ui()
        self.video_writer = None
        self.model = model 
        self.smooth_filter = OneEuroFilter()
        self.fps_timer = FPS_Timer()

    def bind_ui(self):
        self.ui.record_checkbox.setDisabled(True)
        self.ui.camera_checkbox.clicked.connect(self.toggle_camera)
        self.ui.record_checkbox.clicked.connect(self.toggle_record)
        self.ui.select_checkbox.clicked.connect(self.toggle_select)
        self.ui.select_keypoint_checkbox.clicked.connect(self.toggle_select_kpt)
        self.ui.frame_slider.valueChanged.connect(self.video_analyze_frame)
        self.ui.play_btn.clicked.connect(self.play_btn_clicked)
        self.ui.back_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        )
        self.ui.forward_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        )

    def init_var(self):
        self.select_person_id = None
        self.select_kpt_id = None
        self.trigger_record_timer = Timer(3)
        self.trigger_pitch_timer = Timer(3)
        self.select_kpt_buffer = []
        self.record_buffer = []
        self.is_play = False
        self.processed_images=-1
        self.region = [(100, 250), (450, 600)]
        self.fps_control = 1
        self.pre_person_df = pd.DataFrame()
        self.camera_scene = QGraphicsScene()
        self.person_df = pd.DataFrame()
        self.person_data = []
        self.frame_buffer = queue.Queue(maxsize=100)
        
        self.video_name = ""
        self.frame_count = 0
        self.kpts_dict = joints_dict()['haple']['keypoints']

    def toggle_camera(self):
        if self.ui.camera_checkbox.isChecked():
            self.open_camera()
            frame_width = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.video_thread.cap.get(cv2.CAP_PROP_FPS))
            self.ui.image_resolution_label.setText(f"(0, 0) - ({frame_width} x {frame_height}), FPS: {fps}")
            self.ui.record_checkbox.setEnabled(True)
            self.video_silder(visible=False)
        else:
            self.close_camera()
            self.ui.image_resolution_label.setText(f"(0, 0) - ")
            self.ui.record_checkbox.setDisabled(True)
            self.video_silder(visible=True)

    def toggle_record(self):
        if self.ui.record_checkbox.isChecked():
            self.start_recording()
        else:
            self.stop_recording()

    def toggle_select(self):
        if not self.ui.select_checkbox.isChecked():
            self.select_person_id = None
        else:
            self.person_id_selector(0.0, 0.0)
    
    def toggle_select_kpt(self):
        if not self.ui.select_keypoint_checkbox.isChecked():
            self.select_kpt_id = None
            self.select_kpt_buffer = []
        else:
            if self.ui.pitch_input.currentIndex() == 0:
                self.select_kpt_id = 10
            else:
                self.select_kpt_id = 9

    def toggle_analyze_video(self, video):
        self.video_ui_set(video)
        self.checkbox_controller(record=False, show_skeleton=True, show_bbox=False)
                                    #   select_person=True, select_kpt=True, show_kpt_angle= True)
        self.processed_frames = set()
        self.ui.play_btn.click()

    def open_camera(self):
        self.video_thread = VideoCaptureThread(camera_index=self.ui.camera_id_input.value())
        self.video_thread.frame_ready.connect(self.buffer_frame)
        self.video_thread.start_capture()

    def close_camera(self):
        self.video_thread.stop_capture()
        self.camera_scene.clear()

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def buffer_frame(self, frame: np.ndarray):
        self.frame_count += 1
        if self.ui.show_skeleton_checkbox.isChecked():
            if self.ui.record_checkbox.isChecked():
                self.fps_control = 30
            else:
                self.fps_control = 15

        if self.frame_count % self.fps_control == 0:
            try:
                if self.frame_buffer.full():
                    self.frame_buffer.get_nowait()
                self.frame_buffer.put_nowait(frame)
            except queue.Full:
                print("Frame buffer is full. Dropping frame.")

            self.real_time_analyze_frame()

        #確認有沒有在錄影
        if self.video_writer is not None:
            self.video_writer.write(frame)
            #存取投球的畫面
            self.record_buffer.append(frame)

        #設定秒數的錄影
        if self.trigger_pitch_timer.is_time_up():
            self.checkbox_controller(camera=False, record=False)
            self.toggle_analyze_video(self.record_buffer)
            self.trigger_pitch_timer.reset()

    def start_recording(self):
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = f'../../Db/Record/C{self.ui.camera_id_input.value()}_Fps120_{current_time}'
        os.makedirs(output_dir, exist_ok=True)
        video_filename = os.path.join(output_dir, f'C{self.ui.camera_id_input.value()}_Fps120_{current_time}.mp4')
        self.video_name = video_filename
        frame_width = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.video_thread.cap.get(cv2.CAP_PROP_FPS))
        self.video_writer = VideoWriter(video_filename, frame_width, frame_height, fps=fps)
        self.record_buffer = []

    def stop_recording(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

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

    def merge_person_data(self, pred_instances, person_ids: list, frame_num: int = None):
        person_bboxes = pred_instances['bboxes']
        if frame_num is None:
            self.person_data = []

        for person, pid, bbox in zip(pred_instances, person_ids, person_bboxes):
            keypoints_data = np.hstack((
                np.round(person['keypoints'][0], 2),
                np.round(person['keypoint_scores'][0], 2).reshape(-1, 1),
                np.full((len(person['keypoints'][0]), 1), False, dtype=bool)
            ))

            new_kpts = np.full((len(self.kpts_dict), keypoints_data.shape[1]), 0.9)
            new_kpts[:26] = keypoints_data

            person_info = {
                'person_id': pid,
                'bbox': bbox,
                'keypoints': new_kpts
            }

            if frame_num is not None:
                person_info['frame_number'] = frame_num

            self.person_data.append(person_info)

        return pd.DataFrame(self.person_data)

    def real_time_analyze_frame(self):
        if not self.frame_buffer.empty():
            frame = self.frame_buffer.get()
            img = frame.copy()
            if self.ui.show_skeleton_checkbox.isChecked():
                self.detect_kpt(img)
                self.start_pitch()

            self.update_frame(img)

    def video_analyze_frame(self):
        frame_num = self.ui.frame_slider.value()

        if len(self.record_buffer) == 0:
            return

        self.ui.frame_num_label.setText(f'{frame_num}/{len(self.record_buffer) - 1}')
        
        image = self.record_buffer[frame_num].copy()

        # if frame_num == len(self.record_buffer) - 1:
        #     self.ui.play_btn.click()
        #     save_video(self.video_name, self.record_buffer, self.person_df, select_id=self.select_person_id)

        if frame_num not in self.processed_frames:
            self.detect_kpt(image, frame_num= frame_num)
        
        # if self.select_person_id:
        #     self.import_data_to_table(self.select_person_id, frame_num)

        self.update_frame(image)

    def update_frame(self, image: np.ndarray):
        if not self.ui.record_checkbox.isChecked():
            image = draw_region(image)

        if not self.person_df.empty:
            if self.ui.show_skeleton_checkbox.isChecked():
                image = draw_points_and_skeleton(image, self.person_df, joints_dict()['haple']['skeleton_links'],
                                                points_color_palette='gist_rainbow', skeleton_palette_samples='jet',
                                                points_palette_samples=10, confidence_threshold=0.3)
            if self.ui.show_bbox_checkbox.isChecked():
                image = draw_bbox(self.person_df, image)
            if self.ui.select_keypoint_checkbox.isChecked():
                image = draw_traj(self.select_kpt_buffer,image)

        if self.ui.show_line_checkbox.isChecked():
            image = draw_grid(image)

        self.show_image(image, self.camera_scene, self.ui.frame_view)

    def smooth_kpt(self, person_ids: list):
        if self.pre_person_df.empty :
            self.pre_person_df = self.person_df.copy()

        if self.person_df.empty:
            return
        
        for person_id in person_ids: 
            pre_person_data = self.pre_person_df.loc[self.pre_person_df['person_id'] == person_id]
            curr_person_data = self.person_df.loc[self.person_df['person_id'] == person_id]
            
            if curr_person_data.empty or pre_person_data.empty:
                continue 
            
            pre_kpts = pre_person_data.iloc[0]['keypoints'] if not pre_person_data.empty else None
            curr_kpts = curr_person_data.iloc[0]['keypoints'] if not curr_person_data.empty else None
            
            smoothed_kpts = []
            
            if curr_kpts is not None and pre_kpts is not None:
                for pre_kpt, curr_kpt in zip(pre_kpts, curr_kpts): 
                    pre_kptx, pre_kpty = pre_kpt[0], pre_kpt[1]
                    curr_kptx, curr_kpty, curr_conf, curr_label = curr_kpt[0], curr_kpt[1], curr_kpt[2], curr_kpt[3]
                    
                    if all([pre_kptx != 0, pre_kpty != 0, curr_kptx != 0, curr_kpty != 0]):
                        curr_kptx = self.smooth_filter(curr_kptx, pre_kptx)
                        curr_kpty = self.smooth_filter(curr_kpty, pre_kpty)
                    
                    smoothed_kpts.append([curr_kptx, curr_kpty, curr_conf, curr_label])

                self.person_df.at[curr_person_data.index[0], 'keypoints'] = smoothed_kpts
        self.pre_person_df = self.person_df.copy()

    def mousePressEvent(self, event):
        pos = event.pos()
        scene_pos = self.ui.frame_view.mapToScene(pos)
        x, y = scene_pos.x(), scene_pos.y()
        if event.button() == Qt.LeftButton:
            if self.ui.select_checkbox.isChecked():
                self.person_id_selector(x, y)

            if self.ui.select_keypoint_checkbox.isChecked():
                self.kpt_id_selector(x, y)

    def person_id_selector(self, x: float, y: float):
        curr_person_df = None
        if len(self.record_buffer) != 0:
            curr_person_df = self.obtain_data(frame_num = self.ui.frame_slider.value())
        
        if self.pre_person_df.empty and curr_person_df.empty:
            return
        
        if curr_person_df is None:
            search_person_df = self.pre_person_df  
        else:
            search_person_df = curr_person_df


        selected_id = None
        max_area = -1

        for _, row in search_person_df.iterrows():
            person_id = row['person_id']
            x1, y1, x2, y2 = map(int, row['bbox'])
            if x != 0 and y != 0:
                if not (x1 <= x <= x2 and y1 <= y <= y2):
                    continue
            area = (x2 - x1) * (y2 - y1)

            if area > max_area:
                max_area = area
                selected_id = person_id

        self.select_person_id = selected_id

    def kpt_id_selector(self, x: float, y: float):
        def calculate_distance(point1, point2):
            return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        if self.pre_person_df.empty :
            return
        
        selected_id = None
        min_distance = float('inf')
        for _, row in self.pre_person_df.iterrows():
            person_kpts = row['keypoints']
            for kpt_id, kpt in enumerate(person_kpts):
                
                kptx, kpty, kpt_score, _= map(int, kpt)
        
                distance = calculate_distance([kptx, kpty], [x, y])
                if distance < min_distance:
                    min_distance = distance
                    selected_id = kpt_id

        self.select_kpt_id = selected_id

    def start_pitch(self):
        if self.select_person_id is None:
            print("No pitcher")
            return

        trigger_kpt_idx = 10 if self.ui.pitch_input.currentIndex() == 0 else 9
        person_kpt = self.obtain_data()

        if person_kpt is None:
            print("No kpt")
            return

        x, y, _, _ = person_kpt[trigger_kpt_idx]
        x1, y1 = self.region[0]
        x2, y2 = self.region[1]

        if x1 <= x <= x2 and y1 <= y <= y2: 
            if self.trigger_record_timer.start_time is None:
                print("In region")
                self.trigger_record_timer.start()
        else:
            self.trigger_record_timer.reset()
            

        if self.trigger_record_timer.is_time_up():
            self.trigger_pitch_timer.start()
            self.checkbox_controller(record=True, show_skeleton=False, show_bbox=False,
                                      select_person=False, select_kpt=False, show_kpt_angle= False)
            
    def obtain_data(self, frame_num = None, person_id = None, is_kpt = False):
        if self.person_df.empty:
            return None
        if frame_num is None:
            person_kpt = self.person_df.loc[self.person_df['person_id'] == self.select_person_id, 'keypoints']
            return person_kpt.to_numpy()[0] if not person_kpt.empty else None
        else:
            frame_data = self.person_df[self.person_df['frame_number'] == frame_num]
        
            if person_id is not None:
                frame_data = frame_data[frame_data['person_id'] == person_id]
            
            if is_kpt:
                frame_data = frame_data['keypoints']
            
            return frame_data

    def checkbox_controller(self, camera:bool = None, record:bool = None, show_skeleton:bool = None, 
                                show_bbox:bool = None, select_person:bool = None, select_kpt:bool = None, show_kpt_angle:bool = None):
        
        if camera is not None and camera != self.ui.camera_checkbox.isChecked():
            self.ui.camera_checkbox.click()

        if record is not None and record != self.ui.record_checkbox.isChecked():
            self.ui.record_checkbox.click()

        if show_skeleton is not None and show_skeleton != self.ui.show_skeleton_checkbox.isChecked():
            self.ui.show_skeleton_checkbox.click()

        if show_bbox is not None and show_bbox != self.ui.show_bbox_checkbox.isChecked():
            self.ui.show_bbox_checkbox.click()

        if select_person is not None and select_person != self.ui.select_checkbox.isChecked():
            self.ui.select_checkbox.click()

        if select_kpt is not None and select_kpt != self.ui.select_keypoint_checkbox.isChecked():
            self.ui.select_keypoint_checkbox.click()

        if show_kpt_angle is not None and show_kpt_angle != self.ui.show_kpt_angle_checkbox.isChecked():
            self.ui.show_kpt_angle_checkbox.click()

    def video_silder(self, visible:bool):
        elements = [
            self.ui.back_key_btn,
            self.ui.play_btn,
            self.ui.forward_key_btn,
            self.ui.frame_slider,
            self.ui.frame_num_label
        ]
        
        for element in elements:
            element.setVisible(visible)

    def video_ui_set(self, video):
        self.ui.frame_slider.setMinimum(0)
        self.ui.frame_slider.setMaximum(len(video) - 1)
        self.ui.frame_slider.setValue(0)
        self.ui.frame_num_label.setText(f'0/{len(video)-1}')
        image = video[0].copy()
        if self.ui.show_skeleton_checkbox.isChecked():
            self.detect_kpt(image, frame_num = 0)
        self.update_frame(image)
        self.ui.image_resolution_label.setText( "(0,0) -" + f" {video[0].shape[1]} x {video[0].shape[0]}")
        
    def detect_kpt(self, image:np.ndarray, frame_num:int = None):
        self.fps_timer.tic()
        pred_instances, person_ids = process_one_image(self.model, image, select_id=self.select_person_id)
        average_time = self.fps_timer.toc()
        fps = int(1/max(average_time, 0.00001))
        self.ui.fps_info_label.setText(f"{fps:02}")
        self.person_df = self.merge_person_data(pred_instances, person_ids, frame_num)
        self.smooth_kpt(person_ids)
        
        if frame_num is not None:
            self.processed_frames.add(frame_num)

    def play_btn_clicked(self):
        print("play btn click")
        if len(self.record_buffer) == 0:
            return
        self.is_play = not self.is_play
        if self.is_play:
            self.ui.play_btn.setText("||")
            self.play_frame(self.ui.frame_slider.value())
        else:
            self.ui.play_btn.setText("▶︎")

    def play_frame(self, start_num = 0):
        for i in range(start_num, len(self.record_buffer)):
            self.ui.frame_slider.setValue(i)
            if not self.is_play:
                break
            if i > self.processed_images:
                self.processed_images = i
            if i == len(self.record_buffer) - 1 and self.is_play:
                self.play_btn_clicked()
            cv2.waitKey(15)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PosePitchTabControl()
    window.show()
    sys.exit(app.exec_())
