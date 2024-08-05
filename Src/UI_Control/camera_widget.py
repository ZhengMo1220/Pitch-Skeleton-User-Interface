from PyQt5.QtWidgets import *
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtCore import Qt
import numpy as np
import sys
import cv2
import os
from threading import Lock
from camera_ui import Ui_camera_ui
import pandas as pd
import queue
from argparse import ArgumentParser
from lib.cv_thread import VideoCaptureThread, VideoWriter
from datetime import datetime
from lib.timer import Timer
from lib.vis_image import draw_grid, draw_bbox
from lib.vis_pose import draw_points_and_skeleton, joints_dict
from lib.set_parser import set_detect_parser, set_tracker_parser
from topdown_demo_with_mmdet import process_one_image
from image_demo import detect_image
from mmcv.transforms import Compose
from collections import deque
from mmengine.logging import print_log
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "tracker"))
from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer
from mmpose.apis import init_model as init_pose_estimator
from mmpose.utils import adapt_mmdet_pipeline
from lib.one_euro_filter import OneEuroFilter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

class PoseCameraTabControl(QWidget):
    def __init__(self):
        super(PoseCameraTabControl, self).__init__()
        self.ui = Ui_camera_ui()
        self.ui.setupUi(self)
        self.init_var()
        self.bind_ui()
        self.init_model()
        self.video_writer = None  # 初始化视频写入器变量

    def bind_ui(self):
        self.ui.open_camera_btn.clicked.connect(self.toggle_camera)
        self.ui.start_code_btn.clicked.connect(self.toggle_analyze)
        self.ui.record_btn.clicked.connect(self.toggle_record)
        
    def init_model(self):
        self.detector = init_detector(
            self.detect_args.det_config, self.detect_args.det_checkpoint, device=self.detect_args.device)
        self.detector.cfg.test_dataloader.dataset.pipeline[
            0].type = 'mmdet.LoadImageFromNDArray'
        self.detector_test_pipeline = Compose(self.detector.cfg.test_dataloader.dataset.pipeline)
        self.pose_estimator = init_pose_estimator(
            self.detect_args.pose_config,
            self.detect_args.pose_checkpoint,
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=self.detect_args.draw_heatmap)))
        )
        self.tracker = BoTSORT(self.tracker_args, frame_rate=30.0)
        self.smooth_filter = OneEuroFilter()
        self.timer = Timer()

    def init_var(self):
        self.db_path = f"../../Db"
        self.is_opened = False
        self.is_analyze = False
        self.is_record = False
        self.pre_person_df = pd.DataFrame()
        self.camera_scene = QGraphicsScene()
        self.person_df = pd.DataFrame()
        self.frame_buffer = queue.Queue(maxsize=1)
        self.kpts_dict = joints_dict()['haple']['keypoints']
        self.detect_args = set_detect_parser()
        self.tracker_args = set_tracker_parser()

    def toggle_camera(self):
        if self.is_opened:
            self.close_camera()
            self.ui.open_camera_btn.setText("開啟相機")
        else:
            self.open_camera()
            self.ui.open_camera_btn.setText("關閉相機")
    
    def toggle_record(self):
        if self.is_record:
            self.stop_recording()
            self.ui.record_btn.setText("開始錄影")
        else:
            self.start_recording()
            self.ui.record_btn.setText("停止錄影")

    def toggle_analyze(self):
        if self.is_analyze:
            self.is_analyze = False
            self.ui.start_code_btn.setText("開始分析")
        else:
            self.is_analyze = True
            self.ui.start_code_btn.setText("停止分析")

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
        if not self.frame_buffer.full():
            self.frame_buffer.put(frame)
            self.analyze_frame()
        if self.video_writer is not None:
            self.video_writer.write(frame)
    
    def start_recording(self):
        output_dir = f'../../Db/record/'
        os.makedirs(output_dir, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(output_dir, f'C{self.ui.camera_id_input.value()}_{current_time}.mp4')
        frame_width = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_writer = VideoWriter(video_filename, frame_width, frame_height)
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
            if self.is_analyze:
                self.timer.tic()
                pred_instances, person_ids = process_one_image(self.detect_args, frame, self.detector, self.detector_test_pipeline, self.pose_estimator, self.tracker)
                average_time = self.timer.toc()
                fps= int(1/max(average_time,0.00001))
                if fps <10:
                    self.ui.fps_label.setText(f"FPS: 0{fps}")
                else:
                    self.ui.fps_label.setText(f"FPS: {fps}")
                person_kpts = self.merge_keypoint_datas(pred_instances)
                person_bboxes = pred_instances['bboxes']
                self.merge_person_datas(person_ids, person_bboxes, person_kpts)
                self.smooth_kpt(person_ids)
            self.update_frame(frame)

    def update_frame(self, image:np.ndarray):
        if not self.person_df.empty:
            if self.ui.show_skeleton_checkBox.isChecked():
                image = draw_points_and_skeleton(image, self.person_df, joints_dict()['haple']['skeleton_links'],
                                                points_color_palette='gist_rainbow', skeleton_palette_samples='jet',
                                                points_palette_samples=10, confidence_threshold=0.3)
            if self.ui.show_bbox_checkbox.isChecked():
                image = draw_bbox(self.person_df, image)
        if self.ui.show_line_checkBox.isChecked():
            image = draw_grid(image)
        self.show_image(image, self.camera_scene, self.ui.camer_frame_view)
        self.person_df = pd.DataFrame()

    def smooth_kpt(self, person_ids):
        if self.pre_person_df.empty or self.person_df.empty:
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PoseCameraTabControl()
    window.show()
    sys.exit(app.exec_())
