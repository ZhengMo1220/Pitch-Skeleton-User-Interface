from PyQt5.QtWidgets import *
# from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtGui import QPainter, QPen, QColor, QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QPointF, QTimer
import numpy as np
import sys
import cv2
import os
from UI import Ui_MainWindow
import matplotlib.pyplot as plt
import pandas as pd
import queue
from argparse import ArgumentParser
from argparse import ArgumentParser
import cv2
import numpy as np
from lib.cv_thread import VideoCaptureThread
from lib.util import DataType
from lib.analyze import obtain_analyze_information
from lib.timer import Timer
from lib.vis_image import draw_set_line, draw_analyze_infromation, draw_bbox, draw_butt_point,draw_butt_width
from lib.vis_pose import draw_points_and_skeleton, joints_dict
from lib.vis_graph import init_graph, update_graph
from Widget.store import Store_Widget
from topdown_demo_with_mmdet import process_one_image
from image_demo import detect_image
from mmcv.transforms import Compose
from mmengine.logging import print_log
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加相對於當前文件的父目錄的子目錄到系統路徑
sys.path.append(os.path.join(current_dir, "..", "tracker"))
from pathlib import Path
from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer
from mmpose.apis import init_model as init_pose_estimator
from mmpose.utils import adapt_mmdet_pipeline
from lib.one_euro_filter import OneEuroFilter
import pyqtgraph as pg
# 設置背景和前景顏色

# from pyqtgraph import LabelItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

class Pose2DTabControl(QMainWindow):
   def __init__(self):
      super(Pose2DTabControl, self).__init__()
      self.ui = Ui_MainWindow()
      self.ui.setupUi(self)
      self.init_var()
      self.bind_ui()
      self.add_parser()
      self.set_tracker_parser()
      self.init_model()

   def bind_ui(self):
      self.ui.open_camera_btn.clicked.connect(self.toggle_camera)
      self.ui.Frame_View.mousePressEvent = self.mousePressEvent

   def init_model(self):
      self.detector = init_detector(
         self.args.det_config, self.args.det_checkpoint, device=self.args.device)
      self.detector.cfg.test_dataloader.dataset.pipeline[
         0].type = 'mmdet.LoadImageFromNDArray'
      self.detector_test_pipeline = Compose(self.detector.cfg.test_dataloader.dataset.pipeline)
      self.pose_estimator = init_pose_estimator(
         self.args.pose_config,
         self.args.pose_checkpoint,
         cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=self.args.draw_heatmap)))
      )
      self.tracker = BoTSORT(self.tracker_args, frame_rate=30.0)
      pg.setConfigOptions(foreground=QColor(113, 148, 116), antialias=True)
      self.timer = Timer()

   def init_var(self):
      self.db_path = f"../../Db"
      self.is_opened = False
      self.video_scene = QGraphicsScene()
      # self.video_thread = VideoCaptureThread()
      self.person_df = pd.DataFrame()
      self.frame_buffer = queue.Queue(maxsize=1)
      self.kpts_dict = joints_dict()['haple']['keypoints']
            
   def add_parser(self):
      self.parser = ArgumentParser()
      self.parser.add_argument('--det-config', default='../mmyolo_main/yolov7_x_syncbn_fast_8x16b-300e_coco.py', help='Config file for detection')
      self.parser.add_argument('--det-checkpoint', default='../../Db/pretrain/yolov7_x_syncbn_fast_8x16b-300e_coco_20221124_215331-ef949a68.pth', help='Checkpoint file for detection')
      self.parser.add_argument('--pose-config', default='../mmpose_main/configs/body_2d_keypoint/topdown_heatmap/haple/ViTPose_base_simple_halpe_256x192.py', help='Config file for pose')
      self.parser.add_argument('--pose-checkpoint', default='../../Db/pretrain/best_coco_AP_epoch_f9_8.pth', help='Checkpoint file for pose')
      self.parser.add_argument(
      '--device', default='cuda:0', help='Device used for inference')
      self.parser.add_argument(
      '--det-cat-id',
      type=int,
      default=0,
      help='Category id for bounding box detection model')
      self.parser.add_argument(
         '--score-thr', type=float, default=0.3, help='Bbox score threshold')
      self.parser.add_argument(
         '--nms-thr',
         type=float,
         default=0.3,
         help='IoU threshold for bounding box NMS')
      self.parser.add_argument(
         '--kpt-thr',
         type=float,
         default=0.3,
         help='Visualizing keypoint thresholds')
      self.parser.add_argument(
         '--draw-heatmap',
         action='store_true',
         default=False,
         help='Draw heatmap predicted by the model')
      self.parser.add_argument(
         '--show-kpt-idx',
         action='store_true',
         default=False,
         help='Whether to show the index of keypoints')
      self.parser.add_argument(
         '--skeleton-style',
         default='mmpose',
         type=str,
         choices=['mmpose', 'openpose'],
         help='Skeleton style selection')
      self.parser.add_argument(
         '--radius',
         type=int,
         default=3,
         help='Keypoint radius for visualization')
      self.parser.add_argument(
         '--thickness',
         type=int,
         default=1,
         help='Link thickness for visualization')
      self.parser.add_argument(
         '--show-interval', type=int, default=0, help='Sleep seconds per frame')
      self.parser.add_argument(
         '--alpha', type=float, default=0.8, help='The transparency of bboxes')
      self.parser.add_argument(
         '--draw-bbox', action='store_true', help='Draw bboxes of instances')
      self.args = self.parser.parse_args()

   def set_tracker_parser(self):
      parser = ArgumentParser()
      # tracking args
      parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
      parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
      parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
      parser.add_argument("--track_buffer", type=int, default=360, help="the frames for keep lost tracks")
      parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
      parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                           help="threshold for filtering out boxes of which aspect ratio are above the given value.")
      parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
      parser.add_argument("--fuse-score", dest="mot20", default=True, action='store_true',
                           help="fuse score and iou for association")

      # CMC
      parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

      # ReID
      parser.add_argument("--with-reid", dest="with_reid", default=False , help="with ReID module.")
      parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                           type=str, help="reid config file path")
      parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                           type=str, help="reid config file path")
      parser.add_argument('--proximity_thresh', type=float, default=0.5,
                           help='threshold for rejecting low overlap reid matches')
      parser.add_argument('--appearance_thresh', type=float, default=0.25,
                           help='threshold for rejecting low appearance similarity reid matches')

      self.tracker_args = parser.parse_args()

      self.tracker_args.jde = False
      self.tracker_args.ablation = False

   def toggle_camera(self):
      if self.is_opened:
         self.close_camera()
      else:
         self.open_camera()

   def open_camera(self):
      self.video_thread = VideoCaptureThread()
      self.video_thread.frame_ready.connect(self.buffer_frame)
      self.video_thread.start_capture()
      self.is_opened = True

   def buffer_frame(self, frame):
        if not self.frame_buffer.full():
            self.frame_buffer.put(frame)
            self.analyze_frame()

   def close_camera(self):
      self.video_thread.stop_capture()
      self.video_scene.clear()
      self.is_opened = False

   # def closeEvent(self, event):
   #    self.video_thread.stop_capture()
   #    event.accept()

   def show_image(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView):
      scene.clear()
      image = cv2.circle(image.copy(), (0, 0), 10, (0, 0, 255), -1)
      h, w = image.shape[:2]
      qImg = QImage(image, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
      pixmap = QPixmap.fromImage(qImg)
      scene.addPixmap(pixmap)
      GraphicsView.setScene(scene)
      GraphicsView.setAlignment(Qt.AlignLeft)
      GraphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

   def merge_keypoint_datas(self, pred_instances):
      person_kpts = []
      for person in pred_instances:
         kpts = np.round(person['keypoints'][0], 2)
         kpt_scores = np.round(person['keypoint_scores'][0], 2)
         kpt_datas = np.hstack((kpts, kpt_scores.reshape(-1, 1)))
         kpt_datas = np.hstack((kpt_datas, np.full((len(kpt_datas), 1), False, dtype=bool)))
         person_kpts.append(kpt_datas)
      return person_kpts

   def merge_person_datas(self, person_ids, person_bboxes, person_kpts):
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
         self.timer.tic()
         frame = self.frame_buffer.get()
         average_time =self.timer.toc()
         # fps= int(1/max(average_time,0.00001))
         print("Frame capture time: "+str(average_time))
         # 此處添加幀處理邏輯，例如分析或顯示影像
         pred_instances, person_ids = process_one_image(self.args, frame, self.detector, self.detector_test_pipeline, self.pose_estimator, self.tracker)
         
         person_kpts = self.merge_keypoint_datas(pred_instances)
         person_bboxes = pred_instances['bboxes']
         self.merge_person_datas(person_ids, person_bboxes, person_kpts)
      self.update_frame(frame)

   def update_frame(self, image):
      self.timer.tic()
      if not self.person_df.empty:
         if self.ui.show_skeleton_checkBox.isChecked():
            image = draw_points_and_skeleton(image, self.person_df, joints_dict()['haple']['skeleton_links'],
                                             points_color_palette='gist_rainbow', skeleton_palette_samples='jet',
                                             points_palette_samples=10, confidence_threshold=0.3)
         if self.ui.show_bbox_checkbox.isChecked():
            image = draw_bbox(self.person_df, image)
      average_time =self.timer.toc()
      print("Draw time: "+str(average_time))
      # for bbox, id in zip(bboxes,person_ids):
      #    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
      #    # color = tuple(colors[id % len(colors)])
      #    color = (0,255,0)
      #    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
      #    image = cv2.putText(image, str(id), (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 1.5, color, 2)
      self.timer.tic()
      self.show_image(image, self.video_scene, self.ui.Frame_View)
      average_time =self.timer.toc()
      print("Show time: "+str(average_time))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Pose2DTabControl()
    window.show()
    sys.exit(app.exec_())
