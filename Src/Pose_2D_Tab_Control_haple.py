from PyQt5.QtWidgets import *
# from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtGui import QPainter, QPen, QColor, QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QPointF
import numpy as np
import sys
import cv2
import os
from UI_Control.UI import Ui_MainWindow
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
from argparse import ArgumentParser
import cv2
import numpy as np
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "UI_Control"))
sys.path.append(os.path.join(current_dir, "tracker"))
# print(sys.path)
from UI_Control.lib.cv_thread import VideoToImagesThread
from UI_Control.lib.util import DataType
from UI_Control.lib.analyze import obtain_analyze_information
from UI_Control.lib.timer import Timer
from UI_Control.lib.vis_image import draw_set_line, draw_analyze_infromation, draw_bbox, draw_butt_point,draw_butt_width
from UI_Control.lib.vis_pose import draw_points_and_skeleton, joints_dict
from UI_Control.lib.vis_graph import init_graph, update_graph
from UI_Control.Widget.store import Store_Widget
from UI_Control.topdown_demo_with_mmdet import process_one_image
from UI_Control.image_demo import detect_image
from mmcv.transforms import Compose
from mmengine.logging import print_log

from pathlib import Path
from tracker.tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer
from mmpose.apis import init_model as init_pose_estimator
from mmpose.utils import adapt_mmdet_pipeline
from UI_Control.lib.one_euro_filter import OneEuroFilter
import pyqtgraph as pg
# 設置背景和前景顏色

# from pyqtgraph import LabelItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


class Pose_2d_Tab_Control(QMainWindow):
    def __init__(self):
        super(Pose_2d_Tab_Control, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_var()
        self.bind_ui()
        self.add_parser()
        self.set_tracker_parser()
        self.init_model()

    def bind_ui(self):
        self.clear_table_view()
        self.ui.load_video_btn.clicked.connect(
            lambda: self.load_video(self.ui.video_label, self.db_path + "/videos/"))
        # self.ui.store_video_btn.clicked.connect(self.store_video)
        self.ui.play_btn.clicked.connect(self.play_btn_clicked)
        self.ui.back_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        )
        self.ui.forward_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        )
        self.ui.frame_slider.valueChanged.connect(self.analyze_frame)
        self.ui.correct_btn.clicked.connect(self.switch_kpt_data)
        self.ui.keypoint_table.cellActivated.connect(self.on_cell_clicked)
        self.ui.Frame_View.mousePressEvent = self.mousePressEvent
        self.ui.store_data_btn.clicked.connect(self.show_store_window) 
        self.ui.load_data_btn.clicked.connect(self.load_json)
        self.ui.start_code_btn.clicked.connect(self.start_runner_analyze)
        self.ui.set_length_btn.clicked.connect(self.set_length)
        self.ui.set_fps_btn.clicked.connect(self.set_frame_ratio)
        self.ui.start_analyze_btn.clicked.connect(self.start_runner_analyze)
        self.ui.id_correct_btn.clicked.connect(self.correct_person_id)
        self.ui.display_butt_width_btn.clicked.connect(self.dispaly_butt_width)
        self.ui.display_speed_btn.clicked.connect(self.set_jump_frame)

    def init_model(self):
        self.detector = init_detector(
        self.args.det_config, self.args.det_checkpoint, device=self.args.device)
        self.detector.cfg.test_dataloader.dataset.pipeline[
        0].type = 'mmdet.LoadImageFromNDArray'
        self.detector_test_pipeline = Compose(self.detector.cfg.test_dataloader.dataset.pipeline)
        self.pose_estimator = init_pose_estimator(
        self.args.pose_config,
        self.args.pose_checkpoint,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=self.args.draw_heatmap))))
        self.tracker = BoTSORT(self.tracker_args, frame_rate=30.0)
        self.smooth_tool = OneEuroFilter()
        self.timer = Timer()
        pg.setConfigOptions(foreground=QColor(113,148,116), antialias = True)

    def init_var(self):
        self.db_path = f"../../Db"
        self.is_play=False
        self.processed_images=-1
        self.fps = 30
        self.runner_analyze = False
        self.speed_range = [0,12]
        self.frame_ratio = 1/240
        self.video_images=[]
        self.video_path = ""
        self.is_threading=False
        self.video_scene = QGraphicsScene()
        self.stride_scene = QGraphicsScene()
        self.scene = QGraphicsScene()
        self.video_scene.clear()
        self.stride_scene.clear()
        self.scene.clear()
        self.correct_kpt_idx = 0
        self.video_name = ""
        self.length_ratio = 0
        self.start_frame_num = 0
        self.end_frame_num = 0
        self.line_pos = []
        self.is_set_length = False
        self.analyze_information = {}
        self.processed_frames = set()
        self.person_df = pd.DataFrame()
        self.person_data = []
        self.label_kpt = False
        self.select_id = 0
        self.select_frame = 0
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.graph =  pg.PlotWidget()
        self.kpts_dict = joints_dict()['haple']['keypoints']
        self.jump_frame = [0,0,0]
            
    def add_parser(self):
        self.parser = ArgumentParser()
        self.parser.add_argument('--det-config', default='mmyolo_main/yolov7_x_syncbn_fast_8x16b-300e_coco.py', help='Config file for detection')
        self.parser.add_argument('--det-checkpoint', default='../Db/pretrain/yolov7_x_syncbn_fast_8x16b-300e_coco_20221124_215331-ef949a68.pth', help='Checkpoint file for detection')
        self.parser.add_argument('--pose-config', default='mmpose_main/configs/body_2d_keypoint/topdown_heatmap/haple/ViTPose_base_simple_halpe_256x192.py', help='Config file for pose')
        self.parser.add_argument('--pose-checkpoint', default='../Db/pretrain/best_coco_AP_epoch_f9_8.pth', help='Checkpoint file for pose')
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
        parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
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

    def load_video(self, label_item, path):
        if self.is_play:
            QMessageBox.warning(self, "讀取影片失敗", "請先停止播放影片!")
            return
        self.init_var()
        self.video_path = self.load_data(
                label_item, path, None, DataType.VIDEO)       
        # no video found
        if self.video_path == "":
            return
        label_item.setText("讀取影片中...")
        #run image thread
        self.v_t=VideoToImagesThread(self.video_path)
        self.v_t.emit_signal.connect(self.video_to_frame)
        self.v_t.start()

    def load_data(self, label_item, dir_path="", value_filter=None, mode=DataType.DEFAULT):
        data_path = None
        if mode == DataType.FOLDER:
            data_path = QFileDialog.getExistingDirectory(
                self, mode.value['tips'], dir_path)
        else:
            name_filter = mode.value['filter'] if value_filter == None else value_filter
            data_path, _ = QFileDialog.getOpenFileName(
                None, mode.value['tips'], dir_path, name_filter)
        if label_item == None:
            return data_path
        # check exist
        if data_path:
            label_item.setText(os.path.basename(data_path))
            label_item.setToolTip(data_path)
        else:
            label_item.setText(mode.value['tips'])
            label_item.setToolTip("")
        return data_path  

    def show_image(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView): 
        scene.clear()
        image = image.copy()
        image = cv2.circle(image, (0, 0), 10, (0, 0, 255), -1)
        w, h = image.shape[1], image.shape[0]
        bytesPerline = 3 * w
        qImg = QImage(image, w, h, bytesPerline, QImage.Format_RGB888).rgbSwapped()   
        pixmap = QPixmap.fromImage(qImg)
        scene.addPixmap(pixmap)
        GraphicsView.setScene(scene)
        GraphicsView.setAlignment(Qt.AlignLeft)
        GraphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def show_graph(self, graph, scene, graphicview):
        graph.resize(graphicview.width(),graphicview.height())
        scene.addWidget(graph)
        graphicview.setScene(scene)
        graphicview.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def video_to_frame(self, video_images, fps, count):
        self.total_images = count
        self.ui.frame_slider.setMinimum(0)
        self.ui.frame_slider.setMaximum(count - 1)
        self.ui.frame_slider.setValue(0)
        self.ui.frame_num_label.setText(f'0/{count-1}')
        self.video_images=video_images
        self.show_image(self.video_images[0], self.video_scene, self.ui.Frame_View)
        self.graph = init_graph(self.ui.fps_input.value(), self.speed_range, self.total_images)
        self.show_graph(self.graph, self.scene, self.ui.stride_view)
        self.ui.image_resolution_label.setText( "(0,0) -" + f" {self.video_images[0].shape[1]} x {self.video_images[0].shape[0]}")
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.ui.video_label.setText(self.video_name)
        self.close_thread(self.v_t)
        self.fps = fps

    def close_thread(self, thread):
        thread.stop()
        thread = None
        self.is_threading=False
   
    def play_frame(self, start_num=0):
        for i in range(start_num, self.total_images):
            if not self.is_play:
                break
            if i > self.processed_images:
                self.processed_images = i
                # self.analyze_frame()
           
            self.ui.frame_slider.setValue(i)
            # to the last frame ,stop playing
            if i == self.total_images - 1 and self.is_play:
                self.play_btn_clicked()
            # time.sleep(0.1)
            cv2.waitKey(15)

    def merge_keypoint_datas(self, pred_instances):
        person_kpts = []
        for person in pred_instances:
            kpts = np.round(person['keypoints'][0], 2)
            kpt_scores = np.round(person['keypoint_scores'][0], 2)
            kpt_datas = np.hstack((kpts, kpt_scores.reshape(-1, 1)))     
            # Add a boolean value to each element in kpt_datas
            kpt_datas = np.hstack((kpt_datas, np.full((len(kpt_datas), 1), False, dtype=bool)))
            person_kpts.append(kpt_datas)
        return person_kpts

    def merge_person_datas(self, frame_num, person_ids, person_bboxes, person_kpts):
        for pid, bbox, kpts in zip(person_ids, person_bboxes, person_kpts):
            new_kpts = np.zeros((len(self.kpts_dict),kpts.shape[1]))
            # haple
            new_kpts[:26] = kpts
            new_kpts[26:, 2] = 0.9
            self.person_data.append({
                'frame_number': frame_num,
                'person_id': pid,
                'bbox': bbox,
                'keypoints': new_kpts
            })
        self.person_df = pd.DataFrame(self.person_data)
        # if not self.person_df.empty:
        #     print(self.person_df)
        #     exit()

    def play_btn_clicked(self):
        if self.video_path == "":
            QMessageBox.warning(self, "無法開始播放", "請先讀取影片!")
            return
        self.is_play = not self.is_play
        if self.is_play:
            self.ui.play_btn.setText("||")
            self.play_frame(self.ui.frame_slider.value())
        else:
            self.ui.play_btn.setText("▶︎")

    def update_person_df(self):
        person_id = self.select_id
        # 获取当前帧数
        frame_num = self.ui.frame_slider.value()
        # 获取表格中的数据并更新到 DataFrame 中
        for kpt_idx in range(self.ui.keypoint_table.rowCount()):
            kpt_name = self.ui.keypoint_table.item(kpt_idx, 0).text()
            kpt_x = float(self.ui.keypoint_table.item(kpt_idx, 1).text())
            kpt_y = float(self.ui.keypoint_table.item(kpt_idx, 2).text())
            # 更新 DataFrame 中对应的值
            self.person_df.loc[(self.person_df['frame_number'] == frame_num) &
                                (self.person_df['person_id'] == person_id), 'keypoints'].iloc[0][kpt_idx][:2] = [kpt_x, kpt_y]
        
        self.update_frame()

    def switch_kpt_data(self):
        right_leg_id = [12,14,16,21,23,25]
        left_leg_id = [11,13,15,20,22,24]
        curr_frame_num = self.ui.frame_slider.value()
        person_data = self.person_df.loc[(self.person_df['frame_number'] == curr_frame_num) &
                                (self.person_df['person_id'] == self.select_id)]
        if not person_data.empty :
            person_kpt = person_data.iloc[0]['keypoints']
            correct_kpt = []
            for i in range(len(person_kpt)): 
                kptx , kpty, conf, label = person_kpt[i][0], person_kpt[i][1], person_kpt[i][2], person_kpt[i][3]
                if i in left_leg_id:
                    temp_kptx , temp_kpt_y , temp_kpt_conf, temp_kpt_label = person_kpt[i+1][0], person_kpt[i+1][1], person_kpt[i+1][2], person_kpt[i+1][3]  
                    correct_kpt.append([temp_kptx , temp_kpt_y , temp_kpt_conf, temp_kpt_label])
                    correct_kpt.append([kptx,kpty,conf,label])            
                if i not in left_leg_id and i not in right_leg_id:
                    correct_kpt.append([kptx, kpty, conf, label])  # 设置可信度为默认值
            # 更新 DataFrame 中的数据

            self.person_df.at[person_data.index[0], 'keypoints'] = correct_kpt

            self.update_frame()

    def analyze_frame(self):
        frame_num = self.ui.frame_slider.value()

        self.ui.frame_num_label.setText(
            f'{frame_num}/{self.total_images - 1}')

        # no image to analyze
        if self.total_images <= 0:
            return
        
        ori_image=self.video_images[frame_num].copy()
        image = ori_image.copy()
    
        if self.ui.frame_slider.value() == (self.total_images-1):
            self.ui.play_btn.click()
        print(self.runner_analyze)
        if self.runner_analyze:
            self.analyze_person(frame_num)
        else:
            if frame_num not in self.processed_frames:
                self.detect_kpt(image, frame_num)
                print("detect")
        self.update_frame()
                 
    def update_frame(self):
        curr_person_df, frame_num= self.obtain_curr_data()
        image=self.video_images[frame_num].copy()
        
        image =  draw_set_line(image, self.line_pos)
        if len(self.analyze_information) > 0:
            jump_frame = [self.jump_frame[0]- self.start_frame_num, self.jump_frame[1]-self.start_frame_num, self.jump_frame[2] - self.start_frame_num]
           
            show_jump_speed = True  if self.ui.frame_slider.value() > self.jump_frame[1] and self.jump_frame[0:2] != [0,0] else False
            image = draw_analyze_infromation(image,self.analyze_information, jump_frame,
                                              show_jump_speed, self.length_ratio, self.frame_ratio)
        
        if not curr_person_df.empty and frame_num in self.processed_frames:
            #haple
            if self.ui.show_skeleton_checkBox.isChecked():
                image = draw_points_and_skeleton(image, curr_person_df, joints_dict()['haple']['skeleton_links'], 
                                                points_color_palette='gist_rainbow', skeleton_palette_samples='jet',
                                                points_palette_samples=10, confidence_threshold=0.3)
            if self.ui.show_bbox_checkbox.isChecked():
                image = draw_bbox(curr_person_df, image)

        if self.select_frame != 0:
            image = draw_butt_width(image,self.person_df.loc[(self.person_df['person_id'] == self.select_id)]['keypoints'],
                                    self.select_frame, self.length_ratio, self.analyze_information['run_side'])
            self.ui.toe_land_frame.setValue(self.select_frame + self.start_frame_num)
        # 将原始图像直接显示在 QGraphicsView 中
        self.show_image(image, self.video_scene, self.ui.Frame_View)

    def detect_kpt(self,image,frame_num):
        self.timer.tic()
        pred_instances, person_ids = process_one_image(self.args,image,self.detector,self.detector_test_pipeline,self.pose_estimator,self.tracker)
        average_time = self.timer.toc()
        fps= int(1/max(average_time,0.00001))
        if fps <10:
            self.ui.fps_label.setText(f"FPS: 0{fps}")
        else:
            self.ui.fps_label.setText(f"FPS: {fps}")
        # the keyscore > 1.0??
        person_kpts = self.merge_keypoint_datas(pred_instances)
        person_bboxes = pred_instances['bboxes']
        self.merge_person_datas(frame_num, person_ids, person_bboxes, person_kpts)
        self.processed_frames.add(frame_num)
        self.smooth_kpt(person_ids)

    def analyze_person(self,frame):
        person_kpt = pd.DataFrame()
        if self.select_id != 0:
            person_kpt = self.obtain_person_kpt()
            self.import_data_to_table(self.select_id, frame)
        if frame > self.start_frame_num:
            self.analyze_information, self.jump_frame[2] = obtain_analyze_information(person_kpt, frame,self.start_frame_num,
                                                                  self.length_ratio,self.frame_ratio, self.jump_frame,
                                                                  self.line_pos,self.end_frame_num) 

            self.graph = update_graph(self.graph,self.analyze_information)

    def obtain_curr_data(self):
        curr_person_df = pd.DataFrame()
        frame_num = self.ui.frame_slider.value()
        if not self.person_df.empty:
            curr_person_df = self.person_df.loc[(self.person_df['frame_number'] == frame_num)]
            # print(curr_person_df)
        return curr_person_df, frame_num
    
    def obtain_person_kpt(self):
        person_kpt = pd.DataFrame()
        if not self.person_df.empty:
            person_kpt = self.person_df.loc[(self.person_df['person_id'] == self.select_id)]['keypoints']
            # print(curr_person_df)
        return person_kpt

    def start_runner_analyze(self):
        if len(self.line_pos) < 1:
            QMessageBox.warning(self, "尚未設定長度", "請先設定長度!")
            return
        self.ui.frame_slider.setValue(0)
        self.runner_analyze = True
        if not self.person_df.empty:
            if len(self.person_df['person_id'].unique())> 0:
                self.select_id = self.person_df['person_id'].unique()[0]
                self.smooth_kpt([self.select_id])
                print(self.select_id)

    def import_data_to_table(self, person_id, frame_num):
        # 清空表格視圖
        self.clear_table_view()

        # 獲取特定人員在特定幀的數據
        person_data = self.person_df.loc[(self.person_df['frame_number'] == frame_num) & (self.person_df['person_id'] == person_id)]

        if person_data.empty:
            # print("未找到特定人員在特定幀的數據")
            self.clear_table_view()
            return

        # 確保表格視圖大小足夠
        num_keypoints = len(self.kpts_dict)
        if self.ui.keypoint_table.rowCount() < num_keypoints:
            self.ui.keypoint_table.setRowCount(num_keypoints)

        # 將關鍵點數據匯入到表格視圖中
        for kpt_idx, kpt in enumerate(person_data['keypoints'].iloc[0]): 
            kptx, kpty, kpt_label = kpt[0], kpt[1], kpt[3]
            kpt_name = self.kpts_dict[kpt_idx]
            kpt_name_item = QTableWidgetItem(str(kpt_name))
            kptx_item = QTableWidgetItem(str(np.round(kptx,1)))
            kpty_item = QTableWidgetItem(str(np.round(kpty,1)))
            if kpt_label :
                kpt_label_item = QTableWidgetItem("Y")
            else:
                kpt_label_item = QTableWidgetItem("N")
            kpt_name_item.setTextAlignment(Qt.AlignRight)
            kptx_item.setTextAlignment(Qt.AlignRight)
            kpty_item.setTextAlignment(Qt.AlignRight)
            kpt_label_item.setTextAlignment(Qt.AlignRight)
            self.ui.keypoint_table.setItem(kpt_idx, 0, kpt_name_item)
            self.ui.keypoint_table.setItem(kpt_idx, 1, kptx_item)
            self.ui.keypoint_table.setItem(kpt_idx, 2, kpty_item)
            self.ui.keypoint_table.setItem(kpt_idx, 3, kpt_label_item)

    def clear_table_view(self):
        # 清空表格視圖
        self.ui.keypoint_table.clear()
        # 設置列數
        self.ui.keypoint_table.setColumnCount(4)
        # 設置列標題
        title = ["Keypoint", "X", "Y", "有無更改"]
        self.ui.keypoint_table.setHorizontalHeaderLabels(title)
        # 將列的對齊方式設置為左對齊
        header = self.ui.keypoint_table.horizontalHeader()
        for i in range(4):
            header.setDefaultAlignment(Qt.AlignLeft)

    def on_cell_clicked(self, row, column):
        self.correct_kpt_idx = row
        self.label_kpt = True
    
    def send_to_table(self, kptx, kpty, kpt_label):
    
        kptx_item = QTableWidgetItem(str(kptx))
        kpty_item = QTableWidgetItem(str(kpty))
        if kpt_label :
            kpt_label_item = QTableWidgetItem("Y")
        else:
            kpt_label_item = QTableWidgetItem("N")
        kptx_item.setTextAlignment(Qt.AlignRight)
        kpty_item.setTextAlignment(Qt.AlignRight)
        kpt_label_item.setTextAlignment(Qt.AlignRight)
        self.ui.keypoint_table.setItem(self.correct_kpt_idx, 1, kptx_item)
        self.ui.keypoint_table.setItem(self.correct_kpt_idx, 2, kpty_item)
        self.ui.keypoint_table.setItem(self.correct_kpt_idx, 3, kpt_label_item)
        self.update_person_df()

    def set_length(self):
        self.is_set_length = True
        self.line_pos = []
        self.real_length = self.ui.length_input.value()/100

    def mousePressEvent(self, event):
        if self.label_kpt:
            pos = event.pos()
            scene_pos = self.ui.Frame_View.mapToScene(pos)
            kptx, kpty = scene_pos.x(), scene_pos.y()
            kpt_label = 1
            if event.button() == Qt.LeftButton:
                self.send_to_table(kptx, kpty,kpt_label)
            elif event.button() == Qt.RightButton:
                kptx, kpty = 0, 0
                self.send_to_table(kptx, kpty, 0)
            self.label_kpt = False
        
        if self.is_set_length:
            pos = event.pos()
            scene_pos = self.ui.Frame_View.mapToScene(pos)
            pos_x, pos_y = scene_pos.x(), scene_pos.y()
            pos = [pos_x, pos_y]
            self.line_pos = np.append(self.line_pos, pos)
            self.update_frame()
            if len(self.line_pos) == 4:
                pos_f = np.array([self.line_pos[0], self.line_pos[1]])
                pos_s = np.array([self.line_pos[2], self.line_pos[3]])
                length = np.linalg.norm(pos_f - pos_s)
                self.length_ratio = self.real_length/length
                print(self.line_pos)
                print(self.length_ratio)
                self.is_set_length = False
                QMessageBox.information(self, "設定長度", "設定長度完成!")
    
    def smooth_kpt(self,person_ids):
        for person_id in person_ids:
            # person_id = int(self.ui.ID_selector.currentText())
            person_kpt = self.person_df.loc[(self.person_df['person_id'] == person_id)]['keypoints']
            if len(person_kpt) > 0 and self.start_frame_num ==0 :
                self.start_frame_num = self.ui.frame_slider.value()
            if self.start_frame_num != 0:
                curr_frame = self.ui.frame_slider.value()
                if curr_frame == 0:
                    pre_frame_num = 0
                else:
                    pre_frame_num = curr_frame - 1
            pre_person_data = self.person_df.loc[(self.person_df['frame_number'] == pre_frame_num) &
                                                (self.person_df['person_id'] == person_id)]
            curr_person_data = self.person_df.loc[(self.person_df['frame_number'] == curr_frame) &
                                                (self.person_df['person_id'] == person_id)]
            if not curr_person_data.empty and not pre_person_data.empty:
                pre_kpts = pre_person_data.iloc[0]['keypoints']
                curr_kpts = curr_person_data.iloc[0]['keypoints']
                smoothed_kpts = []
                for pre_kpt, curr_kpt in zip(pre_kpts, curr_kpts): 
                    pre_kptx, pre_kpty = pre_kpt[0], pre_kpt[1]
                    curr_kptx , curr_kpty, curr_conf, curr_label = curr_kpt[0], curr_kpt[1], curr_kpt[2], curr_kpt[3]
                    if pre_kptx != 0 and pre_kpty != 0 and curr_kptx != 0 and curr_kpty !=0:
                 
                        curr_kptx = self.smooth_tool(curr_kptx, pre_kptx)
                        curr_kpty = self.smooth_tool(curr_kpty, pre_kpty)
                    smoothed_kpts.append([curr_kptx, curr_kpty, curr_conf, curr_label])  # 设置可信度为默认值
                # 更新 DataFrame 中的数据
                self.person_df.at[curr_person_data.index[0], 'keypoints'] = smoothed_kpts

    def show_store_window(self):
        if self.person_df.empty:
            print("no data")
            return
        else:
            self.store_window = Store_Widget(self.video_name, self.video_images, self.person_df)
            self.store_window.show()

    def keyPressEvent(self, event):
            if event.key() == ord('D') or event.key() == ord('d'):
                self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
            elif event.key() == ord('A') or event.key() == ord('a'):
                self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
            else:
                super().keyPressEvent(event)

    def load_json(self):
        # 打開文件對話框以選擇 JSON 文件
        json_path, _ = QFileDialog.getOpenFileName(
            self, "选择要加载的 JSON 文件", "", "JSON 文件 (*.json)")

        # 如果用戶取消選擇文件，則返回
        if not json_path:
            return

        try:
            # 从 JSON 文件中读取数据并转换为 DataFrame
            self.person_df = pd.DataFrame()
            self.person_df = pd.read_json(json_path)
            process_frame_nums = self.person_df['frame_number'].unique()
            self.processed_frames = sorted(set(frame_num for frame_num in process_frame_nums if frame_num!=0) )
            self.start_frame_num = min(self.processed_frames) - 1
            self.end_frame_num = max(self.processed_frames) - 1

        except Exception as e:
            print(f"加载 JSON 文件时出错：{e}")

    def set_frame_ratio(self):
        self.frame_ratio = 1 / self.ui.fps_input.value()
        print(self.frame_ratio)
        font = QFont()
        font.setPixelSize(15)
        # self.graph.setLabel('bottom', f'Frame (fps: {self.ui.fps_input.value()})')

        self.graph.setLabel('bottom', f'<span style = "font-size: 18px" >幀 (fps: {self.ui.fps_input.value()})</span>')
        self.graph.getAxis("bottom").setStyle(tickFont=font)    
       
    def correct_person_id(self):
        # 檢查人員DataFrame是否為空
        if self.person_df.empty:
            return

        # 獲取要更正的人員ID和更正後的人員ID
        before_correct_id = self.ui.before_correct_id.value()
        after_correct_id = self.ui.after_correct_id.value()

        # 確保要更正的人員ID和更正後的人員ID都在DataFrame中存在
        if (before_correct_id not in self.person_df['person_id'].unique()) or (after_correct_id not in self.person_df['person_id'].unique()):

            return

        # 獲取當前帧數
        frame_num = self.ui.frame_slider.value()

        # 遍歷從當前帧數到最大處理帧數的範圍
        for i in range(frame_num, max(self.processed_frames)):
            # 尋找要交換的行
            condition_1 = (self.person_df['frame_number'] == i) & (self.person_df['person_id'] == before_correct_id)
            condition_2 = (self.person_df['frame_number'] == i) & (self.person_df['person_id'] == after_correct_id)

            # 交換 remapped_id
            self.person_df.loc[condition_1, 'person_id'] = after_correct_id
            self.person_df.loc[condition_2, 'person_id'] = before_correct_id

        # 更新畫面
        self.update_frame()

    def dispaly_butt_width(self):
        self.select_frame = self.ui.frame_slider.value() - self.start_frame_num
        self.update_frame()

    def set_jump_frame(self):
        self.jump_frame = [self.ui.toe_land_frame.value() , self.ui.toe_off_frame.value(), 0]
        self.ui.frame_slider.setValue(self.ui.toe_land_frame.value())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Pose_2d_Tab_Control()
    window.show()
    sys.exit(app.exec_())
