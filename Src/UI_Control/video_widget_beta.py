from PyQt5.QtWidgets import *
# from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtGui import QPainter, QPen, QColor, QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QPointF, QTimer
import numpy as np
import sys
import cv2
import os
from video_ui import Ui_video_widget
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
from utils.vis_image import ImageDrawer
from utils.selector import Person_selector, Kpt_selector
from Camera.camera_control import VideoLoader, JsonLoader
from skeleton.detect_skeleton import PoseEstimater
from utils.vis_graph import GraphPlotter
from utils.analyze import PoseAnalyzer
from utils.model import Model
import pyqtgraph as pg

class PoseVideoTabControl(QWidget):
    def __init__(self, model:Model, parent = None):
        super(PoseVideoTabControl, self).__init__(parent)
        self.ui = Ui_video_widget()
        self.ui.setupUi(self)
        self.model = model
        self.person_selector = Person_selector()
        self.kpt_selector = Kpt_selector()
        self.pose_estimater = PoseEstimater(self.model)
        self.kpt_dict = self.pose_estimater.joints["haple"]["keypoints"]
        self.pose_analyzer = PoseAnalyzer(self.pose_estimater)
        self.graph_plotter = GraphPlotter(self.pose_analyzer)
        self.image_drawer = ImageDrawer(self.pose_estimater, self.pose_analyzer)
        self.video_loader = VideoLoader(self.image_drawer)
        self.init_var()
        self.bind_ui()
        
    def bind_ui(self):
        self.ui.load_original_video_btn.clicked.connect(
            lambda: self.load_video(is_processed=False))
        self.ui.load_processed_video_btn.clicked.connect(
            lambda: self.load_video(is_processed=True))
        
        self.ui.play_btn.clicked.connect(self.play_btn_clicked)
        self.ui.back_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        )
        self.ui.forward_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        )
        self.ui.frame_slider.valueChanged.connect(self.analyze_frame)
        self.ui.keypoint_table.cellActivated.connect(self.on_cell_clicked)
        self.ui.frame_view.mousePressEvent = self.mousePressEvent
        self.ui.id_correct_btn.clicked.connect(self.correct_id)
        self.ui.start_code_btn.clicked.connect(self.toggle_detect)
        self.ui.select_checkbox.stateChanged.connect(self.toggle_select)
        self.ui.show_skeleton_checkbox.stateChanged.connect(self.toggle_show_skeleton)
        self.ui.select_kpt_checkbox.stateChanged.connect(self.toggle_kpt_select)
        self.ui.show_bbox_checkbox.stateChanged.connect(self.toggle_show_bbox)
        self.ui.show_angle_checkbox.stateChanged.connect(self.toggle_show_angle_info)

    def init_var(self):
        self.is_play = False
        self.video_scene = QGraphicsScene()
        self.curve_scene = QGraphicsScene()
        self.video_scene.clear()
        self.curve_scene.clear()
        self.correct_kpt_idx = 0
        self.label_kpt = False
        pg.setConfigOptions(foreground=QColor(113,148,116), antialias = True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
     
    def load_video(self,is_processed:bool = False):
        if self.is_play:
            self.ui.play_btn.click()
        
        self.reset()
        self.video_loader.load_video()
        self.check_video_load()

        if is_processed:
            json_loader = JsonLoader(self.video_loader.folder_path, self.video_loader.video_name)
            person_df = json_loader.load()
            self.pose_estimater.set_processed_data(person_df)

            
    def reset(self):
        self.init_var()
        self.pose_estimater.reset()
        self.image_drawer.reset()
        self.person_selector.reset()
        self.kpt_selector.reset()
        self.pose_analyzer.reset()
        self.graph_plotter.reset()
        self.image_drawer.reset()
        self.video_loader.reset()

    def check_video_load(self):
        if self.video_loader.video_name is None:
            return
        if self.video_loader.is_loading:
            # 影片正在讀取中，稍後再檢查
            QTimer.singleShot(100, self.check_video_load)  # 每100ms檢查一次
            self.ui.video_name_label.setText("讀取影片中")
        else:
            # 影片讀取完成，執行後續操作
            self.ui.video_name_label.setText(self.video_loader.video_name)
            self.ui.frame_slider.setMinimum(0)
            self.ui.frame_slider.setMaximum(self.video_loader.total_frames - 1)
            self.ui.frame_slider.setValue(0)
            self.ui.frame_num_label.setText(f'0/{self.video_loader.total_frames- 1}')
            self.ui.video_resolution_label.setText( "(0,0) -" + f" {self.video_loader.video_size[0]} x {self.video_loader.video_size[1]}")
            self.graph_plotter._init_graph(self.video_loader.total_frames-1)
            self.model.set_image_size(self.video_loader.video_size)
            self.show_graph(self.curve_scene, self.ui.curve_view)
            self.update_frame(0)

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

    def show_graph(self, scene, graphicview):
        graph = self.graph_plotter._get_update_graph()
        graph.resize(graphicview.width(),graphicview.height())
        scene.addWidget(graph)
        graphicview.setScene(scene)
        graphicview.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
   
    def play_frame(self, start_num=0):
        for i in range(start_num, self.video_loader.total_frames):
            if not self.is_play:
                break
            self.ui.frame_slider.setValue(i)
            if i == self.video_loader.total_frames - 1 and self.is_play:
                self.play_btn_clicked()
            cv2.waitKey(15)

    def play_btn_clicked(self):
        if self.video_loader.video_name == "":
            QMessageBox.warning(self, "無法播放影片", "請讀取影片!")
            return
        if self.video_loader.is_loading:
            QMessageBox.warning(self, "影片讀取中", "請稍等!")
            return
        self.is_play = not self.is_play
        self.ui.play_btn.setText("||" if self.is_play else "▶︎")
        if self.is_play:
            self.play_frame(self.ui.frame_slider.value())

    def analyze_frame(self):
        frame_num = self.ui.frame_slider.value()
        self.ui.frame_num_label.setText(f'{frame_num}/{len(self.video_loader.video_frames) - 1}')
        image = self.video_loader.get_video_image(frame_num)
        _, _, fps= self.pose_estimater.detect_kpt(image,frame_num)
        self.ui.fps_info_label.setText(f"{fps:02d}")
        if self.pose_estimater.person_id is not None:
            if frame_num == 0:
                exit()
            self.pose_analyzer.add_analyze_info(frame_num)
            self.graph_plotter.update_graph(frame_num)
        self.import_data_to_table(frame_num)
        self.update_frame(self.ui.frame_slider.value())
        if frame_num == self.video_loader.total_frames - 1:
            self.ui.play_btn.click()
            self.video_loader.save_video()
                
    def update_frame(self, frame_num:int):
        # frame_num = self.ui.frame_slider.value()
        image = self.video_loader.get_video_image(frame_num)
        drawed_img = self.image_drawer.draw_info(image, frame_num, self.pose_estimater.kpt_buffer)
        self.show_image(drawed_img, self.video_scene, self.ui.frame_view)
    
    def toggle_detect(self):
        self.ui.show_skeleton_checkbox.setChecked(True)
        image = self.video_loader.get_video_image(0)
        _, _, fps= self.pose_estimater.detect_kpt(image, 0)
        self.ui.select_checkbox.setChecked(True)
        self.ui.select_kpt_checkbox.setChecked(True)
        self.ui.show_angle_checkbox.setChecked(True)
        self.ui.play_btn.click()

    def toggle_select(self, state:int):
        if state == 2: 
            self.person_selector.select(search_person_df=self.pose_estimater.get_person_df_data(frame_num=self.ui.frame_slider.value()))
            self.pose_estimater.set_person_id(self.person_selector.selected_id)
        else:
            self.pose_estimater.set_person_id(None)
        self.update_frame(self.ui.frame_slider.value())

    def toggle_kpt_select(self, state:int):
        if state == 2:  
            self.pose_estimater.set_kpt_id(9)
            self.image_drawer.set_show_traj(True)
        else:
            self.pose_estimater.set_kpt_id(None)
            self.image_drawer.set_show_traj(False)

    def toggle_show_skeleton(self, state:int):
        if state == 2:  
            self.pose_estimater.set_detect(True)
            self.image_drawer.set_show_skeleton(True)
        else:
            self.pose_estimater.set_detect(False)
            self.image_drawer.set_show_skeleton(False)

    def toggle_show_bbox(self, state:int):
        if state == 2:  
            self.image_drawer.set_show_bbox(True)
        else:
            self.image_drawer.set_show_bbox(False)

    def toggle_show_angle_info(self, state:int):
        if state == 2:  
            self.image_drawer.set_show_angle_info(True)
        else:
            self.image_drawer.set_show_angle_info(False)

    def clear_table_view(self):
        self.ui.keypoint_table.clear()
        self.ui.keypoint_table.setColumnCount(4)
        title = ["關節點", "X", "Y", "有無更改"]
        self.ui.keypoint_table.setHorizontalHeaderLabels(title)
        header = self.ui.keypoint_table.horizontalHeader()
        for i in range(4):
            header.setDefaultAlignment(Qt.AlignLeft)

    def import_data_to_table(self, frame_num:int):
        self.clear_table_view()
        person_id = self.pose_estimater.person_id
        if person_id is None:
            return
        person_data = self.pose_estimater.get_person_df_data(frame_num=frame_num, is_select=True)
        if person_data.empty:
            self.clear_table_view()
            self.ui.select_checkbox.setChecked(False)
            return
        
        num_keypoints = len(self.kpt_dict)
        if self.ui.keypoint_table.rowCount() < num_keypoints:
            self.ui.keypoint_table.setRowCount(num_keypoints)

        for kpt_idx, kpt in enumerate(person_data['keypoints'].iloc[0]): 
            kptx, kpty, kpt_label = kpt[0], kpt[1], kpt[3]
            kpt_name = self.kpt_dict[kpt_idx]
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

    def on_cell_clicked(self, row, column):
        self.correct_kpt_idx = row
        self.label_kpt = True
     
    def send_to_table(self, kptx:float, kpty:float, kpt_label:int):
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

        self.pose_estimater.update_person_df(kptx, kpty, self.ui.frame_slider.value(), self.correct_kpt_idx)
        self.update_frame(frame_num=self.ui.frame_slider.value())

    def mousePressEvent(self, event):
        view_rect = self.ui.frame_view.rect()
        pos = event.pos()

        if not view_rect.contains(pos):
            return
        search_person_df = self.pose_estimater.get_person_df_data(frame_num = self.ui.frame_slider.value())
        scene_pos = self.ui.frame_view.mapToScene(pos)
        x, y = scene_pos.x(), scene_pos.y()

        if self.ui.select_checkbox.isChecked() and not self.label_kpt :
            if event.button() == Qt.LeftButton:
                self.person_selector.select(x, y, search_person_df)
                self.pose_estimater.set_person_id(self.person_selector.selected_id)
        
        if self.ui.select_kpt_checkbox.isChecked() and not self.label_kpt :
            if event.button() == Qt.LeftButton:
                self.kpt_selector.select(x, y, search_person_df)
                self.pose_estimater.set_kpt_id(self.kpt_selector.selected_id)

        if self.label_kpt:
            if event.button() == Qt.LeftButton:
                self.send_to_table(x, y, 1)
            elif event.button() == Qt.RightButton:
                self.send_to_table(0, 0, 0)
            self.label_kpt = False

            self.update_frame(self.ui.frame_slider.value())

    def keyPressEvent(self, event):
        if event.key() == ord('D') or event.key() == ord('d'):
            self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        elif event.key() == ord('A') or event.key() == ord('a'):
            self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        else:
            super().keyPressEvent(event)
       
    def correct_id(self):
        before_correct_id = self.ui.before_correct_id.value()
        after_correct_id = self.ui.after_correct_id.value()
        self.pose_estimater.correct_person_id(before_correct_id, after_correct_id)
        self.update_frame(self.ui.frame_slider.value())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PoseVideoTabControl()
    window.show()
    sys.exit(app.exec_())
