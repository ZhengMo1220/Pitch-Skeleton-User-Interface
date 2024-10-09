from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import Qt, QTimer
import numpy as np
import sys
import os
from pitch_ui import Ui_pitch_ui
from datetime import datetime
from utils.timer import Timer
from cv_utils.cv_control import Camera, VideoLoader
from utils.selector import PersonSelector, KptSelector
from utils.analyze import PoseAnalyzer
import cv2
from utils.vis_graph import GraphPlotter
from utils.vis_image import ImageDrawer
from skeleton.detect_skeleton import PoseEstimater
import pyqtgraph as pg
from utils.model import Model

class PosePitchTabControl(QWidget):
    def __init__(self, model:Model, parent=None):
        super().__init__(parent)
        self.ui = Ui_pitch_ui()
        self.ui.setupUi(self)
        self.model = model
        self.initVar()
        self.init_pose_estimater()
        self.bindUI()
        self.video_state()

    def initVar(self):
        self.camera = None
        self.timer = None
        self.record_timer = None
        self.view_scene = QGraphicsScene()
        self.person_selector = None
        self.kpt_selector = None

        #pyqtgraph setting
        pg.setConfigOptions(foreground = QColor(113,148,116), antialias = True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

    def video_state(self):
        self.is_play = False
        self.view_scene = QGraphicsScene()
        self.curve_scene = QGraphicsScene()
        self.view_scene.clear()
        self.curve_scene.clear()
        self.correct_kpt_idx = 0
        self.label_kpt = False

    def init_pose_estimater(self):
        """Initialize the pose estimator and related components."""
        self.pose_estimater = PoseEstimater(self.model)
        self.pose_analyzer = PoseAnalyzer(self.pose_estimater)
        self.graph_plotter = GraphPlotter(self.pose_analyzer)
        self.image_drawer = ImageDrawer(self.pose_estimater, self.pose_analyzer)
        self.video_loader = VideoLoader(self.image_drawer)

    def bindUI(self):
        """Bind UI elements to their corresponding functions."""
        self.ui.camera_checkbox.stateChanged.connect(self.toggleCamera)
        self.ui.record_checkbox.stateChanged.connect(self.toggleRecord)
        self.ui.select_checkbox.stateChanged.connect(self.toggle_select)
        self.ui.show_skeleton_checkbox.stateChanged.connect(self.toggleShowSkeleton)
        self.ui.select_keypoint_checkbox.stateChanged.connect(self.toggleKptSelect)
        self.ui.show_bbox_checkbox.stateChanged.connect(self.toggleShowBbox)
        self.ui.show_line_checkbox.stateChanged.connect(self.toggleShowGrid)
        
        self.ui.start_pitch_checkbox.stateChanged.connect(self.toggle_pitching)
        self.ui.camera_id_input.valueChanged.connect(self.changeCamera)
        self.ui.pitch_input.currentIndexChanged.connect(self.change_pitcher)

        # video_widget
        self.ui.play_btn.clicked.connect(self.play_btn_clicked)
        self.ui.back_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        )
        self.ui.forward_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        )
        self.ui.frame_slider.valueChanged.connect(self.analyzeFrame)
        self.ui.keypoint_table.cellActivated.connect(self.onCellClicked)
    
    def reset(self):
        self.pose_estimater.reset()
        self.pose_analyzer.reset()
        self.graph_plotter.reset()
        self.image_drawer.reset()
        self.video_loader.reset()
        self.video_state()

    def loadVideo(self, video_path:str):
        self.video_loader = VideoLoader(self.image_drawer)
        self.video_loader.loadVideo(video_path=video_path)
        self.check_video_load()

    def check_video_load(self):
        if self.video_loader.video_name is None:
            return
        if self.video_loader.is_loading:
            # 影片正在讀取中，稍後再檢查
            QTimer.singleShot(100, self.check_video_load)  # 每100ms檢查一次
        else:
            self.ui.frame_slider.setMinimum(0)
            self.ui.frame_slider.setMaximum(self.video_loader.total_frames - 1)
            self.ui.frame_slider.setValue(0)
            self.ui.frame_num_label.setText(f'0/{self.video_loader.total_frames-1}')
            self.ui.image_resolution_label.setText( "(0,0) -" + f" {self.video_loader.video_size[0]} x {self.video_loader.video_size[1]}")
            self.graph_plotter._init_graph(self.video_loader.total_frames)
            self.showGraph(self.curve_scene, self.ui.curve_view)

            self.ui.show_skeleton_checkbox.setChecked(True)
            image = self.video_loader.getVideoImage(0)
            _, self.person_df, fps= self.pose_estimater.detectKpt(image, 0)
            self.ui.fps_info_label.setText(f"{fps:02d}")
            self.update_frame(frame_num=0)
            
            self.ui.select_checkbox.setChecked(True)
            self.ui.select_keypoint_checkbox.setChecked(True)
            self.ui.play_btn.click()

    def toggleCamera(self, state):
        """Toggle the camera on/off based on checkbox state."""
        if state == 2:  # 開啟攝影機
            self.camera = Camera()
            self.timer = QTimer()
            self.timer.timeout.connect(self.analyzeFrame)

            # 開啟攝影機並顯示解析度和幀率
            frame_width, frame_height, fps = self.camera.toggleCamera(True)
            self.ui.image_resolution_label.setText(f"(0, 0) - ({frame_width} x {frame_height}), FPS: {fps}")
            self.model.setImageSize((frame_width, frame_height,3))
            # 啟動定時器並更新 UI
            self.timer.start(1)
            self.video_silder(visible=False)

        else:
            if self.camera is not None:
                self.camera.toggleCamera(False)
            if self.timer is not None:
                self.timer.stop()
            self.update_frame()
            # 清理資源並更新 UI
            self.camera = None
            self.timer = None
            # self.update_frame()
            self.video_silder(visible=True)

    def toggleRecord(self, state):
        """Start or stop video recording."""
        if state == 2:
            self.startRecording()
            print("record!!")
        else:
            if self.camera is None:
                return
            print("stop record!!")
            self.camera.stop_recording()

    def startRecording(self):
        """Start recording the video."""
        if self.camera is None:
            return
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = f'../../Db/Record/C{self.ui.camera_id_input.value()}_Fps120_{current_time}'
        os.makedirs(output_dir, exist_ok=True)
        video_filename = os.path.join(output_dir, f'C{self.ui.camera_id_input.value()}_Fps120_{current_time}.mp4')
        self.camera.startRecording(video_filename)

    def toggle_select(self, state):
        """Select a person based on checkbox state."""
        if state == 2:
            self.person_selector = PersonSelector()
            self.person_selector.select(search_person_df = self.pose_estimater.pre_person_df)
            if self.camera is None:
                self.person_selector.select(search_person_df=self.pose_estimater.getPersonDf(frame_num=self.ui.frame_slider.value()))
            self.pose_estimater.setPersonId(self.person_selector.selected_id)
        else:
            self.pose_estimater.setPersonId(None)
            self.person_selector = None

    def check_time_up(self):
        if self.record_timer is None:
            return
        if not self.record_timer.is_time_up():
            # 影片正在讀取中，稍後再檢查
            QTimer.singleShot(100, self.check_time_up)  # 每100ms檢查一次
        else:
            video_path = self.camera.video_path
            self.ui.record_checkbox.setChecked(False)
            self.ui.camera_checkbox.setChecked(False)
            self.ui.start_pitch_checkbox.setChecked(False)
            self.loadVideo(video_path)

    def toggle_pitching(self, state):
        if state == 2:
            self.reset()
            self.ui.camera_checkbox.setChecked(True)
            self.ui.show_skeleton_checkbox.setChecked(False)
            self.ui.select_checkbox.setChecked(False)
            self.ui.select_keypoint_checkbox.setChecked(False)
            self.ui.record_checkbox.setChecked(True)
            self.record_timer = Timer(3)
            self.record_timer.start()
            self.check_time_up()

    def toggleKptSelect(self, state):
        """Toggle keypoint selection and trajectory visualization."""
        is_checked = state == 2
        if state == 2:
            self.kpt_selector = KptSelector()
            self.pose_estimater.setPersonId(self.person_selector.selected_id)
        else:
            self.kpt_selector = None
        self.pose_estimater.setKptId(10 if is_checked else None)
        self.image_drawer.setShowTraj(is_checked)

    def toggleShowSkeleton(self, state):
        """Toggle skeleton detection and FPS control."""
        is_checked = state == 2
        self.pose_estimater.setDetect(is_checked)
        self.image_drawer.setShowSkeleton(is_checked)
        if self.camera is not None:
            self.camera.setFPSControl(15 if is_checked else 1)

    def toggleShowBbox(self, state):
        """Toggle bounding box visibility."""
        self.image_drawer.setShowBbox(state == 2)

    def toggleShowGrid(self, state):
        """Toggle gridline visibility."""
        self.image_drawer.setShowGrid(state == 2)

    def changeCamera(self):
        """Change the camera based on input value."""
        if self.camera is not None:
            self.camera.setCameraId(self.ui.camera_id_input.value())

    def change_pitcher(self):
        """Change the pitcher based on input value. 9: "左腕", 10: "右腕","""
        self.ui.camera_checkbox.setChecked(False)
        if self.ui.pitch_input.currentIndex() == 0:
            self.pose_estimater.setKptId(10)
        else :
            self.pose_estimater.setKptId(9)

    def analyzeFrame(self):
        """Analyze and process each frame from the camera."""
        if self.camera is not None:
            if not self.camera.frame_buffer.empty():
                frame = self.camera.frame_buffer.get()
                _, self.person_df, fps = self.pose_estimater.detectKpt(frame)
                self.ui.fps_info_label.setText(f"{fps:02d}")
                self.update_frame(img=frame)
        else:
            frame_num = self.ui.frame_slider.value()
            self.ui.frame_num_label.setText(f'{frame_num}/{len(self.video_loader.video_frames) - 1}')
            image = self.video_loader.getVideoImage(frame_num)
            _, self.person_df, fps= self.pose_estimater.detectKpt(image,frame_num)
            self.ui.fps_info_label.setText(f"{fps:02d}")
            if self.pose_estimater.person_id is not None:
                self.importDatatoTable(frame_num)
                self.pose_analyzer.addAnalyzeInfo(frame_num)
                self.graph_plotter.updateGraph(frame_num)
            self.update_frame(frame_num = frame_num)
            if frame_num == self.video_loader.total_frames - 1:
                self.ui.play_btn.click()
                self.video_loader.saveVideo()

    def update_frame(self, img: np.ndarray = None, frame_num:int= None):
        """Update the displayed frame with additional analysis."""
        show_img = self.image_drawer.drawInfo(img = img, kpt_buffer = self.pose_estimater.kpt_buffer)
        if frame_num is not None:
            img = self.video_loader.getVideoImage(frame_num)
            show_img = self.image_drawer.drawInfo(img = img, frame_num=frame_num, kpt_buffer = self.pose_estimater.kpt_buffer)
            
        self.showImage(show_img, self.view_scene, self.ui.frame_view)

    def showImage(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView):
        """Display an image in the QGraphicsView."""
        scene.clear()
        if image is not None:
            h, w = image.shape[:2]
            qImg = QImage(image, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qImg)
            scene.addPixmap(pixmap)
        GraphicsView.setScene(scene)
        GraphicsView.setAlignment(Qt.AlignLeft)
        GraphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        """Handle mouse events for person and keypoint selection."""
        if not self.ui.frame_view.rect().contains(event.pos()):
            return
        
        scene_pos = self.ui.frame_view.mapToScene(event.pos())
        x, y = scene_pos.x(), scene_pos.y()
        search_person_df = self.pose_estimater.pre_person_df
        if self.camera is None:
            search_person_df = self.pose_estimater.getPersonDf(frame_num = self.ui.frame_slider.value())

        if self.person_selector is not None and event.button() == Qt.LeftButton:
            self.person_selector.select(x, y, search_person_df)
            self.pose_estimater.setPersonId(self.person_selector.selected_id)

        if self.kpt_selector is not None and event.button() == Qt.LeftButton:
            self.kpt_selector.select(x, y, search_person_df)
            self.pose_estimater.setKptId(self.kpt_selector.selected_id)

        if self.label_kpt:
            if event.button() == Qt.LeftButton:
                self.sendtoTable(x, y, 1)
            elif event.button() == Qt.RightButton:
                self.sendtoTable(0, 0, 0)
            self.label_kpt = False

    def playFrame(self, start_num:int =0):
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
            self.playFrame(self.ui.frame_slider.value())

    def video_silder(self, visible:bool):
        elements = [
            self.ui.back_key_btn,
            self.ui.play_btn,
            self.ui.forward_key_btn,
            self.ui.frame_slider,
            self.ui.frame_num_label,
            self.ui.curve_view
        ]
        
        for element in elements:
            element.setVisible(visible)

    def showGraph(self, scene:QGraphicsScene, graphicview: QGraphicsView):
        graph = self.graph_plotter.graph
        graph.resize(graphicview.width(),graphicview.height())
        scene.addWidget(graph)
        graphicview.setScene(scene)
        graphicview.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def keyPressEvent(self, event):
        if event.key() == ord('D') or event.key() == ord('d'):
            self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        elif event.key() == ord('A') or event.key() == ord('a'):
            self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        else:
            super().keyPressEvent(event)

    def clearTableView(self):
        self.ui.keypoint_table.clear()
        self.ui.keypoint_table.setColumnCount(4)
        title = ["關節點", "X", "Y", "有無更改"]
        self.ui.keypoint_table.setHorizontalHeaderLabels(title)
        header = self.ui.keypoint_table.horizontalHeader()
        for i in range(4):
            header.setDefaultAlignment(Qt.AlignLeft)

    def importDatatoTable(self, frame_num:int):
        self.clearTableView()
        person_id = self.pose_estimater.person_id
        if person_id is None:
            return
        person_data = self.pose_estimater.getPersonDf(frame_num=frame_num, is_select=True)
        if person_data.empty:
            self.clearTableView()
            self.ui.select_checkbox.click()
            return
        
        num_keypoints = len(self.pose_estimater.joints["haple"]["keypoints"])
        if self.ui.keypoint_table.rowCount() < num_keypoints:
            self.ui.keypoint_table.setRowCount(num_keypoints)

        for kpt_idx, kpt in enumerate(person_data['keypoints'].iloc[0]): 
            kptx, kpty, kpt_label = kpt[0], kpt[1], kpt[3]
            kpt_name = self.pose_estimater.joints["haple"]["keypoints"][kpt_idx]
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

    def onCellClicked(self, row, column):
        self.correct_kpt_idx = row
        self.label_kpt = True
    
    def sendtoTable(self, kptx:float, kpty:float, kpt_label:int):
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
        self.update_person_df(kptx, kpty, kpt_label)

    def update_person_df(self, x, y, label):
        person_id = self.pose_estimater.person_id
        frame_num = self.ui.frame_slider.value()
        self.person_df.loc[(self.person_df['frame_number'] == frame_num) &
                            (self.person_df['person_id'] == person_id), 'keypoints'].iloc[0][self.correct_kpt_idx] = [x, y, 0.9, label]
        self.update_frame(frame_num=frame_num)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PosePitchTabControl()
    window.show()
    sys.exit(app.exec_())
