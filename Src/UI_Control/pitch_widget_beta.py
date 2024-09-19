from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import Qt, QTimer
import numpy as np
import sys
import os
from pitch_ui import Ui_pitch_ui
from datetime import datetime
from utils.timer import Timer
from Camera.camera_control import Camera, VideoLoader
from utils.selector import Person_selector, Kpt_selector
from utils.analyze import PoseAnalyzer
import cv2
from utils.vis_graph import GraphPlotter
from utils.vis_image import ImageDrawer
from skeleton.detect_skeleton import PoseEstimater
import pyqtgraph as pg

class PosePitchTabControl(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.ui = Ui_pitch_ui()
        self.ui.setupUi(self)
        self.model = model
        self.init_var()
        self.init_pose_estimater()
        self.bind_ui()
        self.video_state()

    def init_var(self):
        self.camera = None
        self.timer = None

        self.record_timer = None
        self.view_scene = QGraphicsScene()
        self.person_selector = None
        self.kpt_selector = None
        #pyqtgraph setting
        pg.setConfigOptions(foreground=QColor(113,148,116), antialias = True)
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

    def bind_ui(self):
        """Bind UI elements to their corresponding functions."""
        self.ui.camera_checkbox.stateChanged.connect(self.toggle_camera)
        self.ui.record_checkbox.stateChanged.connect(self.toggle_record)
        self.ui.select_checkbox.stateChanged.connect(self.toggle_select)
        self.ui.show_skeleton_checkbox.stateChanged.connect(self.toggle_show_skeleton)
        self.ui.select_keypoint_checkbox.stateChanged.connect(self.toggle_kpt_select)
        self.ui.show_bbox_checkbox.stateChanged.connect(self.toggle_show_bbox)
        self.ui.show_line_checkbox.stateChanged.connect(self.toggle_show_grid)
        self.ui.start_pitch_checkbox.stateChanged.connect(self.toggle_pitching)
        self.ui.camera_id_input.valueChanged.connect(self.change_camera)
        self.ui.pitch_input.currentIndexChanged.connect(self.change_pitcher)

        # video_widget
        self.ui.play_btn.clicked.connect(self.play_btn_clicked)
        self.ui.back_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        )
        self.ui.forward_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        )
        self.ui.frame_slider.valueChanged.connect(self.analyze_frame)
    
    def reset(self):
        self.pose_estimater.reset()
        self.pose_analyzer.reset()
        self.graph_plotter.reset()
        self.image_drawer.reset()
        self.video_loader.reset()
        self.video_state()

    def load_video(self, video_path:str):
        self.video_loader = VideoLoader(self.image_drawer)
        self.video_loader.load_video(video_path=video_path)
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
            self.show_graph(self.curve_scene, self.ui.curve_view)

            self.ui.show_skeleton_checkbox.setChecked(True)
            image = self.video_loader.get_video_image(0)
            _, self.person_df, fps= self.pose_estimater.detect_kpt(image, 0)
            self.ui.fps_info_label.setText(f"{fps:02d}")
            self.update_frame(frame_num=0)
            
            self.ui.select_checkbox.setChecked(True)
            self.ui.select_keypoint_checkbox.setChecked(True)
            self.ui.play_btn.click()

    def toggle_camera(self, state):
        """Toggle the camera on/off based on checkbox state."""
        if state == 2:  # 開啟攝影機
            self.camera = Camera()
            self.timer = QTimer()
            self.timer.timeout.connect(self.analyze_frame)

            # 開啟攝影機並顯示解析度和幀率
            frame_width, frame_height, fps = self.camera.toggle_camera(True)
            self.ui.image_resolution_label.setText(f"(0, 0) - ({frame_width} x {frame_height}), FPS: {fps}")

            # 啟動定時器並更新 UI
            self.timer.start(1)
            self.video_silder(False)

        else:
            if self.camera is not None:
                self.camera.toggle_camera(False)
            if self.timer is not None:
                self.timer.stop()
            # 清理資源並更新 UI
            self.camera = None
            self.timer = None
            # self.update_frame()
            self.video_silder(True)

    def toggle_record(self, state):
        """Start or stop video recording."""
        if state == 2:
            self.start_recording()
            print("record!!")
        else:
            if self.camera is None:
                return
            print("stop record!!")
            self.camera.stop_recording()

    def start_recording(self):
        """Start recording the video."""
        if self.camera is None:
            return
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = f'../../Db/Record/C{self.ui.camera_id_input.value()}_Fps120_{current_time}'
        os.makedirs(output_dir, exist_ok=True)
        video_filename = os.path.join(output_dir, f'C{self.ui.camera_id_input.value()}_Fps120_{current_time}.mp4')
        self.camera.start_recording(video_filename)

    def toggle_select(self, state):
        """Select a person based on checkbox state."""
        if state == 2:
            self.person_selector = Person_selector()
            self.person_selector.select(search_person_df = self.pose_estimater.pre_person_df)
            if self.camera is None:
                self.person_selector.select(search_person_df=self.pose_estimater.get_person_df_data(frame_num=self.ui.frame_slider.value()))
            self.pose_estimater.set_person_id(self.person_selector.selected_id)
        else:
            self.pose_estimater.set_person_id(None)
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
            self.load_video(video_path)

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

    def toggle_kpt_select(self, state):
        """Toggle keypoint selection and trajectory visualization."""
        is_checked = state == 2
        if state == 2:
            self.kpt_selector = Kpt_selector()
            self.pose_estimater.set_person_id(self.person_selector.selected_id)
        else:
            self.kpt_selector = None
        self.pose_estimater.set_kpt_id(9 if is_checked else None)
        self.image_drawer.set_show_traj(is_checked)

    def toggle_show_skeleton(self, state):
        """Toggle skeleton detection and FPS control."""
        is_checked = state == 2
        self.pose_estimater.set_detect(is_checked)
        self.image_drawer.set_show_skeleton(is_checked)
        if self.camera is not None:
            self.camera.set_fps_control(15 if is_checked else 1)

    def toggle_show_bbox(self, state):
        """Toggle bounding box visibility."""
        self.image_drawer.set_show_bbox(state == 2)

    def toggle_show_grid(self, state):
        """Toggle gridline visibility."""
        self.image_drawer.set_show_grid(state == 2)

    def change_camera(self):
        """Change the camera based on input value."""
        if self.camera is not None:
            self.camera.set_camera_idx(self.ui.camera_id_input.value())

    def change_pitcher(self):
        """Change the pitcher based on input value. 9: "左腕", 10: "右腕","""
        self.ui.camera_checkbox.setChecked(False)
        if self.ui.pitch_input.currentIndex() == 0:
            self.pose_estimater.set_kpt_id(10)
        else :
            self.pose_estimater.set_kpt_id(9)

    def analyze_frame(self):
        """Analyze and process each frame from the camera."""
        if self.camera is not None:
            # print("camera")
            if not self.camera.frame_buffer.empty():
                frame = self.camera.frame_buffer.get()
                _, self.person_df, fps = self.pose_estimater.detect_kpt(frame)
                self.ui.fps_info_label.setText(f"{fps:02d}")
                self.update_frame(img=frame)
        else:
            # print("video")
            frame_num = self.ui.frame_slider.value()
            self.ui.frame_num_label.setText(f'{frame_num}/{len(self.video_loader.video_frames) - 1}')
            image = self.video_loader.get_video_image(frame_num)
            _, self.person_df, fps= self.pose_estimater.detect_kpt(image,frame_num)
            self.ui.fps_info_label.setText(f"{fps:02d}")
            if self.pose_estimater.person_id is not None:
                self.pose_analyzer.add_analyze_info(frame_num)
                self.graph_plotter.update_graph(frame_num)
            self.update_frame(frame_num = frame_num)
            if frame_num == self.video_loader.total_frames - 1:
                self.ui.play_btn.click()
                self.video_loader.save_video()

    def update_frame(self, img: np.ndarray = None, frame_num:int= None):
        """Update the displayed frame with additional analysis."""
        show_image = self.image_drawer.draw_info(img = img, kpt_buffer = self.pose_estimater.kpt_buffer)
        if frame_num is not None:
            img = self.video_loader.get_video_image(frame_num)
            show_image = self.image_drawer.draw_info(img = img, frame_num=frame_num, kpt_buffer = self.pose_estimater.kpt_buffer)
            
        self.show_image(show_image, self.view_scene, self.ui.frame_view)

    def show_image(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView):
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
        search_person_df = self.pose_estimater.get_pre_person_df()
        if self.camera is None:
            search_person_df = self.pose_estimater.get_person_df_data(frame_num = self.ui.frame_slider.value())

        if self.person_selector is not None and event.button() == Qt.LeftButton:
            self.person_selector.select(x, y, search_person_df)
            self.pose_estimater.set_person_id(self.person_selector.selected_id)

        if self.kpt_selector is not None and event.button() == Qt.LeftButton:
            self.kpt_selector.select(x, y, search_person_df)
            self.pose_estimater.set_kpt_id(self.kpt_selector.selected_id)

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

    def show_graph(self, scene, graphicview):
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PosePitchTabControl()
    window.show()
    sys.exit(app.exec_())
