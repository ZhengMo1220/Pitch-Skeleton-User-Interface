from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import numpy as np
import sys
import cv2
import os
from camera_ui import Ui_camera_ui
from datetime import datetime
from Camera.camera_control import Camera
from utils.selector import Person_selector, Kpt_selector
from utils.vis_image import ImageDrawer
from skeleton.detect_skeleton import PoseEstimater

class PoseCameraTabControl(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.ui = Ui_camera_ui()
        self.ui.setupUi(self)
        
        self.model = model
        self.init_var()
        self.init_pose_estimater()
        self.bind_ui()

    def init_var(self):
        """Initialize variables and timer."""
        self.camera = Camera()
        self.timer = QTimer()
        self.timer.timeout.connect(self.analyze_frame)
        self.camera_scene = QGraphicsScene()

    def init_pose_estimater(self):
        """Initialize the pose estimator and related components."""
        self.person_selector = Person_selector()
        self.kpt_selector = Kpt_selector()
        self.pose_estimater = PoseEstimater(self.model)
        self.image_drawer = ImageDrawer(self.pose_estimater)

    def bind_ui(self):
        """Bind UI elements to their corresponding functions."""
        self.ui.camera_checkBox.stateChanged.connect(self.toggle_camera)
        self.ui.record_checkBox.stateChanged.connect(self.toggle_record)
        self.ui.select_checkBox.stateChanged.connect(self.toggle_select)
        self.ui.show_skeleton_checkBox.stateChanged.connect(self.toggle_show_skeleton)
        self.ui.select_keypoint_checkbox.stateChanged.connect(self.toggle_kpt_select)
        self.ui.show_bbox_checkbox.stateChanged.connect(self.toggle_show_bbox)
        self.ui.show_line_checkBox.stateChanged.connect(self.toggle_show_grid)
        self.ui.camera_id_input.valueChanged.connect(self.change_camera)

    def toggle_camera(self, state):
        """Toggle the camera on/off based on checkbox state."""
        if state == 2:
            frame_width, frame_height, fps = self.camera.toggle_camera(True)
            self.ui.image_resolution_label.setText(f"(0, 0) - ({frame_width} x {frame_height}), FPS: {fps}")
            self.timer.start(1)
        else:
            self.camera.toggle_camera(False)
            self.timer.stop()

    def toggle_record(self, state):
        """Start or stop video recording."""
        if state == 2:
            self.start_recording()
        else:
            self.camera.stop_recording()

    def start_recording(self):
        """Start recording the video."""
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = f'../../Db/Record/C{self.ui.camera_id_input.value()}_Fps120_{current_time}'
        os.makedirs(output_dir, exist_ok=True)
        video_filename = os.path.join(output_dir, f'C{self.ui.camera_id_input.value()}_Fps120_{current_time}.mp4')
        self.camera.start_recording(video_filename)

    def toggle_select(self, state):
        """Select a person based on checkbox state."""
        if state == 2:
            self.person_selector.select(search_person_df = self.pose_estimater.pre_person_df)
            self.pose_estimater.set_person_id(self.person_selector.selected_id)
        else:
            self.pose_estimater.set_person_id(None)

    def toggle_kpt_select(self, state):
        """Toggle keypoint selection and trajectory visualization."""
        is_checked = state == 2
        self.pose_estimater.set_kpt_id(9 if is_checked else None)
        self.image_drawer.set_show_traj(is_checked)

    def toggle_show_skeleton(self, state):
        """Toggle skeleton detection and FPS control."""
        is_checked = state == 2
        self.pose_estimater.set_detect(is_checked)
        self.image_drawer.set_show_skeleton(is_checked)
        self.camera.set_fps_control(15 if is_checked else 1)

    def toggle_show_bbox(self, state):
        """Toggle bounding box visibility."""
        self.image_drawer.set_show_bbox(state == 2)

    def toggle_show_grid(self, state):
        """Toggle gridline visibility."""
        self.image_drawer.set_show_grid(state == 2)

    def change_camera(self):
        """Change the camera based on input value."""
        self.camera.set_camera_idx(self.ui.camera_id_input.value())

    def analyze_frame(self):
        """Analyze and process each frame from the camera."""
        if not self.camera.frame_buffer.empty():
            frame = self.camera.frame_buffer.get().copy()
            _, self.person_df, fps = self.pose_estimater.detect_kpt(frame)
            self.ui.fps_info_label.setText(f"{fps:02d}")
            self.update_frame(frame)

    def update_frame(self, image: np.ndarray):
        """Update the displayed frame with additional analysis."""
        self.image_drawer.draw_info(img = image, kpt_buffer = self.pose_estimater.kpt_buffer)
        self.show_image(image, self.camera_scene, self.ui.camer_frame_view)

    def show_image(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView):
        """Display an image in the QGraphicsView."""
        scene.clear()
        h, w = image.shape[:2]
        qImg = QImage(image, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        scene.addPixmap(pixmap)
        GraphicsView.setScene(scene)
        GraphicsView.setAlignment(Qt.AlignLeft)
        GraphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        """Handle mouse events for person and keypoint selection."""
        if not self.ui.camer_frame_view.rect().contains(event.pos()):
            return
        
        scene_pos = self.ui.camer_frame_view.mapToScene(event.pos())
        x, y = scene_pos.x(), scene_pos.y()
        search_person_df = self.pose_estimater.get_pre_person_df()

        if self.ui.select_checkBox.isChecked() and event.button() == Qt.LeftButton:
            self.person_selector.select(x, y, search_person_df)
            self.pose_estimater.set_person_id(self.person_selector.selected_id)

        if self.ui.select_keypoint_checkbox.isChecked() and event.button() == Qt.LeftButton:
            self.kpt_selector.select(x, y, search_person_df)
            self.pose_estimater.set_kpt_id(self.kpt_selector.selected_id)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PoseCameraTabControl()
    window.show()
    sys.exit(app.exec_())
