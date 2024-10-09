from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import numpy as np
import sys
import cv2
import os
from camera_ui import Ui_camera_ui
from datetime import datetime
from cv_utils.cv_control import Camera
from utils.selector import PersonSelector, KptSelector
from utils.vis_image import ImageDrawer
from skeleton.detect_skeleton import PoseEstimater
from utils.model import Model

class PoseCameraTabControl(QWidget):
    def __init__(self, model:Model, parent=None):
        super().__init__(parent)
        self.ui = Ui_camera_ui()
        self.ui.setupUi(self)
        self.model = model
        self.initVar()
        self.init_pose_estimater()
        self.bindUI()

    def initVar(self):
        """Initialize variables and timer."""
        self.camera = Camera()
        self.timer = QTimer()
        self.timer.timeout.connect(self.analyzeFrame)
        self.camera_scene = QGraphicsScene()

    def init_pose_estimater(self):
        """Initialize the pose estimator and related components."""
        self.person_selector = PersonSelector()
        self.kpt_selector = KptSelector()
        self.pose_estimater = PoseEstimater(self.model)
        self.image_drawer = ImageDrawer(self.pose_estimater)

    def bindUI(self):
        """Bind UI elements to their corresponding functions."""
        self.ui.cameraCheckBox.stateChanged.connect(self.toggleCamera)
        self.ui.recordCheckBox.stateChanged.connect(self.toggleRecord)
        self.ui.selectCheckBox.stateChanged.connect(self.toggleSelect)
        self.ui.showSkeletonCheckBox.stateChanged.connect(self.toggleShowSkeleton)
        self.ui.selectKptCheckBox.stateChanged.connect(self.toggleKptSelect)
        self.ui.showBboxCheckBox.stateChanged.connect(self.toggleShowBbox)
        self.ui.showLineCheckBox.stateChanged.connect(self.toggleShowGrid)
        self.ui.CameraIdInput.valueChanged.connect(self.changeCamera)

    def toggleCamera(self, state:int):
        """Toggle the camera on/off based on checkbox state."""
        if state == 2:
            frame_width, frame_height, fps = self.camera.toggleCamera(True)
            self.model.setImageSize((frame_width, frame_height))
            self.ui.ResolutionLabel.setText(f"(0, 0) - ({frame_width} x {frame_height}), FPS: {fps}")
            self.timer.start(1)
        else:
            self.camera.toggleCamera(False)
            self.timer.stop()

    def toggleRecord(self, state:int):
        """Start or stop video recording."""
        if state == 2:
            self.startRecording()
            self.ui.showSkeletonCheckBox.setChecked(False)
        else:
            self.camera.stop_recording()

    def startRecording(self):
        """Start recording the video."""
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = f'../../Db/Record/C{self.ui.CameraIdInput.value()}_Fps120_{current_time}'
        os.makedirs(output_dir, exist_ok=True)
        video_filename = os.path.join(output_dir, f'C{self.ui.CameraIdInput.value()}_Fps120_{current_time}.mp4')
        self.ui.showSkeletonCheckBox.setChecked(False)
        self.camera.startRecording(video_filename)

    def toggleSelect(self, state:int):
        """Select a person based on checkbox state."""
        if not self.ui.showSkeletonCheckBox.isChecked():
            self.ui.selectCheckBox.setCheckState(0)
            QMessageBox.warning(self, "無法選擇人", "請選擇顯示人體骨架!")
            return
        if state == 2:
            self.person_selector.select(search_person_df = self.pose_estimater.pre_person_df)
            self.pose_estimater.setPersonId(self.person_selector.selected_id)
        else:
            self.pose_estimater.setPersonId(None)

    def toggleKptSelect(self, state:int):
        """Toggle keypoint selection and trajectory visualization."""
        if not self.ui.selectCheckBox.isChecked():
            self.ui.selectKptCheckBox.setCheckState(0)
            QMessageBox.warning(self, "無法選擇關節點", "請選擇人!")
            return
        is_checked = state == 2
        self.pose_estimater.setKptId(10 if is_checked else None)
        self.pose_estimater.clearKptBuffer()
        self.image_drawer.setShowTraj(is_checked)

    def toggleShowSkeleton(self, state:int):
        """Toggle skeleton detection and FPS control."""
        is_checked = state == 2
        self.pose_estimater.setDetect(is_checked)
        self.image_drawer.setShowSkeleton(is_checked)
        self.camera.setFPSControl(15 if is_checked else 1)

    def toggleShowBbox(self, state:int):
        """Toggle bounding box visibility."""
        self.image_drawer.setShowBbox(state == 2)

    def toggleShowGrid(self, state:int):
        """Toggle gridline visibility."""
        self.image_drawer.setShowGrid(state == 2)

    def changeCamera(self):
        """Change the camera based on input value."""
        self.camera.setCameraId(self.ui.CameraIdInput.value())

    def analyzeFrame(self):
        """Analyze and process each frame from the camera."""
        if not self.camera.frame_buffer.empty():
            frame = self.camera.frame_buffer.get().copy()
            _, _, fps = self.pose_estimater.detectKpt(frame, is_video=False)
            self.ui.FPSInfoLabel.setText(f"{fps:02d}")
            self.update_frame(frame)

    def update_frame(self, frame: np.ndarray):
        """Update the displayed frame with additional analysis."""
        drawed_img = self.image_drawer.drawInfo(img = frame, kpt_buffer = self.pose_estimater.kpt_buffer)
        self.showImage(drawed_img, self.camera_scene, self.ui.FrameView)

    def showImage(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView):
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
        if not self.ui.FrameView.rect().contains(event.pos()):
            return
        
        scene_pos = self.ui.FrameView.mapToScene(event.pos())
        x, y = scene_pos.x(), scene_pos.y()
        search_person_df = self.pose_estimater.pre_person_df

        if self.ui.selectCheckBox.isChecked() and event.button() == Qt.LeftButton:
            print(search_person_df)
            self.person_selector.select(x, y, search_person_df)
            self.pose_estimater.setPersonId(self.person_selector.selected_id)

        if self.ui.selectKptCheckBox.isChecked() and event.button() == Qt.LeftButton:
            self.kpt_selector.select(x, y, search_person_df)
            self.pose_estimater.setKptId(self.kpt_selector.selected_id)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PoseCameraTabControl()
    window.show()
    sys.exit(app.exec_())
