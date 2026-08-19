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

class PoseCameraTabControl(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_camera_ui()
        self.ui.setupUi(self)
        # self.model = model
        # self.init_pose_estimater()
        self.initVar()
        self.bindUI()

    def initVar(self):
        """Initialize variables and timer."""
        self.camera = Camera()
        self.timer = QTimer()
        self.timer.timeout.connect(self.analyzeFrame)
        self.camera_scene = QGraphicsScene()

    def bindUI(self):
        """Bind UI elements to their corresponding functions."""
        self.ui.cameraCheckBox.stateChanged.connect(self.toggleCamera)
        self.ui.recordCheckBox.stateChanged.connect(self.toggleRecord)
        self.ui.CameraIdInput.valueChanged.connect(self.changeCamera)

    def toggleCamera(self, state:int):
        """Toggle the camera on/off based on checkbox state."""
        if state == 2:
            # frame_width, frame_height, fps = 
            self.camera.toggleCamera(True)
            # self.ui.ResolutionLabel.setText(f"(0, 0) - ({frame_width} x {frame_height}), FPS: {fps}")
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

    def changeCamera(self):
        """Change the camera based on input value."""
        self.camera.setCameraId(self.ui.CameraIdInput.value())

    def analyzeFrame(self):
        """Analyze and process each frame from the camera."""
        if not self.camera.frame_buffer.empty():
            frame = self.camera.frame_buffer.get().copy()
            # _, _, fps = self.pose_estimater.detectKpt(frame, is_video=False)
            # self.ui.FPSInfoLabel.setText(f"{fps:02d}")
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
