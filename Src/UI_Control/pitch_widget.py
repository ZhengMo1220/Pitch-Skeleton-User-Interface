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
from utils.analyze import PoseAnalyzer, JointAreaChecker
from ui_utils.table_control import KeypointTable
from ui_utils.graphicview_control import FrameView
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
        self.setupComponents()
        self.initVar()
        self.bindUI()

    def initVar(self):
        """Initialize variables and timer."""
        self.is_video = True if self.camera is None else False
        self.view_scene = QGraphicsScene()
        self.curve_scene = QGraphicsScene()
        self.view_scene.clear()
        self.curve_scene.clear()
        self.is_pitching = False
        self.initVideoVar()    

    def initVideoVar(self):
        self.is_play = False
        self.correct_kpt_idx = 0
        self.is_processed = False
        self.play_times = 2
        pg.setConfigOptions(foreground=QColor(113,148,116), antialias = True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

    def setupComponents(self): 
        self.camera = None
        self.timer = None
        self.countdown_timer = None
        self.record_checker = None
        self.record_timer = None
        self.pose_estimater = PoseEstimater(self.model)
        self.pose_analyzer = PoseAnalyzer(self.pose_estimater)
        self.graph_plotter = GraphPlotter(self.pose_analyzer)
        self.image_drawer = ImageDrawer(self.pose_estimater, self.pose_analyzer)
        self.video_loader = VideoLoader(self.image_drawer)

    def bindUI(self):
        """Bind UI element to their corresponding functions."""
        self.ui.cameraIdInput.valueChanged.connect(self.changeCamera)
        self.ui.pitchInput.currentIndexChanged.connect(self.changePitcher)
        self.bindVideoUI()
        self.bindCheckBox()
    
    def bindVideoUI(self):
        self.kpt_table = KeypointTable(self.ui.KptTable, self.pose_estimater)
        self.ui.KptTable.cellActivated.connect(self.kpt_table.onCellClicked)
        self.frame_view = FrameView(self.ui.FrameView, self.view_scene, self.video_loader)
        self.ui.playBtn.clicked.connect(self.playBtnClicked)
        self.ui.backKeyBtn.clicked.connect(
            lambda: self.ui.frameSlider.setValue(self.ui.frameSlider.value() - 1)
        )
        self.ui.forwardKeyBtn.clicked.connect(
            lambda: self.ui.frameSlider.setValue(self.ui.frameSlider.value() + 1)
        )
        self.ui.frameSlider.valueChanged.connect(self.analyzeFrame)

        self.ui.FrameView.mousePressEvent = self.mousePressEvent

    def bindCheckBox(self):
        """Bind UI CheckBox to their corresponding functions."""
        self.ui.cameraCheckBox.stateChanged.connect(self.toggleCamera)
        self.ui.recordCheckBox.stateChanged.connect(self.toggleRecord)
        self.ui.selectCheckBox.stateChanged.connect(self.toggleSelect)
        self.ui.showSkeletonCheckBox.stateChanged.connect(self.toggleShowSkeleton)
        self.ui.selectKptCheckBox.stateChanged.connect(self.toggleKptSelect)
        self.ui.showAngleCheckBox.stateChanged.connect(self.toggleShowAngleInfo)
        self.ui.showBboxCheckBox.stateChanged.connect(self.toggleShowBbox)
        self.ui.showLineCheckBox.stateChanged.connect(self.toggleShowGrid)  
        self.ui.startPitchCheckBox.stateChanged.connect(self.togglePitching)

    def playBtnClicked(self):
        if self.video_loader.video_name == "":
            QMessageBox.warning(self, "無法播放影片", "請讀取影片!")
            return
        if self.video_loader.is_loading:
            QMessageBox.warning(self, "影片讀取中", "請稍等!")
            return
        self.is_play = not self.is_play
        self.ui.playBtn.setText("||" if self.is_play else "▶︎")
        if self.is_play:
            self.playFrame(self.ui.frameSlider.value())

    def playFrame(self, start_num:int=0):
        for i in range(start_num, self.video_loader.total_frames):
            if not self.is_play:
                break
            self.ui.frameSlider.setValue(i)
            if i == self.video_loader.total_frames - 1 and self.is_play:
                self.playBtnClicked()
            cv2.waitKey(15)

    def videoSilder(self, visible:bool):
        elements = [
            self.ui.backKeyBtn,
            self.ui.playBtn,
            self.ui.forwardKeyBtn,
            self.ui.frameSlider,
            self.ui.frameNumLabel,
            self.ui.CurveView
        ]
        
        for element in elements:
            element.setVisible(visible)

    def changePitcher(self):
        """Change the pitcher based on input value. 9: "左腕", 10: "右腕","""
        kpt_id = 10 if self.ui.pitchInput.currentIndex() == 0 else 9
        self.pose_estimater.setKptId(kpt_id)
        self.pose_estimater.setPitchHandId(kpt_id)

    def toggleCamera(self, state:int):
        """Toggle the camera on/off based on checkbox state."""
        if state == 2:
            if self.is_play:
                self.ui.cameraCheckBox.setCheckState(0)
                QMessageBox.warning(self, "無法開啟相機", "請先暫停播放影片")
                return
            self.reset()
            self.ui.showAngleCheckBox.setCheckState(0)
            self.ui.selectCheckBox.setCheckState(0)
            self.ui.selectKptCheckBox.setCheckState(0)
            self.camera = Camera()
            self.timer = QTimer()
            self.timer.timeout.connect(self.analyzeFrame)
            self.is_video = False
            frame_width, frame_height, fps = self.camera.toggleCamera(True)
            self.model.setImageSize((frame_width, frame_height))
            self.ui.ResolutionLabel.setText(f"(0, 0) - ({frame_width} x {frame_height}), FPS: {fps}")
            self.timer.start(1)
            self.videoSilder(visible=False)
        else:
            if self.camera is not None:
                self.camera.toggleCamera(False)
            if self.timer is not None:
                self.timer.stop()

            self.view_scene.clear()
            self.curve_scene.clear()
            self.is_video = True
            self.videoSilder(visible=True)
            self.camera = None
            self.timer = None

    def togglePitching(self, state):
        if state == 2:
            if not self.ui.cameraCheckBox.isChecked():
                self.ui.startPitchCheckBox.setCheckState(0)
                QMessageBox.warning(self, "無法開始投球模式", "請先開啟相機")
                return
            if not self.ui.showSkeletonCheckBox.isChecked():
                self.ui.startPitchCheckBox.setCheckState(0)
                QMessageBox.warning(self, "無法開始投球模式", "請選擇顯示人體骨架!")
                return
            if not self.ui.selectCheckBox.isChecked():
                self.ui.startPitchCheckBox.setCheckState(0)
                QMessageBox.warning(self, "無法開始投球模式", "請選擇人!")
                return
            self.is_pitching = True
            self.record_checker = JointAreaChecker(self.camera.frame_size)
        else:
            self.is_pitching = False

    def toggleRecord(self, state:int):
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
            self.ui.recordCheckBox.setCheckState(0)
            return
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = f'../../Db/Record/C{self.ui.cameraIdInput.value()}_Fps120_{current_time}'
        os.makedirs(output_dir, exist_ok=True)
        video_filename = os.path.join(output_dir, f'C{self.ui.cameraIdInput.value()}_Fps120_{current_time}.mp4')
        self.camera.startRecording(video_filename)

    def toggleSelect(self, state:int):
        """Select a person based on checkbox state."""
        if state == 2:
            if not self.ui.showSkeletonCheckBox.isChecked():
                self.ui.selectCheckBox.setCheckState(0)
                QMessageBox.warning(self, "無法選擇人", "請選擇顯示人體骨架!")
                return
            self.person_selector = PersonSelector()
            frame_num = self.ui.frameSlider.value() if self.is_video else None
            search_person_df = self.pose_estimater.getPersonDf(frame_num=frame_num) if frame_num is not None else self.pose_estimater.pre_person_df
            self.person_selector.select(search_person_df = search_person_df)
            self.pose_estimater.setPersonId(self.person_selector.selected_id)
        else:
            self.pose_estimater.setPersonId(None)
            self.person_selector = None

    def toggleKptSelect(self, state:int):
        """Toggle keypoint selection and trajectory visualization."""  
        if state ==2:
            if not self.ui.selectCheckBox.isChecked():
                self.ui.selectKptCheckBox.setCheckState(0)
                QMessageBox.warning(self, "無法選擇關節點", "請選擇人!")
                return
            self.kpt_selector = KptSelector()
            self.pose_estimater.setKptId(10)
            self.image_drawer.setShowTraj(True)
        else:
            self.kpt_selector = None
            self.pose_estimater.setKptId(None)
            self.pose_estimater.clearKptBuffer()
            self.image_drawer.setShowTraj(False)

    def toggleShowSkeleton(self, state:int):
        """Toggle skeleton detection and FPS control."""
        print("skeleton: "+ str(state))
        if state == 2 and self.ui.recordCheckBox.isChecked():
            self.ui.showSkeletonCheckBox.setCheckState(0)
            QMessageBox.warning(self, "無法選擇人", "請選擇顯示人體骨架!")
            return
        is_checked = state == 2
        if not is_checked:
            self.ui.showAngleCheckBox.setCheckState(0)
            self.ui.selectKptCheckBox.setCheckState(0)
            self.ui.selectCheckBox.setCheckState(0)
            
        self.pose_estimater.setDetect(is_checked)
        self.image_drawer.setShowSkeleton(is_checked)
        if self.camera is not None and not self.is_video:
            self.camera.setFPSControl(15 if is_checked else 1)

    def toggleShowBbox(self, state:int):
        """Toggle bounding box visibility."""
        self.image_drawer.setShowBbox(state == 2)

    def toggleShowAngleInfo(self, state:int):
        if not self.ui.selectCheckBox.isChecked():
            self.ui.showAngleCheckBox.setCheckState(0)
            QMessageBox.warning(self, "無法顯示關節點角度資訊", "請選擇人!")
            return
        if state == 2:  
            self.image_drawer.setShowAngleInfo(True)
        else:
            self.image_drawer.setShowAngleInfo(False)

    def toggleShowGrid(self, state:int):
        """Toggle gridline visibility."""
        self.image_drawer.setShowGrid(state == 2)
    
    def changeCamera(self):
        """Change the camera based on input value."""
        if self.camera is not None:
            self.camera.setCameraId(self.ui.cameraIdInput.value())

    def analyzeFrame(self):
        """Analyze and process each frame from the camera or video"""
        fps = 0
        if self.is_video:
            frame_num = self.ui.frameSlider.value()
            self.ui.frameNumLabel.setText(f'{frame_num}/{len(self.video_loader.video_frames) - 1}')
            frame = self.video_loader.getVideoImage(frame_num)
            _, _, fps= self.pose_estimater.detectKpt(frame, frame_num, is_video=True)
            if self.pose_estimater.person_id is not None:
                self.pose_analyzer.addAnalyzeInfo(frame_num)
                self.graph_plotter.updateGraph(frame_num)
                self.kpt_table.importDataToTable(frame_num)
            
            self.updateFrame(frame_num=frame_num)

            if frame_num == self.video_loader.total_frames - 1:
                self.handleVideoEnd()

        else:
            if not self.camera.frame_buffer.empty():
                frame = self.camera.frame_buffer.get().copy()
                _, _, fps = self.pose_estimater.detectKpt(frame, is_video=False)
                if self.is_pitching:
                    self.pitherAnaylze()
                self.updateFrame(frame=frame)
                
        self.ui.FPSInfoLabel.setText(f"{fps:02d}")
    
    def handleVideoEnd(self):
        """Handle the logic when video reaches its end."""
        self.play_times -= 1
        self.video_loader.saveVideo()

        if self.play_times > 0:
            # Replay the video
            self.playBtnClicked()
            self.ui.frameSlider.setValue(0)
            self.playBtnClicked()
        else:
            # Stop playback and reset
            self.playBtnClicked()
            self.ui.cameraCheckBox.setCheckState(2)
            self.ui.showSkeletonCheckBox.setCheckState(0)
            self.ui.showSkeletonCheckBox.setCheckState(2)
            self.play_times = 2

    def pitherAnaylze(self):
        if self.record_checker is not None:
            pos = self.pose_estimater.getPrePersonDf()
            if self.record_checker.is_joint_in_area(pos):
                self.initCountdownTimer(2)
        else:
            self.initRecorderTimer(2)

    def initCountdownTimer(self, duration:int):
        if self.countdown_timer is None:
            self.countdown_timer = Timer(duration)
            self.countdown_timer.start()
    
    def initRecorderTimer(self, duration:int):
        if self.record_timer is None:
            self.ui.showBboxCheckBox.setCheckState(0)
            self.ui.selectKptCheckBox.setCheckState(0)
            self.ui.selectCheckBox.setCheckState(0)
            self.ui.showSkeletonCheckBox.setCheckState(0)
            self.ui.recordCheckBox.setCheckState(2)
            self.record_timer = Timer(duration)
            self.record_timer.start()

    def updateFrame(self, frame: np.ndarray = None, frame_num:int = None):
        """Update the displayed frame with additional analysis."""
        # 更新當前的frame和frame_num
        if self.is_video and frame_num is not None:
            frame = self.video_loader.getVideoImage(frame_num)
        countdown_time = self.updateTimers() 
        drawed_img = self.image_drawer.drawInfo(frame, frame_num, self.pose_estimater.kpt_buffer, countdown_time)
        self.showImage(drawed_img, self.view_scene, self.ui.FrameView)

    def updateTimers(self):
        countdown_time = None
        if self.countdown_timer is not None:
            countdown_time = self.countdown_timer.get_remaining_time()
            if countdown_time == 0:
                self.resetCountdownTimer()
        if self.record_timer is not None:
            countdown_time = self.record_timer.get_remaining_time()
            if countdown_time == 0:
                self.resetRecordTimer()
        return countdown_time

    def resetCountdownTimer(self):
        self.countdown_timer = None
        self.record_checker = None

    def resetRecordTimer(self):
        self.record_timer = None
        video_path = self.camera.video_path
        self.video_loader.reset()
        self.ui.recordCheckBox.setCheckState(0)
        self.ui.startPitchCheckBox.setCheckState(0)
        self.ui.cameraCheckBox.setCheckState(0)
        self.video_loader.loadVideo(video_path)
        self.checkVideoLoad()

    def checkVideoLoad(self):
        """檢查影片是否讀取完成，並更新 UI 元素。"""
        # 檢查是否有影片名稱，若無則不執行後續操作
        if not self.video_loader.video_name:
            return
        # 若影片正在讀取中，定時檢查讀取狀況
        if self.video_loader.is_loading:
            # self.ui.videoNameLabel.setText("讀取影片中")
            QTimer.singleShot(100, self.checkVideoLoad)  # 每100ms 檢查一次
            return
        # 影片讀取完成後更新 UI 元素
        self.updateVideoInfo()
        self.initAnalyzeFrame()

    def updateVideoInfo(self):
        """更新與影片相關的資訊顯示在 UI 上。"""
        self.reset()
        self.initFrameSlider()
        self.initGraph()
        self.updateFrame(frame_num=0)
        self.model.setImageSize(self.video_loader.video_size)
        video_size = self.video_loader.video_size
        self.ui.ResolutionLabel.setText(f"(0,0) - {video_size[0]} x {video_size[1]}")

    def initAnalyzeFrame(self):
        self.ui.showSkeletonCheckBox.setCheckState(2)
        frame = self.video_loader.getVideoImage(0)
        _, _, _= self.pose_estimater.detectKpt(frame, 0, is_video=True)
        self.ui.selectCheckBox.setCheckState(2)
        self.ui.selectKptCheckBox.setCheckState(2)
        self.ui.showAngleCheckBox.setCheckState(2)
        self.ui.playBtn.click()

    def initFrameSlider(self):
        """初始化影片滑桿和相關的標籤。"""
        total_frames = self.video_loader.total_frames
        self.ui.frameSlider.setMinimum(0)
        self.ui.frameSlider.setMaximum(total_frames - 1)
        self.ui.frameSlider.setValue(0)
        self.ui.frameNumLabel.setText(f'0/{total_frames - 1}')

    def resetFrameSlider(self):
        self.ui.frameSlider.setValue(0)
        self.ui.frameSlider.setRange(0, 0)

    def initGraph(self):
        """初始化圖表和模型設定。"""
        total_frames = self.video_loader.total_frames
        print("video frames: "+ str(total_frames))
        self.graph_plotter._init_graph(total_frames) 
        self.showGraph(self.curve_scene, self.ui.CurveView)

    def showGraph(self, scene:QGraphicsScene, graphicview:QGraphicsView):
        scene.clear()
        graph = self.graph_plotter.graph
        graph.resize(graphicview.width(),graphicview.height())
        scene.addWidget(graph)
        graphicview.setScene(scene)
        graphicview.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def reset(self):
        self.pose_estimater.reset()
        self.pose_analyzer.reset()
        self.graph_plotter.reset()
        self.image_drawer.reset()
        self.resetFrameSlider()
        self.view_scene.clear()
        self.curve_scene.clear()

    def showImage(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView):
        """Display an image in the QGraphicsView."""
        scene.clear()
        if image is None:
            print(self.video_loader.total_frames)
            # print(self.video_loader.video_frames)
            exit()
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
        search_person_df = self.pose_estimater.getPersonDf(frame_num = self.ui.frameSlider.value()) if self.is_video else self.pose_estimater.pre_person_df 

        if self.ui.selectCheckBox.isChecked() and not self.kpt_table.label_kpt :
            if event.button() == Qt.LeftButton:
                self.person_selector.select(x, y, search_person_df)
                self.pose_estimater.setPersonId(self.person_selector.selected_id)

        if self.ui.selectKptCheckBox.isChecked() and not self.kpt_table.label_kpt :
            if event.button() == Qt.LeftButton:
                self.kpt_selector.select(x, y, search_person_df)
                self.pose_estimater.setKptId(self.kpt_selector.selected_id)

        if self.kpt_table.label_kpt:
            if event.button() == Qt.LeftButton:
                self.kpt_table.sendToTable(x, y, 1, self.ui.frameSlider.value())
            elif event.button() == Qt.RightButton:
                self.kpt_table.sendToTable(0, 0, 0, self.ui.frameSlider.value())

        if self.is_video:
            self.updateFrame(frame_num=self.ui.frameSlider.value())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PosePitchTabControl()
    window.show()
    sys.exit(app.exec_())
