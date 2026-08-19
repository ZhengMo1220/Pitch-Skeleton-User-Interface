from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from collections import deque
import numpy as np
import sys
import os
from pitch_ui_2tabs_beta_sync_v4 import Ui_pitch_ui
from playVideo_2 import FullscreenVideoDialog
from datetime import datetime
from utils.timer import Timer
from cv_utils.cv_control import Camera, VideoLoader
from utils.selector import PersonSelector, KptSelector
from utils.analyze_3d import PoseAnalyzer
import cv2
from utils.vis_image import ImageDrawer
from skeleton.detect_skeleton import PoseEstimater
import pyqtgraph as pg
from utils.model import Model
from PitchingAnalyzer import PitchingAnalyzer
from triangulate_3d_viewer import Triangulate3DViewer
from cv_utils.cv_thread import FramesToVideoWriterThread


class Triangulate3DThread(QThread):
    """後台線程處理 3D 三角測量，避免阻塞 UI"""
    finished = pyqtSignal()
    
    def __init__(self, viewer_3d, frame_L, frame_R, frame_num):
        super().__init__()
        self.viewer_3d = viewer_3d
        self.frame_L = frame_L
        self.frame_R = frame_R
        self.frame_num = frame_num
    
    def run(self):
        try:
            self.viewer_3d.add_2d_keypoints(self.frame_L, self.frame_R, self.frame_num)
        except Exception as e:
            print(f"3D triangulation error: {e}")
        finally:
            self.finished.emit()

class PosePitchTabControl(QWidget):
    def __init__(self, model:Model, parent=None):
        super().__init__(parent)
        self.ui = Ui_pitch_ui()
        self.ui.setupUi(self)
        self.model = model
        self.setupComponents()
        self.initVar()
        self.bindUI()
        self.fullscreen_view_1 = None  # 對應 FrameView
        self.fullscreen_view_2 = None  # 對應 FrameView_2
        self.play2videos = False
        self.triangulate_thread = None  # 3D 三角測量後台線程
        self.load_video_list("../../Db/Record")
        self.video_filename = None  # 用來保存最新錄製的影片路徑
        self.video_filename_2 = None

    def initVar(self):
        """Initialize variables and timer."""
        self.is_video = True if self.camera is None else False
        self.view_scene = QGraphicsScene()
        self.view_scene_2 = QGraphicsScene()
        self.view_scene.clear()
        self.view_scene_2.clear()
        self.is_pitching = False
        self.output_dir = None
        self.viewer_3d = None  # 3D 視圖
        # 投球錄影快取（已由 Camera 負責管理）
        self.is_auto_recording = False
        self.camera_fps = 60
        self.video_writer_thread = None  # 後台寫入線程
        
        # 關鍵幀相關
        self.fc_frame_num = None  # Frame at Contact (接觸點)
        self.mer_frame_num = None  # Mid-External Rotation (中等外旋)
        self.br_frame_num = None   # Back Rotation (背部旋轉)
        self.key_frames_scene_fc = QGraphicsScene()
        self.key_frames_scene_mer = QGraphicsScene()
        self.key_frames_scene_br = QGraphicsScene()
        self.video_saved = False  # 標記影片是否已儲存
        
        self.ui.redRatioSlider.setMinimum(25)
        self.ui.redRatioSlider.setMaximum(400)
        self.ui.blueRatioSlider.setMinimum(25)
        self.ui.blueRatioSlider.setMaximum(400)
        self.ui.gainSlider.setMinimum(0)
        self.ui.gainSlider.setMaximum(290)
        self.initVideoVar()  

    def initVideoVar(self):
        self.is_play = False
        self.is_processed = False
        self.play_times = 2
        self.video_saved = False  # 重置影片儲存標記
        pg.setConfigOptions(foreground=QColor(113,148,116), antialias = True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

    def setupComponents(self): 
        self.K_F = np.array([
            [7131.197489913678, 0.0, 1190.6066203243458],
            [0.0, 7107.943256043428, 665.2288622157741],
            [0.0, 0.0, 1.0]
        ])
        self.K_S = np.array([
            [1063.0766604691114, 0.0, 962.1115766023166],
            [0.0, 1061.6146093201994, 570.8085693424371],
            [0.0, 0.0, 1.0]
        ])
        self.F = np.array([
            [-1.23527137e-06,  3.74634658e-05, -1.36469340e-02],
            [1.59530549e-05,  9.03916270e-07, -6.06394789e-02],
            [-4.66549815e-03,  3.39782280e-02,  1.00000000e+00]
        ])  
        self.camera = Camera()
        self.timer = QTimer()
        self.timer.timeout.connect(self.analyzeFrame)

        self.countdown_timer = None
        self.record_timer = None
        self.pose_estimater = PoseEstimater(self.model)
        self.pose_estimater_2 = PoseEstimater(self.model)
        self.pose_analyzer = PoseAnalyzer(self.pose_estimater, self.K_S, self.K_F, self.F)
        self.pose_analyzer_2 = PoseAnalyzer(self.pose_estimater_2, self.K_S, self.K_F, self.F)
        self.image_drawer = ImageDrawer(self.pose_estimater, self.pose_analyzer)
        self.image_drawer_2 = ImageDrawer(self.pose_estimater_2, self.pose_analyzer_2)
        self.video_loader = VideoLoader(self.image_drawer,self.image_drawer_2)
        
        # 投球動作分析器
        self.pitching_analyzer = PitchingAnalyzer()

        self.ui.videoTree.setSelectionMode(QAbstractItemView.MultiSelection)
        self.tree =  self.ui.videoTree

    def resizeEvent(self, event):
        new_size = event.size()
        print(f"PoseCameraTabControl resized to: {new_size.width()}x{new_size.height()}")
        # 在此執行你想要的操作
        if self.video_loader.video_name is not None:
            self.updateFrame(frame_num=self.ui.frameSlider.value())
        super().resizeEvent(event)

    def bindUI(self):
        """Bind UI element to their corresponding functions."""
        self.ui.pitchInput.currentIndexChanged.connect(self.changePitcher)
        self.bindVideoUI()
        self.bindCheckBox()
         # 綁定白平衡滑桿
        self.ui.redRatioSlider.valueChanged.connect(self.onRedRatioChanged)
        self.ui.blueRatioSlider.valueChanged.connect(self.onBlueRatioChanged)
        self.ui.gainSlider.valueChanged.connect(self.onGainChanged)
        # 綁定側面攝影機 radio button
        self.ui.sideCamera.toggled.connect(self.updateCameraSliders) 
        # self.ui.frontCamera.toggled.connect(self.updateCameraSliders)    
    
    def bindVideoUI(self):
        self.tree.itemSelectionChanged.connect(self.play_video_from_item)
        self.ui.slowMotionInput.currentIndexChanged.connect(self.updatePlaybackRate)
        self.ui.playBtn.clicked.connect(self.playBtnClicked)
        self.ui.backKeyBtn.clicked.connect(
            lambda: self.ui.frameSlider.setValue(self.ui.frameSlider.value() - 1)
        )
        self.ui.forwardKeyBtn.clicked.connect(
            lambda: self.ui.frameSlider.setValue(self.ui.frameSlider.value() + 1)
        )
        self.ui.frameSlider.valueChanged.connect(self.analyzeFrame)
        self.ui.frameSlider.valueChanged.connect(self.syncFullscreenSlider)

        self.ui.FrameView.mousePressEvent = self.mousePressEvent
        self.ui.FrameView_2.mousePressEvent = self.mousePressEvent

    def bindCheckBox(self):
        """Bind UI CheckBox to their corresponding functions."""
        self.ui.cameraCheckBox.stateChanged.connect(self.toggleCamera)
        self.ui.recordCheckBox.stateChanged.connect(self.toggleRecord)
        self.ui.selectCheckBox.stateChanged.connect(self.toggleSelect)
        self.ui.showSkeletonCheckBox.stateChanged.connect(self.toggleShowSkeleton)
        self.ui.selectKptCheckBox.stateChanged.connect(self.toggleKptSelect)
        self.ui.showBboxCheckBox.stateChanged.connect(self.toggleShowBbox)
        self.ui.showLineCheckBox.stateChanged.connect(self.toggleShowGrid)  
        self.ui.startPitchCheckBox.stateChanged.connect(self.togglePitching)
        self.ui.skeletonVideoCheckBox.stateChanged.connect(self.toggleSkeletonVideo)

    def onRedRatioChanged(self, value):
        ratio = value / 100.0  # 假設滑桿範圍 100~400，對應 1.0~4.0
        self.ui.red_ratio_value.setText(f"{ratio:.2f}")
        cam = self.get_selected_camera()
        if cam is not None:
            cam.update_white_balance(red_ratio=ratio)

    def onBlueRatioChanged(self, value):
        ratio = value / 100.0
        self.ui.blue_ratio_value.setText(f"{ratio:.2f}")
        cam = self.get_selected_camera()
        if cam is not None:
            cam.update_white_balance(blue_ratio=ratio)

    def onGainChanged(self, value):
        ratio = value / 10.0
        self.ui.gain_value.setText(f"{ratio:.1f}")
        cam = self.get_selected_camera()
        if cam is not None:
            cam.update_white_balance(gain=ratio)

    def playBtnClicked(self):
        if self.video_loader.video_name == "" or self.video_loader.video_name_2 == "":
            QMessageBox.warning(self, "無法播放影片", "請讀取影片!")
            return
        if self.video_loader.is_loading:
            QMessageBox.warning(self, "影片讀取中", "請稍等!")
            return
        self.is_play = not self.is_play
        self.ui.playBtn.setText("||" if self.is_play else "▶︎")
        if self.is_play:
            # 開始播放時自動啟用骨架檢測
            self.ui.skeletonVideoCheckBox.setCheckState(2)
            self.playFrame(self.ui.frameSlider.value())
        else:
            if hasattr(self, "_play_timer") and self._play_timer.isActive():
                self._play_timer.stop()

    def playFrame(self, start_num:int=0):
        """Start playing frames using a timer instead of blocking calls."""
        if not hasattr(self, "_play_timer"):
            self._play_timer = QTimer()
            self._play_timer.timeout.connect(self._playNextFrame)

        self._current_play_frame = start_num
        self._play_timer.start(self._get_play_interval_ms())

    def _playNextFrame(self):
        """Internal method to play the next frame."""
        if not self.is_play or self._current_play_frame >= self.video_loader.total_frames:
            if hasattr(self, "_play_timer"):
                self._play_timer.stop()
            if self.is_play:
                self.playBtnClicked()
            return

        self.ui.frameSlider.setValue(self._current_play_frame)
        self._current_play_frame += 1

    def _get_play_interval_ms(self) -> int:
        """Compute playback interval based on video fps and UI playback rate."""
        base_fps = self.video_loader.video_fps if self.video_loader.video_fps else 60
        try:
            rate = float(self.ui.slowMotionInput.currentText())
        except Exception:
            rate = 1.0
        rate = max(rate, 0.01)
        interval_ms = int(1000 / (base_fps * rate))
        return max(interval_ms, 1)

    def videoSilder(self, visible:bool):
        elements = [
            self.ui.backKeyBtn,
            self.ui.playBtn,
            self.ui.forwardKeyBtn,
            self.ui.frameSlider,
            self.ui.frameNumLabel,
            self.ui.fcView, self.ui.FC_label,
            self.ui.merView, self.ui.MER_label,
            self.ui.brView, self.ui.BR_label
        ]
        
        for element in elements:
            element.setVisible(visible)

    def changePitcher(self):
        """Change the pitcher based on input value. 9: "左腕", 10: "右腕","""
        kpt_id = 10 if self.ui.pitchInput.currentIndex() == 0 else 9
        self.pose_estimater.setKptId(kpt_id)
        self.pose_estimater.setPitchHandId(kpt_id)

    def get_selected_camera(self):
        """根據 radio button 回傳目前選擇的攝影機物件"""
        if self.ui.sideCamera.isChecked():
            return self.camera.video_thread.camera1
        else:
            self.ui.frontCamera.setChecked(True)
            return self.camera.video_thread.camera2
        
    def updateCameraSliders(self):
        """根據目前選擇的攝影機更新 slider 和 label 的值"""
        cam = self.get_selected_camera()
        try:
            gain_value = cam.full_config['gain_settings']['gain_value']
            redRatio_value = cam.full_config['white_balance_settings']['white_balance_red_ratio']
            blueRatio_value = cam.full_config['white_balance_settings']['white_balance_blue_ratio']
            gain_slider_value = int(gain_value * 10)
            red_slider_value = int(redRatio_value * 100)
            blue_slider_value = int(blueRatio_value * 100)
            self.ui.gainSlider.setValue(gain_slider_value)
            self.ui.gain_value.setText(f"{gain_value:.1f}")
            self.ui.redRatioSlider.setValue(red_slider_value)
            self.ui.red_ratio_value.setText(f"{redRatio_value:.2f}")
            self.ui.blueRatioSlider.setValue(blue_slider_value)
            self.ui.blue_ratio_value.setText(f"{blueRatio_value:.2f}")
        except Exception as e:
            print(f"讀取 config 設定失敗: {e}")

    def toggleCamera(self, state:int):
        """Toggle the camera on/off based on checkbox state."""
        if state == 2:
            self.reset()
            self.ui.selectCheckBox.setCheckState(0)
            self.ui.selectKptCheckBox.setCheckState(0)
            self.is_video = False
            frame_width, frame_height, fps = self.camera.toggleCamera(True)
            self.camera_fps = fps if fps else 60
            self.model.setImageSize((frame_width, frame_height))
            self.ui.ResolutionLabel.setText(f"(0, 0) - ({frame_width} x {frame_height}), FPS: {fps}")
            self.timer.start(1)
            self.videoSilder(visible=False)
            self.updateCameraSliders()
        else:
            if self.camera is not None:
                self.camera.toggleCamera(False)
            if self.timer is not None:
                self.timer.stop()

            self.view_scene.clear()
            self.view_scene_2.clear()
            self.ui.selectCheckBox.setCheckState(0)
            self.ui.showSkeletonCheckBox.setCheckState(0)
            self.is_video = True
            self.videoSilder(visible=True)
            
            # 清理 GPU 記憶體
            if self.model is not None:
                self.model.clear_gpu_cache()

    def togglePitching(self, state):
        if state == 2:
            if not self.ui.cameraCheckBox.isChecked():
                self.ui.startPitchCheckBox.setCheckState(0)
                QMessageBox.warning(self, "無法開始投球模式", "請先開啟相機")
                return
            if not self.ui.showSkeletonCheckBox.isChecked():
                self.ui.showSkeletonCheckBox.setCheckState(2)
             
            self.is_pitching = True
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
            self.load_video_list("../../Db/Record")

    def startRecording(self):
        """Start recording the video."""
        if self.camera is None:
            self.ui.recordCheckBox.setCheckState(0)
            return
        current_date = datetime.now().strftime("%Y%m%d")
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f'../../Db/Record/Pitcher{self.ui.PitcherID.currentText()}_{current_date}/{current_time}'
        os.makedirs(self.output_dir, exist_ok=True)
        self.video_filename = os.path.join(self.output_dir, f'CS_Fps180_{current_time}.mp4')
        self.video_filename_2 = os.path.join(self.output_dir, f'CF_Fps180_{current_time}.mp4')
        self.camera.startRecording(self.video_filename, self.video_filename_2)

    def toggleSelect(self, state:int):
        """Select a person based on checkbox state."""
        if state == 2:
            if not self.ui.showSkeletonCheckBox.isChecked():
                self.ui.selectCheckBox.setCheckState(0)
                QMessageBox.warning(self, "無法選擇人", "請選擇顯示人體骨架!")
                return
            self.person_selector = PersonSelector()
            self.person_selector_2 = PersonSelector()
            frame_num = self.ui.frameSlider.value() if self.is_video else None
            search_person_df = self.pose_estimater.getPersonDf(frame_num=frame_num) if frame_num is not None else self.pose_estimater.pre_person_df
            search_person_df_2 = self.pose_estimater_2.getPersonDf(frame_num=frame_num) if frame_num is not None else self.pose_estimater_2.pre_person_df
            self.person_selector.select(search_person_df = search_person_df)
            self.person_selector_2.select(search_person_df = search_person_df_2)
            self.pose_estimater.setPersonId(self.person_selector.selected_id)
            self.pose_estimater_2.setPersonId(self.person_selector_2.selected_id)
        else:
            self.pose_estimater.setPersonId(None)
            self.pose_estimater_2.setPersonId(None)
            self.person_selector = None
            self.person_selector_2 = None

    def toggleKptSelect(self, state:int):
        """Toggle keypoint selection and trajectory visualization."""  
        if state ==2:
            if not self.ui.selectCheckBox.isChecked():
                self.ui.selectKptCheckBox.setCheckState(0)
                QMessageBox.warning(self, "無法選擇關節點", "請選擇人!")
                return
            self.kpt_selector = KptSelector()
            self.pose_estimater.setKptId(10)
            self.pose_estimater_2.setKptId(10)
            self.image_drawer.setShowTraj(True)
            self.image_drawer_2.setShowTraj(True)
        else:
            self.kpt_selector = None
            self.pose_estimater.setKptId(None)
            self.pose_estimater_2.setKptId(None)
            self.pose_estimater.clearKptBuffer()
            self.pose_estimater_2.clearKptBuffer()
            self.image_drawer.setShowTraj(False)
            self.image_drawer_2.setShowTraj(False)

    def toggleShowSkeleton(self, state:int):
        """Toggle skeleton detection and FPS control."""
        print("skeleton state: "+ str(state))
        if state == 2 and self.ui.recordCheckBox.isChecked():
            self.ui.showSkeletonCheckBox.setCheckState(0)
            QMessageBox.warning(self, "無法顯示人體骨架!", "正在錄影中!")
            return
        is_checked = state == 2
        if not is_checked:
            self.ui.selectKptCheckBox.setCheckState(0)
            self.ui.selectCheckBox.setCheckState(0)
            
        self.pose_estimater.setDetect(is_checked)
        self.pose_estimater_2.setDetect(is_checked)
        self.image_drawer.setShowSkeleton(is_checked)
        self.image_drawer_2.setShowSkeleton(is_checked)
        # if self.camera is not None and not self.is_video:
        #     self.camera.setFPSControl(30 if is_checked else 15)

    def toggleSkeletonVideo(self, state:int):
        if state==2:
            self.updateVideoInfo()
            self.initAnalyzeFrame()
        else:
            self.reset()

    def toggleShowBbox(self, state:int):
        """Toggle bounding box visibility."""
        self.image_drawer.setShowBbox(state == 2)
        self.image_drawer_2.setShowBbox(state == 2)

    def toggleShowGrid(self, state:int):
        """Toggle gridline visibility."""
        self.image_drawer.setShowGrid(state == 2)
        self.image_drawer_2.setShowGrid(state == 2)

    def analyzeFrame(self):
        """Analyze and process each frame from the camera or video"""
        fps = 0
        if self.is_video:
            frame_num = self.ui.frameSlider.value()
            self.ui.frameNumLabel.setText(f'{frame_num}/{self.video_loader.total_frames - 1}')
            frame,frame_2 = self.video_loader.getVideoImage(frame_num)
            if self.ui.skeletonVideoCheckBox.isChecked():
                _, _, fps = self.pose_estimater.detectKpt(frame, frame_num, is_video=True)
                # _, _, fps = self.pose_estimater_2.detectKpt(frame_2, frame_num, is_video=True)
                
                if self.viewer_3d:
                # 取得2D骨架資料
                    person_df_L = self.pose_estimater.getPersonDf(frame_num=frame_num)
                    person_df_R = self.pose_estimater_2.getPersonDf(frame_num=frame_num)
                    if person_df_L is not None and person_df_R is not None and not person_df_L.empty and not person_df_R.empty:
                        # 取第一個人的 keypoints
                        frame_L = {"keypoints": person_df_L.iloc[0]["keypoints"]}
                        frame_R = {"keypoints": person_df_R.iloc[0]["keypoints"]}
                        # 使用後台線程處理 3D 三角測量，避免阻塞 UI
                        self.triangulate_thread = Triangulate3DThread(
                            self.viewer_3d, frame_L, frame_R, frame_num
                        )
                        self.triangulate_thread.finished.connect(
                            lambda: self.pose_analyzer.addAnalyzeInfo(frame_num, frame_L, frame_R)
                        )
                        self.triangulate_thread.start()
            self.updateFrame(frame_num=frame_num)
            _, shouhip_angle_value = self.pose_analyzer.get_frame_angleOrSpeed_data(frame_num, speed_name="shoulder_hiple_angle")
            # print(f"Frame {frame_num} - Shoulder-Hip Angle: {shouhip_angle_value}")
            # 每10幀檢測一次關鍵幀
            if frame_num % 10 == 0 and frame_num > 0:
                self.auto_detect_key_frames()
            
            if frame_num == self.video_loader.total_frames - 1:
                self.handleVideoEnd()

        else:
            if not self.camera.frame_buffer.empty() and not self.camera.frame_buffer_2.empty():
                # import time
                # start_time = time.perf_counter()
                frame = self.camera.frame_buffer.get().copy()
                frame_2 = self.camera.frame_buffer_2.get().copy()
                # pre_frames 終由由都由由更新（已由由 Camera.buffer_frame() 中轉移）
                # if self.is_auto_recording:
                #     self.record_frames.append((frame.copy(), frame_2.copy()))
                #     print(f"Auto recording... Total recorded frames: {len(self.record_frames)}")
                # 只偵測側面攝影機以降低運算負擔
                _, _, fps = self.pose_estimater.detectKpt(frame, is_video=False)
                # _, _, fps = self.pose_estimater_2.detectKpt(frame_2, is_video=False)  # 已停用正面攝影機骨架偵測
                if self.is_pitching:
                    if self.ui.startPitchCheckBox.isChecked() and not self.ui.recordCheckBox.isChecked() and self.pose_estimater.person_id is None:
                        self.ui.selectCheckBox.setCheckState(0)
                        self.ui.showSkeletonCheckBox.setCheckState(0)
                        self.ui.showSkeletonCheckBox.setCheckState(2)
                        self.ui.selectCheckBox.setCheckState(2)
                    self.pitherAnaylze()
                self.updateFrame(frame=frame, frame_2=frame_2)
                
        self.ui.FPSInfoLabel.setText(f"{fps:02d}")
    
    def set_key_frame(self, frame_type: str, frame_num: int):
        """
        設置關鍵幀
        
        Args:
            frame_type: 'FC', 'MER', 或 'BR'
            frame_num: 幀號
        """
        if frame_type == 'FC':
            self.fc_frame_num = frame_num
        elif frame_type == 'MER':
            self.mer_frame_num = frame_num
        elif frame_type == 'BR':
            self.br_frame_num = frame_num
        
        # 立即顯示該關鍵幀
        self.display_key_frames()
    
    def display_key_frames(self):
        """顯示三個關鍵幀到對應的視圖"""
        if not self.is_video:
            return
        
        # 顯示 FC (Frame at Contact)
        if self.fc_frame_num is not None and 0 <= self.fc_frame_num < self.video_loader.total_frames:
            frame_fc, _ = self.video_loader.getVideoImage(self.fc_frame_num)
            self.show_key_frame(frame_fc, self.key_frames_scene_fc, self.ui.fcView, "FC")
        
        # 顯示 MER (Mid-External Rotation)
        if self.mer_frame_num is not None and 0 <= self.mer_frame_num < self.video_loader.total_frames:
            frame_mer, _ = self.video_loader.getVideoImage(self.mer_frame_num)
            self.show_key_frame(frame_mer, self.key_frames_scene_mer, self.ui.merView, "MER")
        
        # 顯示 BR (Back Rotation)
        if self.br_frame_num is not None and 0 <= self.br_frame_num < self.video_loader.total_frames:
            frame_br, _ = self.video_loader.getVideoImage(self.br_frame_num)
            self.show_key_frame(frame_br, self.key_frames_scene_br, self.ui.brView, "BR")
    
    def show_key_frame(self, frame: np.ndarray, scene: QGraphicsScene, view: QGraphicsView, label_text: str):
        """顯示單個關鍵幀"""
        if frame is None:
            return
        
        scene.clear()
        h, w = frame.shape[:2]
        
        # 縮小幀以適應視圖（可選，根據需要調整）
        scale_factor = 0.5
        frame_resized = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
        
        qImg = QImage(frame_resized, frame_resized.shape[1], frame_resized.shape[0], 
                     QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        scene.addPixmap(pixmap)
        view.setScene(scene)
        view.setAlignment(Qt.AlignCenter)
        view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
    
    def auto_detect_key_frames(self):
        """
        根據 analyze_3d 的數據自動檢測三個關鍵幀
        FC: Frame at Contact (接觸點)
        MER: Mid-External Rotation (中等外旋)
        BR: Back Rotation (背部旋轉)
        """
        if not self.is_video or self.pose_analyzer is None:
            return
        
        if self.pose_analyzer.analyze_df.empty:
            print("[KeyFrame] No analysis data available yet")
            return
        
        # 獲取所有已分析的幀號
        all_frames = sorted(self.pose_analyzer.analyze_df['frame_number'].unique().tolist())
        
        # 至少需要5幀數據才進行檢測，避免數據不足導致的不準確
        if len(all_frames) < 5:
            print(f"[KeyFrame] 數據不足，跳過檢測 (當前: {len(all_frames)}/5 幀)")
            return
        
        print(f"[KeyFrame] 檢測關鍵幀... 共有 {len(all_frames)} 幀數據")
        
        # FC (Frame at Contact): 腳踝速度低且膝蓋角度在特定範圍
        # 這通常發生在投球動作的接觸點
        fc_frame = self.pose_analyzer.find_first_matching_frame(
            all_frames,
            ankle_speed_max=0.8,
            knee_angle_min=40,
            knee_angle_max=50,
            sh_angle_min=25
        )
        if fc_frame is not None:
            self.fc_frame_num = fc_frame
            print(f"[KeyFrame] FC detected at frame {fc_frame}")
        
        # MER (Mid-External Rotation): 肩膀外旋中期
        # 使用肩膀-臀部角度來判斷外旋程度
        mer_frame = None
        if len(all_frames) >= 3:
            # 查找肩膀-臀部角度最大值的幀（代表最大外旋）
            sh_hip_angles = []
            for frame in all_frames:
                _, angle = self.pose_analyzer.get_frame_angleOrSpeed_data(
                    frame, speed_name='shoulder_hiple_angle'
                )
                sh_hip_angles.append((frame, angle))
            
            # 找最大角度的幀（外旋最充分）
            if sh_hip_angles:
                sh_hip_angles = [(f, a) for f, a in sh_hip_angles if not (isinstance(a, float) and np.isnan(a))]
                if sh_hip_angles:
                    mer_frame = max(sh_hip_angles, key=lambda x: x[1])[0]
                    self.mer_frame_num = mer_frame
                    print(f"[KeyFrame] MER detected at frame {mer_frame}")
        
        # BR (Back Rotation): 最後的背部旋轉階段
        # 使用腳踝速度變化來判斷
        br_frame = None
        if len(all_frames) >= 2:
            # 查找腳踝速度最高的幀（代表最大加速階段）
            ankle_speeds = []
            for frame in all_frames:
                _, speed = self.pose_analyzer.get_frame_angleOrSpeed_data(
                    frame, speed_name='ankle_speed'
                )
                ankle_speeds.append((frame, speed))
            
            # 找最大速度的幀
            if ankle_speeds:
                ankle_speeds = [(f, s) for f, s in ankle_speeds 
                               if not (isinstance(s, float) and np.isnan(s)) and s > 0]
                if ankle_speeds:
                    br_frame = max(ankle_speeds, key=lambda x: float(x[1]) if isinstance(x[1], (int, float, np.number)) else 0)[0]
                    self.br_frame_num = br_frame
                    print(f"[KeyFrame] BR detected at frame {br_frame}")
        
        # 顯示檢測到的關鍵幀
        self.display_key_frames()
        
        # 打印檢測結果
        print(f"[KeyFrame] 檢測完成 - FC: {self.fc_frame_num}, MER: {self.mer_frame_num}, BR: {self.br_frame_num}")
    
    def handleVideoEnd(self):
        """Handle the logic when video reaches its end."""
        self.play_times -= 1
        if self.ui.skeletonVideoCheckBox.isChecked():
            # 只在第一次回放時儲存影片
            if not self.video_saved:
                self.video_loader.saveVideo(self.output_dir)
                self.video_saved = True
            # 影片結束後自動檢測關鍵幀
            QTimer.singleShot(500, self.auto_detect_key_frames)

        if self.play_times > 0:
            # Replay the video
            self.playBtnClicked()
            self.ui.frameSlider.setValue(0)
            self.playBtnClicked()
        elif self.play2videos:
            self.ui.skeletonVideoCheckBox.setCheckState(0)
            self.ui.showSkeletonCheckBox.setCheckState(0)
            self.ui.selectCheckBox.setCheckState(2)
            self.ui.cameraCheckBox.setCheckState(0)
            self.play2videos = False
            self.play_times = 2
        else:
            # Stop playback and reset
            self.playBtnClicked()
            self.ui.cameraCheckBox.setCheckState(2)
            self.ui.showSkeletonCheckBox.setCheckState(0)
            self.ui.showSkeletonCheckBox.setCheckState(2)
            self.ui.selectCheckBox.setCheckState(2)
            self.ui.skeletonVideoCheckBox.setCheckState(0)
            self.ui.startPitchCheckBox.setCheckState(2)
            self.play_times = 2
            # print(self.play_times)

    def pitherAnaylze(self):
        """使用 PitchingAnalyzer 分析投球動作"""
        # 確認已選擇人物
        if self.pose_estimater.person_id is None:
            return
        
        # 取得當前骨架關鍵點
        # 從當前檢測結果中取出選中人物的 keypoints（即時模式）
        keypoints = self.pose_estimater.getPersonDf(is_select=True, is_kpt=True)
        if keypoints is None or len(keypoints) == 0:
            return
        
        # 使用 PitchingAnalyzer 處理
        signal = self.pitching_analyzer.process(keypoints)
        # print(f"Detected state: {self.pitching_analyzer.state}, Signal: {signal}")
        
        # 處理信號
        if signal == "START":
            # 從 START 開始錄影，並帶入前60幀預存
            self.startAutoRecording()
        elif signal == "STOP":
            # 收到 STOP 後結束錄影並存檔
            self.stopAutoRecording()
    
    def initRecorderTimer(self, duration:int):
        if self.record_timer is not None and self.record_timer.is_running():
            return
        if self.record_timer is None:
            self.ui.showBboxCheckBox.setCheckState(0)
            self.ui.selectKptCheckBox.setCheckState(0)
            self.ui.selectCheckBox.setCheckState(0)
            self.ui.showSkeletonCheckBox.setCheckState(0)
            self.ui.recordCheckBox.setCheckState(2)
            self.record_timer = Timer(duration)
            self.record_timer.start()

    def updateFrame(self, frame: np.ndarray = None, frame_2: np.ndarray = None, frame_num:int = None):
        """Update the displayed frame with additional analysis."""
        # 更新當前的frame和frame_num
        if self.is_video and frame_num is not None:
            frame,frame_2 = self.video_loader.getVideoImage(frame_num)
            # print(f"frame_num: {frame_num}")
        countdown_time = self.updateTimers() 
        drawed_img = self.image_drawer.drawInfo(frame, frame_num, self.pose_estimater.kpt_buffer, countdown_time)
        drawed_img_2 = self.image_drawer_2.drawInfo(frame_2, frame_num, self.pose_estimater_2.kpt_buffer, countdown_time)
        self.showImage(drawed_img, self.view_scene, self.ui.FrameView)
        self.showImage(drawed_img_2, self.view_scene_2, self.ui.FrameView_2)

        if self.fullscreen_view_1:
            qimg = QImage(drawed_img, drawed_img.shape[1], drawed_img.shape[0], QImage.Format_RGB888).rgbSwapped()
            # self.fullscreen_view_1.resize(800, 600)
            self.fullscreen_view_1.set_frame(qimg)
        if self.fullscreen_view_2:
            qimg2 = QImage(drawed_img_2, drawed_img_2.shape[1], drawed_img_2.shape[0], QImage.Format_RGB888).rgbSwapped()
            # self.fullscreen_view_2.resize(800, 600)
            self.fullscreen_view_2.set_frame(qimg2)
    

    def updateTimers(self):
        countdown_time = None
        if self.record_timer is not None:
            countdown_time = self.record_timer.get_remaining_time()
            if countdown_time == 0:
                self.resetRecordTimer()
        return countdown_time

    def resetRecordTimer(self):
        import gc
        
        self.record_timer = None
        video_path = self.camera.video_path
        video_path_2 = self.camera.video_path_2
        
        self.ui.recordCheckBox.setCheckState(0)
        self.ui.startPitchCheckBox.setCheckState(0)
        self.ui.cameraCheckBox.setCheckState(0)
        
        # 徹底清理舊視頻數據後再加載新視頻
        if hasattr(self, 'video_loader') and self.video_loader is not None:
            self.video_loader.reset()
        gc.collect()  # 強制垃圾回收
        
        self.video_loader.loadVideo(video_path, video_path_2)
        self.checkVideoLoad()
        self.viewer_3d = Triangulate3DViewer(self.K_S, self.K_F, self.F)
        self.viewer_3d.setJsonPaths(self.video_loader.folder_path, self.video_loader.video_name)
        # 初始化 3D 視圖以顯示骨架
        self.updateVideoInfo()

    def checkVideoLoad(self):
        """檢查影片是否讀取完成，並更新 UI 元素。"""
        # 檢查是否有影片名稱，若無則不執行後續操作
        if not self.video_loader.video_name and not self.video_loader.video_name_2:
            return
        # 若影片正在讀取中，定時檢查讀取狀況
        if self.video_loader.is_loading:
            # self.ui.videoNameLabel.setText("讀取影片中")
            QTimer.singleShot(100, self.checkVideoLoad)  # 每100ms 檢查一次
            return
        # 若載入失敗或無有效幀，提示並返回
        if not self.video_loader.video_frames or not self.video_loader.video_frames_2:
            QMessageBox.warning(self, "無法播放影片", "影片載入失敗或沒有有效畫面，請重新選擇影片")
            return
        # 影片讀取完成後更新 UI 元素
        self.updateVideoInfo()
        # self.initAnalyzeFrame()

    def load_video_list(self, root_folder):
        """掃描所有子資料夾並分類加入 Tree"""
        video_exts = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
        self.tree.clear()  # 清空 TreeWidget
        for pitcher_folder in os.listdir(root_folder):
            pitcher_path = os.path.join(root_folder, pitcher_folder)
            if not os.path.isdir(pitcher_path):
                continue

            pitcher_item = QTreeWidgetItem(self.tree)
            pitcher_item.setText(0, pitcher_folder)

            for play_folder in os.listdir(pitcher_path):
                play_path = os.path.join(pitcher_path, play_folder)
                if not os.path.isdir(play_path):
                    continue

                play_item = QTreeWidgetItem(pitcher_item)
                play_item.setText(0, play_folder)

                for file in os.listdir(play_path):
                    if file.lower().endswith(video_exts):
                        video_path = os.path.join(play_path, file)
                        video_item = QTreeWidgetItem(play_item)
                        video_item.setText(0, file)
                        video_item.setData(0, 1, video_path)  # 儲存影片路徑

    def play_video_from_item(self):
        """從 Tree 中選擇兩部影片並播放"""
        import gc
        
        selected_items = [item for item in self.tree.selectedItems()
                        if item.data(0, 1) is not None]  # 只保留影片節點

        # 清除 checkbox 狀態
        self.ui.startPitchCheckBox.setCheckState(0)
        self.ui.showSkeletonCheckBox.setCheckState(0)
        self.ui.cameraCheckBox.setCheckState(0)

        # 多於 2 個影片則只保留前兩個
        if len(selected_items) > 2:
            for item in selected_items[2:]:
                item.setSelected(False)
            selected_items = selected_items[:2]

        # 確認剛好選取 2 個影片節點
        if len(selected_items) == 2:
            self.is_video = True
            video_paths = [os.path.abspath(item.data(0, 1)) for item in selected_items]
            video_path1, video_path2 = video_paths
            self.output_dir = os.path.dirname(video_path1) 
            
            # 清理舊數據
            self.video_loader.reset()
            gc.collect()  # 強制垃圾回收
            
            self.video_loader.loadVideo(video_path1, video_path2)
            self.checkVideoLoad()
            self.play2videos = True

            # 清除選取狀態
            for item in selected_items:
                item.setSelected(False)

    def updateVideoInfo(self):
        """更新與影片相關的資訊顯示在 UI 上。"""
        # 重置分析器和UI显示（video_loader已在play_video_from_item中清理过）
        self.reset()
        
        self.initFrameSlider()
        self.updateFrame(frame_num=0)
        self.model.setImageSize(self.video_loader.video_size)
        video_size = self.video_loader.video_size
        self.ui.ResolutionLabel.setText(f"(0,0) - {video_size[0]} x {video_size[1]}")
        
        # 初始化 3D 視圖（用於後續顯示 3D 骨架）
        self.viewer_3d = Triangulate3DViewer(self.K_S, self.K_F, self.F)
        self.viewer_3d.setJsonPaths(self.video_loader.folder_path, self.video_loader.video_name)
        
        # 在 Vispywidget 中顯示 3D canvas
        if self.ui.Vispywidget.layout() is None:
            self.ui.Vispywidget.setLayout(QVBoxLayout())
        else:
            # 清空舊的 widget
            while self.ui.Vispywidget.layout().count():
                self.ui.Vispywidget.layout().takeAt(0).widget().deleteLater()
        
        # 添加 Triangulate3DViewer 的 canvas
        canvas = self.viewer_3d.get_canvas()
        self.ui.Vispywidget.layout().addWidget(canvas.native)
        
        self.ui.playBtn.click()

    def initAnalyzeFrame(self):
        self.ui.showSkeletonCheckBox.setCheckState(2)
        frame,frame_2 = self.video_loader.getVideoImage(0)
        _, _, _= self.pose_estimater.detectKpt(frame, 0, is_video=True)
        _, _, _= self.pose_estimater_2.detectKpt(frame_2, 0, is_video=True)
        self.ui.selectCheckBox.setCheckState(2)
        self.ui.selectKptCheckBox.setCheckState(0)
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
        self.ui.frameNumLabel.setText(f'0/0')

    def reset(self):
        # 重置所有分析器和绘制器
        self.pose_estimater.reset()
        self.pose_estimater_2.reset()
        self.pose_analyzer.reset()
        self.pose_analyzer_2.reset()
        self.image_drawer.reset()
        self.image_drawer_2.reset()
        
        # 清理UI显示
        self.resetFrameSlider()
        self.view_scene.clear()
        self.view_scene_2.clear()
        
        # 重置投球分析器
        self.pitching_analyzer.reset()
        
        # 清理摄像头和预存帧及 frame_buffer
        self.camera.pre_frames.clear()
        if self.camera.frame_buffer is not None:
            while not self.camera.frame_buffer.empty():
                try:
                    self.camera.frame_buffer.get_nowait()
                except:
                    break
        if self.camera.frame_buffer_2 is not None:
            while not self.camera.frame_buffer_2.empty():
                try:
                    self.camera.frame_buffer_2.get_nowait()
                except:
                    break
        self.is_auto_recording = False
        
        # 清除關鍵幀
        self.fc_frame_num = None
        self.mer_frame_num = None
        self.br_frame_num = None
        self.key_frames_scene_fc.clear()
        self.key_frames_scene_mer.clear()
        self.key_frames_scene_br.clear()

    def showImage(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView):
        """Display an image in the QGraphicsView."""
        scene.clear()
        if image is None:
            print(self.video_loader.total_frames)
            # print(self.video_loader.video_frames)
            exit()
        h, w = image.shape[:2]
        qImg = QImage(image, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
        del image
        pixmap = QPixmap.fromImage(qImg)        
        # del qImg
        scene.addPixmap(pixmap)
        GraphicsView.setScene(scene)
        GraphicsView.setAlignment(Qt.AlignLeft)
        GraphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        """Handle mouse events for person, keypoint selection, and fullscreen view."""
        global_pose = event.globalPos()
        pos = self.ui.FrameView.mapFromGlobal(global_pose)
        clicked_view = None
        if self.ui.FrameView.rect().contains(pos):
            clicked_view = 1
            scene_pos = self.ui.FrameView.mapToScene(self.ui.FrameView.mapFromParent(pos))
        else:
            clicked_view = 2
            scene_pos = self.ui.FrameView_2.mapToScene(self.ui.FrameView_2.mapFromParent(pos))

        x, y = scene_pos.x(), scene_pos.y()
        frame_num = self.ui.frameSlider.value()
        search_person_df = self.pose_estimater.getPersonDf(frame_num) if self.is_video else self.pose_estimater.pre_person_df

    def show_fullscreen(self, view_index):
        if view_index == 1:
            self.fullscreen_view_1 = FullscreenVideoDialog("放大畫面 左", self)
            self.fullscreen_view_1.control_signal.connect(self.handleFullscreenControl)
            self.fullscreen_view_1.set_slider_range(0, self.video_loader.total_frames - 1)
            self.fullscreen_view_1.set_slider_value(self.ui.frameSlider.value())
            self.fullscreen_view_1.slider_signal.connect(self.onFullscreenSliderChanged)
            self.fullscreen_view_1.show()
        elif view_index == 2:
            self.fullscreen_view_2 = FullscreenVideoDialog("放大畫面 右", self)
            self.fullscreen_view_2.control_signal.connect(self.handleFullscreenControl)
            self.fullscreen_view_2.set_slider_range(0, self.video_loader.total_frames - 1)
            self.fullscreen_view_2.set_slider_value(self.ui.frameSlider.value())
            self.fullscreen_view_2.slider_signal.connect(self.onFullscreenSliderChanged)
            self.fullscreen_view_2.show()

    def handleFullscreenControl(self, command):
        """處理全螢幕窗口的控制信號"""
        if command == "prev":
            self.ui.frameSlider.setValue(self.ui.frameSlider.value() - 1)
        elif command == "next":
            self.ui.frameSlider.setValue(self.ui.frameSlider.value() + 1)
        elif command == "pause":
            self.is_play = False
        elif command == "play":
            self.is_play = True

    def onFullscreenSliderChanged(self, value):
        """全螢幕slider移動時，同步主視窗frameSlider"""
        self.ui.frameSlider.setValue(value)

    def syncFullscreenSlider(self, value):
        if self.fullscreen_view_1:
            self.fullscreen_view_1.set_slider_value(value)
        if self.fullscreen_view_2:
            self.fullscreen_view_2.set_slider_value(value)

    def updatePlaybackRate(self):
        """更新播放速率時的處理"""
        if self.is_play and hasattr(self, "_play_timer"):
            self._play_timer.start(self._get_play_interval_ms())
    
    def onVideoWriteFinished(self):
        """視頻寶入完成後自動載入並回放"""
        print("[AutoRecord] Video write finished, loading for playback...")
        self.video_loader.reset()
        self.ui.recordCheckBox.setCheckState(0)
        self.ui.startPitchCheckBox.setCheckState(0)
        self.ui.cameraCheckBox.setCheckState(0)
        self.video_loader.loadVideo(self.video_filename, self.video_filename_2)
        self.checkVideoLoad()
        self.viewer_3d = Triangulate3DViewer(self.K_S, self.K_F, self.F)
        self.viewer_3d.setJsonPaths(self.video_loader.folder_path, self.video_loader.video_name)
    
    def onVideoWriteError(self, error_msg):
        """視頻寶入出錯"""
        print(f"[AutoRecord] Error: {error_msg}")
        QMessageBox.warning(self, "錄影錯誤", f"視頻寶入失敗：{error_msg}")

    def startAutoRecording(self):
        """開始自動錄影：包含前60幀預存，直接獲取原始帧（每一帧）。"""
        if self.is_auto_recording:
            return
        # 建立輸出路徑
        current_date = datetime.now().strftime("%Y%m%d")
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f'../../Db/Record/Pitcher{self.ui.PitcherID.currentText()}_{current_date}/{current_time}'
        os.makedirs(self.output_dir, exist_ok=True)
        self.video_filename = os.path.join(self.output_dir, f'CS_Fps{self.camera_fps}_{current_time}.mp4')
        self.video_filename_2 = os.path.join(self.output_dir, f'CF_Fps{self.camera_fps}_{current_time}.mp4')

        # 適役 Camera 的自動錄影，從 pre_frames 開始
        self.camera.start_auto_recording()
        self.is_auto_recording = True
        
        actual_fps = self.camera.video_thread.camera1.get_fps() if self.camera.video_thread else None
        print(f"[AutoRecord] START, preload {len(self.camera.record_frames)} frames, 相機FPS: {actual_fps}")

    def stopAutoRecording(self):
        """收到 STOP 信號後在後台寫檔並自動回放。"""
        if not self.is_auto_recording:
            return
        self.is_auto_recording = False
        
        # 停止 Camera 的自動錄影，取得所有錄製的幀
        frames_copy, actual_fps = self.camera.stop_auto_recording()
        
        # 若沒有任何幀，直接返回
        if not frames_copy:
            print("[AutoRecord] 無可寫入的幀，略過存檔")
            return
        
        write_fps = round(actual_fps)
        print(f"[AutoRecord] Actual recorded FPS: {actual_fps}, writing at {write_fps} FPS")
        
        # 在後台執行寫檔，不阻塞 UI
        self.video_writer_thread = FramesToVideoWriterThread(
            frames_copy, 
            self.video_filename, 
            self.video_filename_2, 
            fps=write_fps
        )
        self.video_writer_thread.finished.connect(self.onVideoWriteFinished)
        self.video_writer_thread.error.connect(self.onVideoWriteError)
        self.video_writer_thread.start()
        
        print(f"[AutoRecord] STOP, writing {len(frames_copy)} frames in background...")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PosePitchTabControl()
    window.show()
    sys.exit(app.exec_())
