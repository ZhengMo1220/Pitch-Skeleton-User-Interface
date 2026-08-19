import os
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(os.environ["CONDA_PREFIX"], "Library", "plugins", "platforms")
import vispy
vispy.use('pyqt5')
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QColor, QImage, QPixmap, QPainter, QFont
from PyQt5.QtCore import Qt, QTimer, QRect
import numpy as np
import sys
import cv2
from video_ui import Ui_video_widget
# from video_ui_graph import Ui_video_widget
from utils.vis_image import ImageDrawer
from utils.selector import PersonSelector, KptSelector
from cv_utils.cv_control import VideoLoader, JsonLoader
from skeleton.detect_skeleton import PoseEstimater
from utils.vis_graph_3d import GraphPlotter
from utils.analyze_3d import PoseAnalyzer
from utils.model import Model
import pyqtgraph as pg
from triangulate_3d_viewer import Triangulate3DViewer
import time

class SliderColorOverlay(QWidget):
    def __init__(self, slider: QSlider, segments: list):
        """
        slider: 你要疊的 QSlider
        segments: list of tuples (start_frame, end_frame, color)
        """
        super().__init__(slider.parent())
        self.slider = slider
        self.segments = segments
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # 透明且不阻擋滑鼠
        self.setGeometry(self.slider.geometry())
        self.show()

    def updateSegments(self, segments):
        self.segments = segments
        self.update()

    def resizeEvent(self, event):
        """當覆蓋層本身被 resize 時，確保尺寸與 Slider 相同"""
        self.setGeometry(self.slider.geometry())
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self.setGeometry(self.slider.geometry())
        groove_rect = self.slider.rect().adjusted(2, self.slider.height()//2 - 4, -2, -(self.slider.height()//2 - 4))
        total = self.slider.maximum() - self.slider.minimum()

        # 計算上下可用空間高度
        top_area_height = groove_rect.top()
        bottom_area_height = self.height() - groove_rect.bottom()

        # 設置文字字體
        font = QFont("Arial", 7)
        painter.setFont(font)

        for i, (start, end, color) in enumerate(self.segments):
            # 1. 檢查是否為 None
            if end is None or total is None:
                return  # 如果資料還沒準備好，就不要畫這個部分，直接結束繪圖
            # 2. 檢查分母是否為 0 (避免 ZeroDivisionError)
            if total == 0:
                return 
            # 3. 確保 start 也不為 None (如果有用到的話)
            start_val = start if start is not None else 0
            duration = end - start_val
            x1 = groove_rect.left() + (start_val / total) * groove_rect.width()
            x2 = groove_rect.left() + (end / total) * groove_rect.width()
            segment_width = int(x2 - x1)
            painter.fillRect(QRect(int(x1), groove_rect.top(), segment_width, groove_rect.height()), color)
            painter.setPen(color)
            
            # 計算百分比
            percentage = (duration / total) * 100
            
            # 格式化標籤：Frame 數 / 百分比
            label = f"{duration*0.017:.2f}s ({percentage:.1f}%)"
            
            # 定義文字繪圖區域
            # 讓文字繪製在軌道上方，並位於該區塊的中心
            metrics = painter.fontMetrics()
            text_width_needed = metrics.horizontalAdvance(label) + 10 # 4px padding
            text_x = int(x1 + segment_width / 2 - text_width_needed / 2)
            # 決定 Y 層次： i 為偶數畫上方，i 為奇數畫下方
            if i % 2 == 0:
                # 繪製在上方區域 (Y 從 0 開始)
                text_y_start = 0
                text_height = top_area_height
                
            else:
                # 繪製在下方區域 (Y 從軌道底部開始)
                text_y_start = groove_rect.bottom()
                text_height = bottom_area_height
            
            # 文字矩形：使用整個區塊寬度，但高度只佔上半部分
            text_rect = QRect(text_x, text_y_start, text_width_needed, text_height)

            # 確保文字區塊足夠大，避免繪製過多小區塊的標籤
            painter.drawText(text_rect, Qt.AlignCenter, label)

class PoseVideoTabControl(QWidget):
    def __init__(self, model:Model, parent = None):
        super(PoseVideoTabControl, self).__init__(parent)
        self.ui = Ui_video_widget()
        self.ui.setupUi(self)
        self.model = model
        self.initVar()
        self.setupComponents()
        self.bindUI()
        
    def initVar(self):
        self.is_play = False
        self.is_video = False
        self.view_scene = QGraphicsScene()
        self.curve_scene = QGraphicsScene()
        self.view_scene.clear()
        self.curve_scene.clear()
        self.correct_kpt_idx = 0
        self.is_processed = False
        self.overlay = None  # 初始化 SliderColorOverlay

        # self.table = self.ui.tableWidget
        self.json_path = None
        self.json_path2 = None
        self.viewer3d = None
        self.foot_contact_frame = None
        self.max_shoulder_angle_frame = None
        self.wrist_speed_frame = None
        self.speed_angle_names = ["ankle_speed", "left_knee_3d", "shoulder_hiple_angle", "shoulder_angle", "wrist_speed"]
        self.title_names = ["腳踝速度", "膝蓋角度", "肩峰連線與髖關節連線角度", "肩膀外旋角度", "手腕速度"]

        # self.K_F = np.array([
        #     [11229.920949545698, 0.0, 937.199020465423],
        #     [0.0, 11240.535783693984, 586.8641683267341],
        #     [0.0, 0.0, 1.0]
        # ])
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

        # self.F = np.array([
        #     [1.20973516e-07, -5.15926324e-06,  2.69228899e-03],
        #     [-2.18398570e-06, -7.16241028e-07,  8.11364487e-03],
        #     [5.00374629e-04, -4.56687478e-03,  1.00000000e+00]
        # ])

        self.F = np.array([
            [-1.23527137e-06,  3.74634658e-05, -1.36469340e-02],
            [1.59530549e-05,  9.03916270e-07, -6.06394789e-02],
            [-4.66549815e-03,  3.39782280e-02,  1.00000000e+00]
        ])
        
        pg.setConfigOptions(foreground=QColor(113,148,116), antialias = True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

    def resizeEvent(self, event):
        new_size = event.size()
        # 在此執行你想要的操作
        if self.video_loader.video_name is not None:
            self.updateFrame(self.ui.frameSlider.value())
        super().resizeEvent(event)  

    def initFrameSlider(self):
        """初始化影片滑桿和相關的標籤。"""
        total_frames = self.video_loader.total_frames
        self.ui.frameSlider.setMinimum(0)
        self.ui.frameSlider.setMaximum(total_frames - 1)
        self.ui.frameSlider.setValue(0)
        self.ui.frameNumLabel.setText(f'0/{total_frames - 1}')

    def initGraph(self):
        """初始化圖表和模型設定。"""
        total_frames = self.video_loader.total_frames
        self.graph_plotter._init_graph(total_frames) 
        self.toggleShowGraph("ankle_speed", self.ui.ankleGraphCheckbox.checkState())
        self.toggleShowGraph("left_knee_3d", self.ui.kneeGraphCheckbox.checkState())
        self.toggleShowGraph("shoulder_hiple_angle", self.ui.shouhipGraphCheckbox.checkState())
        self.toggleShowGraph("shoulder_angle", self.ui.shoulderGraphCheckbox.checkState())
        self.toggleShowGraph("wrist_speed", self.ui.wristGraphCheckbox.checkState())
        self.showGraph(self.ui.graph_widget)

    def resetPhaseDetection(self):
        self.ui.footContact_ptn.setDisabled(True)
        self.ui.maxER_ptn.setDisabled(True)
        self.ui.release_ptn.setDisabled(True)
        self.foot_contact_frame = None
        self.max_shoulder_angle_frame = None
        self.wrist_speed_frame = None
        self.ui.footContact_label.setText("")
        self.ui.maxER_label.setText("")
        self.ui.release_label.setText("")

    def reset3DInfo(self):
        self.ui.ankle_label.setText("(m/s)")
        self.ui.knee_label.setText("°")
        self.ui.shouhip_label.setText("°")
        self.ui.shoulder_label.setText("°")
        self.ui.wrist_label.setText("(m/s)")
        self.ui.ankleGraphCheckbox.setChecked(False)
        self.ui.kneeGraphCheckbox.setChecked(False)
        self.ui.shouhipGraphCheckbox.setChecked(False)
        self.ui.shoulderGraphCheckbox.setChecked(False)
        self.ui.wristGraphCheckbox.setChecked(False)

    def bindUI(self):
        self.ui.footContact_ptn.setDisabled(True)
        self.ui.maxER_ptn.setDisabled(True)
        self.ui.release_ptn.setDisabled(True)

        self.ui.footContact_ptn.clicked.connect(
            lambda: self.ui.frameSlider.setValue(self.foot_contact_frame)
        )
        self.ui.maxER_ptn.clicked.connect(
            lambda: self.ui.frameSlider.setValue(self.max_shoulder_angle_frame + 1)
        )  
        self.ui.release_ptn.clicked.connect(
            lambda: self.ui.frameSlider.setValue(self.wrist_speed_frame + 1)
        )

        self.ui.loadOriginalVideoBtn.clicked.connect(
            lambda: self.loadVideo(is_processed=False))
        self.ui.loadProcessedVideoBtn.clicked.connect(
            lambda: self.loadVideo(is_processed=True))
        
        self.ui.playBtn.clicked.connect(self.playBtnClicked)
        self.ui.backKeyBtn.clicked.connect(
            lambda: self.ui.frameSlider.setValue(self.ui.frameSlider.value() - 1)
        )
        self.ui.forwardKeyBtn.clicked.connect(
            lambda: self.ui.frameSlider.setValue(self.ui.frameSlider.value() + 1)
        )
        self.ui.frameSlider.valueChanged.connect(self.analyzeFrame)
        # self.kpt_table = KeypointTable(self.ui.KptTable,self.pose_estimater)
        # self.ui.KptTable.cellActivated.connect(self.kpt_table.onCellClicked)
        self.ui.FrameView.mousePressEvent = self.mousePressEvent
        # self.ui.IdCorrectBtn.clicked.connect(self.correctId)
        self.ui.startCodeBtn.clicked.connect(self.toggleDetect)
        # self.ui.selectCheckBox.stateChanged.connect(self.toggleSelect)
        self.ui.showSkeletonCheckBox.stateChanged.connect(self.toggleShowSkeleton)
        # self.ui.selectKptCheckBox.stateChanged.connect(self.toggleKptSelect)
        self.ui.showBboxCheckBox.stateChanged.connect(self.toggleShowBbox)
        # self.ui.showAngleCheckBox.stateChanged.connect(self.toggleShowAngleInfo)
        self.ui.ankleGraphCheckbox.stateChanged.connect(lambda state: self.toggleShowGraph("ankle_speed", state))
        self.ui.kneeGraphCheckbox.stateChanged.connect(lambda state: self.toggleShowGraph("left_knee_3d", state))
        self.ui.shouhipGraphCheckbox.stateChanged.connect(lambda state: self.toggleShowGraph("shoulder_hiple_angle", state))
        self.ui.shoulderGraphCheckbox.stateChanged.connect(lambda state: self.toggleShowGraph("shoulder_angle", state))
        self.ui.wristGraphCheckbox.stateChanged.connect(lambda state: self.toggleShowGraph("wrist_speed", state))

        # self.ui.zoomInButton.toggled.connect(self.toggleZoomIn)

        if self.ui.vispy_widget.layout() is None:
            self.ui.vispy_widget.setLayout(QVBoxLayout())

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

    def mousePressEvent(self, event):
        view_rect = self.ui.FrameView.rect()
        pos = event.pos()

        if not view_rect.contains(pos):
            return
        if self.video_loader.video_name is not None:
            self.updateFrame(frame_num=self.ui.frameSlider.value())

    def keyPressEvent(self, event):
        if event.key() == ord('D') or event.key() == ord('d'):
            self.ui.frameSlider.setValue(self.ui.frameSlider.value() + 1)
        elif event.key() == ord('A') or event.key() == ord('a'):
            self.ui.frameSlider.setValue(self.ui.frameSlider.value() - 1)
        else:
            super().keyPressEvent(event)

    def setupComponents(self): 
        self.person_selector = PersonSelector()
        self.kpt_selector = KptSelector()
        self.pose_estimater = PoseEstimater(self.model)
        self.pose_estimater_2 = PoseEstimater(self.model)
        self.pose_analyzer = PoseAnalyzer(self.pose_estimater, self.K_S, self.K_F, self.F)
        self.pose_analyzer_2 = PoseAnalyzer(self.pose_estimater_2, self.K_S, self.K_F, self.F)
        self.graph_plotter = GraphPlotter(self.pose_analyzer, speed_name=self.speed_angle_names, title_names=self.title_names)
        self.image_drawer = ImageDrawer(self.pose_estimater, self.pose_analyzer)
        self.image_drawer_2 = ImageDrawer(self.pose_estimater_2, self.pose_analyzer_2)
        self.video_loader = VideoLoader(self.image_drawer, self.image_drawer_2)

    def reset(self):
        self.resetPhaseDetection()
        self.reset3DInfo()
        self.viewer3d.reset()
        self.person_selector.reset()
        self.kpt_selector.reset()
        self.pose_estimater.reset()
        self.pose_analyzer.reset()
        self.graph_plotter.reset()
        self.image_drawer.reset()
        self.pose_estimater_2.reset()
        self.pose_analyzer_2.reset()
        self.image_drawer_2.reset()
        self.view_scene.clear()
        self.curve_scene.clear()
        # 清除舊的 SliderColorOverlay
        if self.overlay is not None:
            self.overlay.setParent(None)
            self.overlay = None
   
    def loadVideo(self, is_processed:bool = False, video_path:str = None, video_path_2:str = None):
        if self.is_play:
            self.ui.playBtn.click()
        self.is_processed = is_processed
        self.video_loader.loadVideo(video_path, video_path_2)
        self.checkVideoLoad()
        self.ui.showSkeletonCheckBox.setChecked(False)

        layout = self.ui.vispy_widget.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        self.viewer3d = Triangulate3DViewer(self.K_S, self.K_F, self.F)
        self.viewer3d.setJsonPaths(self.video_loader.folder_path, self.video_loader.video_name)
        layout.addWidget(self.viewer3d.get_canvas().native)

    def loadProcessedData(self):
        json_loader = JsonLoader(self.video_loader.folder_path, self.video_loader.video_name)
        json_loader_2 = JsonLoader(self.video_loader.folder_path, self.video_loader.video_name_2)
        self.json_path, person_df = json_loader.load()
        self.json_path2, person_df_2 = json_loader_2.load()
        self.pose_estimater.setProcessedData(person_df)
        self.pose_estimater_2.setProcessedData(person_df_2)

    def checkVideoLoad(self):
        """檢查影片是否讀取完成，並更新 UI 元素。"""
        # 檢查是否有影片名稱，若無則不執行後續操作
        if not self.video_loader.video_name:
            return
        # 若影片正在讀取中，定時檢查讀取狀況
        if self.video_loader.is_loading:
            self.ui.videoNameLabel.setText("讀取影片中")
            QTimer.singleShot(100, self.checkVideoLoad)  # 每100ms 檢查一次
            return
        # 影片讀取完成後更新 UI 元素
        self.updateVideoInfo()

    def updateVideoInfo(self):
        """更新與影片相關的資訊顯示在 UI 上。"""
        self.reset()
        self.initFrameSlider()
        self.initGraph()
        self.updateFrame(0)
        self.model.setImageSize(self.video_loader.video_size)
        self.ui.videoNameLabel.setText(self.video_loader.video_name)
        self.ui.videoNameLabel_3.setText(self.video_loader.video_name_2)
        video_size = self.video_loader.video_size
        self.ui.ResolutionLabel.setText(f"(0,0) - {video_size[0]} x {video_size[1]}")
        if self.is_processed:
            self.loadProcessedData()
        else:
            self.ui.startCodeBtn.setEnabled(True)

    def showImage(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView): 
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

    def showGraph(self, container_widget: QWidget):
        layout = container_widget.layout()
        
        # 1. 如果沒有 Layout，就建立一個新的
        if layout is None:
            layout = QVBoxLayout(container_widget)
        
        # 2. 【關鍵】無論是新建立的還是原本就有的，都要強制去除邊距
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 3. 如果圖表還沒被加入，才加入 (避免重複加入報錯)
        # 檢查 layout 中是否已經包含 graph widget
        if layout.indexOf(self.graph_plotter.graph) == -1:
            layout.addWidget(self.graph_plotter.graph)

    def playFrame(self, start_num:int=0):
        for i in range(start_num, self.video_loader.total_frames):
            if not self.is_play:
                break
            self.ui.frameSlider.setValue(i)
            if i == self.video_loader.total_frames - 1 and self.is_play:
                self.playBtnClicked()
            cv2.waitKey(15)

    def phaseDetection(self, frame_num:int):
        # set text labels
        _, ankle_speed_value = self.pose_analyzer.get_frame_angleOrSpeed_data(frame_num, speed_name="ankle_speed")
        _, knee_angle_value = self.pose_analyzer.get_frame_angleOrSpeed_data(frame_num, speed_name="left_knee_3d")
        _, shouhip_angle_value = self.pose_analyzer.get_frame_angleOrSpeed_data(frame_num, speed_name="shoulder_hiple_angle")
        _, shoulder_angle_value = self.pose_analyzer.get_frame_angleOrSpeed_data(frame_num, speed_name="shoulder_angle")
        _, wrist_speed_value = self.pose_analyzer.get_frame_angleOrSpeed_data(frame_num, speed_name="wrist_speed")

        self.ui.ankle_label.setText(f'{ankle_speed_value:.2f}(m/s)')
        self.ui.knee_label.setText(f'{knee_angle_value}°')
        self.ui.shouhip_label.setText(f'{shouhip_angle_value}°')
        self.ui.shoulder_label.setText(f'{shoulder_angle_value}°')
        self.ui.wrist_label.setText(f'{wrist_speed_value:.2f}(m/s)')

        if self.foot_contact_frame is not None and self.max_shoulder_angle_frame is not None and self.wrist_speed_frame is not None:
            self.graph_plotter.draw_vertical_lines(
                    x1=self.foot_contact_frame,x2=self.max_shoulder_angle_frame+1,x3=self.wrist_speed_frame+1, width=1)

        if frame_num == self.video_loader.total_frames - 1 and self.ui.footContact_ptn.isEnabled() == False:
            _, foot_contact = self.pose_analyzer.get_frame_angleOrSpeed_data(speed_name="ankle_speed")
            # _, foot_contact = self.pose_analyzer.get_frame_angleOrSpeed_data(speed_name="shoulder_hiple_angle")
            _, shoulder_angle = self.pose_analyzer.get_frame_angleOrSpeed_data(speed_name="shoulder_angle")
            _, wrist_speed = self.pose_analyzer.get_frame_angleOrSpeed_data(speed_name="wrist_speed")

            max_shoulder_angle_frame = np.argmax(shoulder_angle)

            foot_contact_before_max_er = foot_contact[:max_shoulder_angle_frame + 1]
            foot_contact_frame = self.pose_analyzer.find_first_matching_frame(
                frames=list(range(1, len(foot_contact_before_max_er) + 1)),
                ankle_speed_max=0.5,
                knee_angle_min=100,
                knee_angle_max=140,
                sh_angle_min=10
            )
            # foot_contact_frame = np.argmax(foot_contact_before_max_er)

            wrist_speed_after_max_er = wrist_speed[max_shoulder_angle_frame + 1:]
            wrist_speed_frame = np.argmax(wrist_speed_after_max_er) + max_shoulder_angle_frame + 1

            self.ui.footContact_label.setText(f"frame {foot_contact_frame}")
            self.ui.maxER_label.setText(f"frame {max_shoulder_angle_frame+1}")
            self.ui.release_label.setText(f"frame {wrist_speed_frame+1}")
            self.ui.footContact_ptn.setDisabled(False)
            self.ui.maxER_ptn.setDisabled(False)
            self.ui.release_ptn.setDisabled(False)
            self.foot_contact_frame = foot_contact_frame
            self.max_shoulder_angle_frame = max_shoulder_angle_frame
            self.wrist_speed_frame = wrist_speed_frame

            segments = [
                (0, foot_contact_frame, QColor("#1C7CDB")),
                (foot_contact_frame, max_shoulder_angle_frame+1, QColor("#C90E0E")),
                (max_shoulder_angle_frame+1, wrist_speed_frame+1, QColor("#F2A900")),
                (wrist_speed_frame+1, self.video_loader.total_frames, QColor("#00AC2D"))
            ]

            # 移除舊的 overlay 再建立新的
            if self.overlay is not None:
                self.overlay.setParent(None)
            self.overlay = SliderColorOverlay(self.ui.frameSlider, segments)
            self.graph_plotter.draw_vertical_lines(
                    x1=foot_contact_frame,x2=max_shoulder_angle_frame+1,x3=wrist_speed_frame+1, width=1)
            # 傳入 plotter
            self.graph_plotter.set_motion_phases(segments)
            # 接著呼叫 updateGraph 就會自動渲染背景顏色
            self.graph_plotter.updateGraph(frame_num)

    def analyzeFrame(self):
        fps = 0
        frame_num = self.ui.frameSlider.value()
        self.ui.frameNumLabel.setText(f'{frame_num}/{len(self.video_loader.video_frames) - 1}')
        frame, frame_2 = self.video_loader.getVideoImage(frame_num)
        t0 = time.time()
        _, _, fps= self.pose_estimater.detectKpt(frame, frame_num, is_video=True, is_processed=self.is_processed)
        t1 = time.time()
        _, _, fps= self.pose_estimater_2.detectKpt(frame_2, frame_num, is_video=True, is_processed=self.is_processed)
        t2 = time.time()
        # self.ui.FPSInfoLabel.setText(f"{fps:02d}")
        t3 = time.time()
        if self.viewer3d:
        # 取得2D骨架資料
            person_df_L = self.pose_estimater.getPersonDf(frame_num=frame_num)
            person_df_R = self.pose_estimater_2.getPersonDf(frame_num=frame_num)
            if not person_df_L.empty and not person_df_R.empty:
                # 取第一個人的 keypoints
                frame_L = {"keypoints": person_df_L.iloc[0]["keypoints"]}
                frame_R = {"keypoints": person_df_R.iloc[0]["keypoints"]}
                self.viewer3d.add_2d_keypoints(frame_L, frame_R, frame_num=frame_num)
                # pairs = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (18, 19)]
                # max_3d_idx = max(self.viewer3d.all_3d_frames.keys())
                # for i, (idx1, idx2) in enumerate(pairs):
                #     if 0 <= frame_num <= max_3d_idx:
                #         dist = self.viewer3d.get_distance_between(idx1, idx2, frame_num-1)
                #         self.table.setItem(i, 3, QTableWidgetItem(f"{dist*1992.586:.2f}"))
                #     else:
                #         self.table.setItem(i, 3, QTableWidgetItem("N/A"))
                self.pose_analyzer.addAnalyzeInfo(frame_num, frame_L, frame_R)
                self.graph_plotter.updateGraph(frame_num)
                self.phaseDetection(frame_num)
        t4 = time.time()

        if frame_num == self.video_loader.total_frames - 1 and not self.is_processed and self.ui.showSkeletonCheckBox.isChecked():
            self.video_loader.saveVideo(self.video_loader.folder_path)
        self.updateFrame(frame_num)

    def updateFrame(self, frame_num:int):
        image, image_2 = self.video_loader.getVideoImage(frame_num)
        drawed_img = self.image_drawer.drawInfo(image, frame_num, self.pose_estimater.kpt_buffer)
        drawed_img_2 = self.image_drawer_2.drawInfo(image_2, frame_num, self.pose_estimater_2.kpt_buffer)
        self.showImage(drawed_img, self.view_scene, self.ui.FrameView)
        self.showImage(drawed_img_2, self.curve_scene, self.ui.FrameView_2)

    def toggleDetect(self):
        self.ui.showSkeletonCheckBox.setChecked(True)
        frame, frame_2 = self.video_loader.getVideoImage(0)
        _, _, _= self.pose_estimater.detectKpt(frame, 0, is_video=True)
        _, _, _= self.pose_estimater_2.detectKpt(frame_2, 0, is_video=True)
        self.ui.playBtn.click()

    def toggleSelect(self, state:int):
        if not self.ui.showSkeletonCheckBox.isChecked():
            # self.ui.selectCheckBox.setCheckState(0)
            QMessageBox.warning(self, "無法選擇人", "請選擇顯示人體骨架!")
            return
        if state == 2: 
            self.person_selector.select(search_person_df=self.pose_estimater.getPersonDf(frame_num=self.ui.frameSlider.value()))
            self.pose_estimater.setPersonId(self.person_selector.selected_id)
        else:
            self.pose_estimater.setPersonId(None)
        self.updateFrame(self.ui.frameSlider.value())

    # def toggleKptSelect(self, state:int):
    #     """Toggle keypoint selection and trajectory visualization."""
    #     if not self.ui.selectCheckBox.isChecked():
    #         self.ui.selectKptCheckBox.setCheckState(0)
    #         QMessageBox.warning(self, "無法選擇關節點", "請選擇人!")
    #         return
    #     if state == 2:  
    #         self.pose_estimater.setKptId(10)
    #         self.image_drawer.setShowTraj(True)
    #     else:
    #         self.pose_estimater.setKptId(None)
    #         self.image_drawer.setShowTraj(False)
    #     self.updateFrame(self.ui.frameSlider.value())

    def toggleShowSkeleton(self, state:int):
        is_checked = state == 2
        self.pose_estimater.setDetect(is_checked)
        self.image_drawer.setShowSkeleton(is_checked)
        self.pose_estimater_2.setDetect(is_checked)
        self.image_drawer_2.setShowSkeleton(is_checked)
        self.updateFrame(self.ui.frameSlider.value())

    def toggleShowBbox(self, state:int):
        if state == 2:  
            self.image_drawer.setShowBbox(True)
            self.image_drawer_2.setShowBbox(True)
            self.updateFrame(self.ui.frameSlider.value())
        if state == 0:
            self.image_drawer.setShowBbox(False)
            self.image_drawer_2.setShowBbox(False)
            self.updateFrame(self.ui.frameSlider.value())

    # def toggleShowAngleInfo(self, state:int):
    #     if not self.ui.selectCheckBox.isChecked():
    #         self.ui.showAngleCheckBox.setCheckState(0)
    #         QMessageBox.warning(self, "無法顯示關節點角度資訊", "請選擇人!")
    #         return
    #     if state == 2:  
    #         self.image_drawer.setShowAngleInfo(True)
    #     else:
    #         self.image_drawer.setShowAngleInfo(False)

    def toggleShowGraph(self, name, state:int):
        enabled = state == 2
        self.graph_plotter.setPlotVisibility(name, enabled)
        self.graph_plotter.updateGraph(self.ui.frameSlider.value())

    def toggleZoomIn(self, checked:bool):
        """
        當 zoomInButton 被選中時放大，取消選中時顯示原本的範圍。
        
        checked: True 時放大到關鍵動作幀段，False 時顯示全景
        """
        if not hasattr(self.graph_plotter, 'plots') or not self.graph_plotter.plots:
            return
        
        total_frames = self.video_loader.total_frames if self.video_loader.video_name else 100
        
        if checked:
            # 放大模式：顯示從腳踝接觸到釋放的關鍵動作段
            x_min = self.max_shoulder_angle_frame - 10 if self.max_shoulder_angle_frame else 0
            x_max = self.max_shoulder_angle_frame + 10 if self.max_shoulder_angle_frame else total_frames
            self.graph_plotter.zoomIn(x_min, x_max)
        else:
            # 原景模式：顯示全部幀
            self.graph_plotter.zoomIn(0, total_frames)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    model = Model()
    window = PoseVideoTabControl(model)
    window.show()
    sys.exit(app.exec_())
