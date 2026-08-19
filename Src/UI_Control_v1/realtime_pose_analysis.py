import sys
import os

# 解決 OpenMP 重複初始化問題
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import json
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, QTableWidget,
                             QTableWidgetItem, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QColor, QFont
from vispy import scene
import pyqtgraph as pg


class RealtimePhaseDetector:
    """
    實時運動階段檢測器 - 基於實時數據而不是未來數據
    4個階段：
    1. Windup Phase (準備期): 開始 ~ 足踝接觸
    2. Cocking Phase (蓄能期): 足踝接觸 ~ 最大外旋
    3. Acceleration Phase (加速期): 最大外旋 ~ 釋放點
    4. Follow-through Phase (跟進期): 釋放點 ~ 結束
    """
    
    def __init__(self, ankle_speed_threshold=0.5, knee_angle_range=(100, 140), 
                 shoulder_angle_threshold=10, wrist_speed_threshold=0.2):
        """
        ankle_speed_threshold: 腳踝速度閾值 (m/s)
        knee_angle_range: 膝蓋角度範圍 (度)
        shoulder_angle_threshold: 肩膀外旋最小角度 (度)
        wrist_speed_threshold: 手腕速度閾值 (m/s)
        """
        self.ankle_speed_threshold = ankle_speed_threshold
        self.knee_angle_range = knee_angle_range
        self.shoulder_angle_threshold = shoulder_angle_threshold
        self.wrist_speed_threshold = wrist_speed_threshold
        
        # 存儲檢測結果
        self.foot_contact_frame = None
        self.max_shoulder_angle_frame = None
        self.max_shoulder_angle = 0
        self.wrist_speed_frame = None
        
        # 歷史數據
        self.ankle_speed_history = []
        self.knee_angle_history = []
        self.shoulder_angle_history = []
        self.wrist_speed_history = []

    def add_frame_data(self, frame_num, ankle_speed, knee_angle, shoulder_angle, wrist_speed):
        """
        實時添加每幀的數據並進行檢測
        """
        self.ankle_speed_history.append((frame_num, ankle_speed))
        self.knee_angle_history.append((frame_num, knee_angle))
        self.shoulder_angle_history.append((frame_num, shoulder_angle))
        self.wrist_speed_history.append((frame_num, wrist_speed))
        
        # 更新最大外旋角度和對應幀
        if shoulder_angle > self.max_shoulder_angle:
            self.max_shoulder_angle = shoulder_angle
            self.max_shoulder_angle_frame = frame_num
        
        # 檢測足踝接觸 (只用過去的數據)
        if self.foot_contact_frame is None:
            self._detect_foot_contact(frame_num)
        
        # 檢測釋放點 (必須在檢測到最大外旋之後)
        if self.max_shoulder_angle_frame is not None and self.wrist_speed_frame is None:
            if frame_num > self.max_shoulder_angle_frame:
                self._detect_release_point(frame_num)

    def _detect_foot_contact(self, frame_num):
        """
        檢測足踝接觸：
        - 腳踝速度 < 閾值
        - 膝蓋角度在範圍內
        - 肩膀外旋 > 閾值
        """
        if len(self.ankle_speed_history) < 1:
            return
        
        _, ankle_speed = self.ankle_speed_history[-1]
        _, knee_angle = self.knee_angle_history[-1]
        _, shoulder_angle = self.shoulder_angle_history[-1]
        
        # 需要至少有一定數量的幀進行穩定判斷
        if len(self.ankle_speed_history) >= 5:
            # 檢查過去5幀的平均速度
            recent_ankle_speeds = [v for _, v in self.ankle_speed_history[-5:]]
            avg_ankle_speed = np.mean(recent_ankle_speeds)
            
            if (avg_ankle_speed < self.ankle_speed_threshold and
                self.knee_angle_range[0] <= knee_angle <= self.knee_angle_range[1] and
                shoulder_angle >= self.shoulder_angle_threshold):
                self.foot_contact_frame = frame_num

    def _detect_release_point(self, frame_num):
        """
        檢測釋放點：最大外旋後，手腕速度達到峰值
        """
        if len(self.wrist_speed_history) < 1:
            return
        
        _, wrist_speed = self.wrist_speed_history[-1]
        
        # 需要至少5幀用於檢測峰值
        if len(self.wrist_speed_history) >= 5:
            # 檢查是否是局部最大值
            recent_wrist_speeds = [v for _, v in self.wrist_speed_history[-5:]]
            
            # 如果當前值是過去5幀中最高的，認為是釋放點
            if wrist_speed == max(recent_wrist_speeds) and wrist_speed > self.wrist_speed_threshold:
                self.wrist_speed_frame = frame_num

    def get_phases(self):
        """
        返回4個運動階段的幀範圍
        """
        phases = {
            "Windup": (0, self.foot_contact_frame) if self.foot_contact_frame else (0, 1),
            "Cocking": (self.foot_contact_frame, self.max_shoulder_angle_frame) if self.foot_contact_frame and self.max_shoulder_angle_frame else (1, 2),
            "Acceleration": (self.max_shoulder_angle_frame, self.wrist_speed_frame) if self.max_shoulder_angle_frame and self.wrist_speed_frame else (2, 3),
            "Follow-through": (self.wrist_speed_frame, None) if self.wrist_speed_frame else (3, None)
        }
        return phases

    def is_detection_complete(self):
        """檢查是否完成了所有關鍵點的檢測"""
        return (self.foot_contact_frame is not None and 
                self.max_shoulder_angle_frame is not None and 
                self.wrist_speed_frame is not None)


class SliderPhaseOverlay(QWidget):
    """
    在 Slider 上繪製運動階段的顏色覆蓋層
    """
    def __init__(self, slider: QSlider):
        super().__init__(slider.parent())
        self.slider = slider
        self.phases = {}  # {"Windup": (0, 100), "Cocking": (100, 200), ...}
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setGeometry(self.slider.geometry())
        self.show()

    def update_phases(self, phases, total_frames):
        """
        更新階段信息
        phases: dict {phase_name: (start_frame, end_frame)}
        """
        self.phases = {}
        colors = {
            "Windup": QColor("#1C7CDB"),  # 藍色
            "Cocking": QColor("#C90E0E"),  # 紅色
            "Acceleration": QColor("#F2A900"),  # 橙色
            "Follow-through": QColor("#00AC2D")  # 綠色
        }
        
        for phase_name, (start, end) in phases.items():
            if start is not None and end is not None:
                self.phases[phase_name] = (start, end, colors.get(phase_name, QColor("gray")))

    def resizeEvent(self, event):
        self.setGeometry(self.slider.geometry())
        super().resizeEvent(event)

    def paintEvent(self, event):
        if not self.phases:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self.setGeometry(self.slider.geometry())
        
        groove_rect = self.slider.rect().adjusted(2, self.slider.height()//2 - 4, -2, -(self.slider.height()//2 - 4))
        total = self.slider.maximum() - self.slider.minimum()
        
        if total == 0:
            return
        
        font = QFont("Arial", 8)
        painter.setFont(font)
        
        for phase_name, (start, end, color) in self.phases.items():
            if start is None or end is None:
                continue
            
            x1 = groove_rect.left() + (start / total) * groove_rect.width()
            x2 = groove_rect.left() + (end / total) * groove_rect.width()
            segment_width = int(x2 - x1)
            
            painter.fillRect(QRect(int(x1), groove_rect.top(), segment_width, groove_rect.height()), color)
            painter.setPen(color)
            
            # 繪製階段名稱
            text_rect = QRect(int(x1), groove_rect.top() - 20, segment_width, 20)
            painter.drawText(text_rect, Qt.AlignCenter, phase_name)


from PyQt5.QtGui import QPainter

class RealtimePoseAnalysisWidget(QMainWindow):
    """
    實時姿態分析窗口 - 模擬自動錄影回放時的分析
    不需要等待整個視頻完成就能檢測關鍵幀
    """
    
    def __init__(self, json_file=None):
        super().__init__()
        self.setWindowTitle("實時姿態分析 - 運動階段分割")
        self.resize(1400, 900)
        
        # 載入數據
        self.frames_data = []
        if json_file:
            self.load_data(json_file)
        
        # 創建檢測器
        self.detector = RealtimePhaseDetector()
        
        # UI 狀態
        self.current_frame_idx = 0
        self.is_playing = False
        self.analysis_complete = False
        
        # 初始化 UI
        self.init_ui()
        
        # 計算分析數據
        if self.frames_data:
            self.compute_analysis()
        
        # 播放定時器
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next_frame)
        self.timer.setInterval(50)  # 20 FPS

    def load_data(self, json_file):
        """載入 3D 關鍵點數據"""
        json_path = Path(json_file)
        if not json_path.exists():
            raise FileNotFoundError(f"找不到文件: {json_file}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.frames_data = []
        for frame in data:
            keypoints = np.array(frame['keypoints'])
            self.frames_data.append(keypoints)
        
        print(f"✅ 載入 {len(self.frames_data)} 幀 3D 數據")

    def compute_analysis(self):
        """計算姿態分析指標"""
        self.ankle_speeds = []
        self.knee_angles = []
        self.shoulder_angles = []
        self.wrist_speeds = []
        
        for i, points_3d in enumerate(self.frames_data):
            # 計算腳踝速度 (關節 9 和 10)
            if i > 0:
                ankle_l = np.linalg.norm(points_3d[9] - self.frames_data[i-1][9])
                ankle_r = np.linalg.norm(points_3d[10] - self.frames_data[i-1][10])
                ankle_speed = (ankle_l + ankle_r) / 2 * 19.93  # 轉換為 m/s
            else:
                ankle_speed = 0
            
            # 計算膝蓋角度 (11-13-15 或 12-14-16)
            knee_l = self._calculate_angle(points_3d[11], points_3d[13], points_3d[15])
            knee_angle = knee_l
            
            # 計算肩膀外旋角度 (6-8-10 三點角度)
            shoulder_angle = self._calculate_angle(points_3d[6], points_3d[8], points_3d[10])
            
            # 計算手腕速度 (關節 10)
            if i > 0:
                wrist_speed = np.linalg.norm(points_3d[10] - self.frames_data[i-1][10]) * 19.93
            else:
                wrist_speed = 0
            
            self.ankle_speeds.append(ankle_speed)
            self.knee_angles.append(knee_angle)
            self.shoulder_angles.append(shoulder_angle)
            self.wrist_speeds.append(wrist_speed)

    def _calculate_angle(self, p1, p2, p3):
        """計算三點形成的角度 (度)"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        return angle

    def init_ui(self):
        """初始化 UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 控制面板
        control_layout = QHBoxLayout()
        
        self.btn_play = QPushButton("播放")
        self.btn_play.clicked.connect(self.toggle_play)
        
        self.btn_reset = QPushButton("重置")
        self.btn_reset.clicked.connect(self.reset)
        
        self.btn_prev = QPushButton("◀ 上一幀")
        self.btn_prev.clicked.connect(self.prev_frame)
        
        self.btn_next = QPushButton("下一幀 ▶")
        self.btn_next.clicked.connect(self.next_frame)
        
        self.lbl_frame = QLabel(f"Frame: 0 / {len(self.frames_data)}")
        self.lbl_status = QLabel("狀態: 待機")
        
        control_layout.addWidget(self.btn_play)
        control_layout.addWidget(self.btn_reset)
        control_layout.addWidget(self.btn_prev)
        control_layout.addWidget(self.btn_next)
        control_layout.addWidget(self.lbl_frame)
        control_layout.addStretch()
        control_layout.addWidget(self.lbl_status)
        
        main_layout.addLayout(control_layout)
        
        # 幀滑桿
        slider_layout = QHBoxLayout()
        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setMinimum(0)
        self.slider_frame.setMaximum(len(self.frames_data) - 1 if self.frames_data else 0)
        self.slider_frame.valueChanged.connect(self.slider_changed)
        
        slider_layout.addWidget(QLabel("幀:"))
        slider_layout.addWidget(self.slider_frame)
        
        main_layout.addLayout(slider_layout)
        
        # 檢測結果表格
        result_layout = QHBoxLayout()
        
        # 左側：3D 可視化
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='white')
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=45, azimuth=110, elevation=10, up='-y', distance=2.5
        )
        axis = scene.visuals.XYZAxis(parent=self.view.scene)
        axis.transform = scene.transforms.STTransform(scale=(0.1, 0.1, 0.1))
        
        self.scatter = scene.visuals.Markers(parent=self.view.scene)
        self.head_lines = scene.visuals.Line(connect='segments', color='orange', width=3, parent=self.view.scene)
        self.left_lines = scene.visuals.Line(connect='segments', color='red', width=3, parent=self.view.scene)
        self.right_lines = scene.visuals.Line(connect='segments', color='blue', width=3, parent=self.view.scene)
        self.trunk_lines = scene.visuals.Line(connect='segments', color='green', width=3, parent=self.view.scene)
        
        result_layout.addWidget(self.canvas.native, 3)
        
        # 右側：檢測結果和數據
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 檢測結果
        result_table = QTableWidget()
        result_table.setRowCount(4)
        result_table.setColumnCount(2)
        result_table.setHorizontalHeaderLabels(["關鍵點", "幀編號"])
        result_table.setColumnWidth(0, 150)
        result_table.setColumnWidth(1, 100)
        
        # 設置標籤列
        result_table.setItem(0, 0, QTableWidgetItem("足踝接觸"))
        result_table.setItem(1, 0, QTableWidgetItem("最大外旋"))
        result_table.setItem(2, 0, QTableWidgetItem("釋放點"))
        result_table.setItem(3, 0, QTableWidgetItem("檢測狀態"))
        
        # 設置值列並保存引用以便更新
        self.result_values = [
            QTableWidgetItem("待檢測"),
            QTableWidgetItem("待檢測"),
            QTableWidgetItem("待檢測"),
            QTableWidgetItem("進行中...")
        ]
        
        for i, item in enumerate(self.result_values):
            result_table.setItem(i, 1, item)
        
        right_layout.addWidget(QLabel("關鍵點檢測結果:"))
        right_layout.addWidget(result_table)
        
        # 實時數據顯示
        data_table = QTableWidget()
        data_table.setRowCount(4)
        data_table.setColumnCount(2)
        data_table.setHorizontalHeaderLabels(["指標", "當前值"])
        data_table.setColumnWidth(0, 120)
        data_table.setColumnWidth(1, 130)
        
        self.data_values = [
            QTableWidgetItem("0.00 m/s"),
            QTableWidgetItem("0.00 °"),
            QTableWidgetItem("0.00 °"),
            QTableWidgetItem("0.00 m/s")
        ]
        
        data_table.setItem(0, 0, QTableWidgetItem("腳踝速度:"))
        data_table.setItem(1, 0, QTableWidgetItem("膝蓋角度:"))
        data_table.setItem(2, 0, QTableWidgetItem("肩膀外旋:"))
        data_table.setItem(3, 0, QTableWidgetItem("手腕速度:"))
        
        for i, item in enumerate(self.data_values):
            data_table.setItem(i, 1, item)
        
        right_layout.addWidget(QLabel("當前幀數據:"))
        right_layout.addWidget(data_table)
        right_layout.addStretch()
        
        result_layout.addWidget(right_panel, 1)
        main_layout.addLayout(result_layout)

    def slider_changed(self, value):
        """滑桿改變時"""
        self.current_frame_idx = value
        self.update_display()

    def update_display(self):
        """更新顯示"""
        if not self.frames_data:
            return
        
        frame_idx = self.current_frame_idx
        points_3d = self.frames_data[frame_idx]
        
        # 更新 3D 顯示
        self.scatter.set_data(points_3d, face_color='orange', size=8)
        
        # 更新 UI 標籤
        self.lbl_frame.setText(f"Frame: {frame_idx + 1} / {len(self.frames_data)}")
        
        # 更新數據表格
        if frame_idx < len(self.ankle_speeds):
            self.data_values[0].setText(f"{self.ankle_speeds[frame_idx]:.2f} m/s")
            self.data_values[1].setText(f"{self.knee_angles[frame_idx]:.1f} °")
            self.data_values[2].setText(f"{self.shoulder_angles[frame_idx]:.1f} °")
            self.data_values[3].setText(f"{self.wrist_speeds[frame_idx]:.2f} m/s")
        
        # 更新檢測狀態
        if self.detector.is_detection_complete():
            self.lbl_status.setText("✅ 檢測完成")
            self.result_values[0].setText(f"{self.detector.foot_contact_frame}")
            self.result_values[1].setText(f"{self.detector.max_shoulder_angle_frame}")
            self.result_values[2].setText(f"{self.detector.wrist_speed_frame}")
            self.result_values[3].setText("完成")
        else:
            self.lbl_status.setText("⏳ 檢測進行中...")
            if self.detector.foot_contact_frame:
                self.result_values[0].setText(f"{self.detector.foot_contact_frame} ✓")
            if self.detector.max_shoulder_angle_frame:
                self.result_values[1].setText(f"{self.detector.max_shoulder_angle_frame} ✓")
            if self.detector.wrist_speed_frame:
                self.result_values[2].setText(f"{self.detector.wrist_speed_frame} ✓")
        
        self.canvas.update()

    def toggle_play(self):
        """播放/暫停"""
        if self.is_playing:
            self.timer.stop()
            self.btn_play.setText("播放")
            self.is_playing = False
        else:
            self.timer.start()
            self.btn_play.setText("暫停")
            self.is_playing = True

    def play_next_frame(self):
        """播放下一幀"""
        if self.current_frame_idx < len(self.frames_data) - 1:
            self.current_frame_idx += 1
            
            # 實時檢測
            if self.current_frame_idx < len(self.ankle_speeds):
                self.detector.add_frame_data(
                    self.current_frame_idx,
                    self.ankle_speeds[self.current_frame_idx],
                    self.knee_angles[self.current_frame_idx],
                    self.shoulder_angles[self.current_frame_idx],
                    self.wrist_speeds[self.current_frame_idx]
                )
            
            self.slider_frame.blockSignals(True)
            self.slider_frame.setValue(self.current_frame_idx)
            self.slider_frame.blockSignals(False)
            self.update_display()
        else:
            self.toggle_play()

    def next_frame(self):
        """下一幀"""
        if self.current_frame_idx < len(self.frames_data) - 1:
            self.current_frame_idx += 1
            
            # 實時檢測
            if self.current_frame_idx < len(self.ankle_speeds):
                self.detector.add_frame_data(
                    self.current_frame_idx,
                    self.ankle_speeds[self.current_frame_idx],
                    self.knee_angles[self.current_frame_idx],
                    self.shoulder_angles[self.current_frame_idx],
                    self.wrist_speeds[self.current_frame_idx]
                )
            
            self.slider_frame.blockSignals(True)
            self.slider_frame.setValue(self.current_frame_idx)
            self.slider_frame.blockSignals(False)
            self.update_display()

    def prev_frame(self):
        """上一幀"""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.slider_frame.blockSignals(True)
            self.slider_frame.setValue(self.current_frame_idx)
            self.slider_frame.blockSignals(False)
            self.update_display()

    def reset(self):
        """重置"""
        self.current_frame_idx = 0
        self.detector = RealtimePhaseDetector()
        self.slider_frame.blockSignals(True)
        self.slider_frame.setValue(0)
        self.slider_frame.blockSignals(False)
        self.update_display()
        if self.is_playing:
            self.toggle_play()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 使用 calibrated_3d_frames.json 或其他 JSON 文件
    json_file = Path(__file__).parent / "calibrated_3d_frames.json"
    
    if len(sys.argv) > 1:
        json_file = Path(sys.argv[1])
    
    if not json_file.exists():
        print(f"❌ 找不到文件: {json_file}")
        print(f"用法: python realtime_pose_analysis.py [json_file_path]")
        sys.exit(1)
    
    window = RealtimePoseAnalysisWidget(str(json_file))
    window.show()
    
    sys.exit(app.exec_())
