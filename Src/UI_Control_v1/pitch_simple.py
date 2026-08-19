"""
投球偵測系統 - 最小可用版本 (MVP)
功能：開啟相機 → 自動偵測投球 → 自動錄3秒 → 播放影片
"""

import sys
import os
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QMessageBox, QListWidget, QCheckBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np

# 導入必要模組
from utils.timer import Timer
from utils.model import Model
from cv_utils.cv_control import Camera
from skeleton.detect_skeleton import PoseEstimater
from PitchingAnalyzer import PitchingAnalyzer


class SimplePitchSystem(QWidget):
    def __init__(self):
        super().__init__()
        self.model = Model()
        self.initUI()
        self.initComponents()
        
    def initUI(self):
        """初始化簡化的UI"""
        self.setWindowTitle("投球偵測系統 - 簡化版")
        self.setGeometry(100, 100, 1400, 800)
        
        # 主佈局
        main_layout = QVBoxLayout()
        
        # === 控制區 ===
        control_layout = QHBoxLayout()
        
        # 相機控制
        self.camera_btn = QPushButton("📹 開啟相機")
        self.camera_btn.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50; color: white;")
        self.camera_btn.clicked.connect(self.toggleCamera)
        
        # 投球偵測開關
        self.detect_checkbox = QCheckBox("🎯 自動偵測投球")
        self.detect_checkbox.setStyleSheet("font-size: 14px;")
        self.detect_checkbox.stateChanged.connect(self.toggleDetection)
        
        # 狀態顯示
        self.status_label = QLabel("狀態: 就緒")
        self.status_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f0f0f0;")
        
        control_layout.addWidget(self.camera_btn)
        control_layout.addWidget(self.detect_checkbox)
        control_layout.addStretch()
        control_layout.addWidget(self.status_label)
        
        # === 顯示區 ===
        display_layout = QHBoxLayout()
        
        # 左側：主攝影機視窗
        left_layout = QVBoxLayout()
        self.camera_label_1 = QLabel("攝影機1")
        self.camera_label_1.setMinimumSize(640, 480)
        self.camera_label_1.setStyleSheet("border: 2px solid #2196F3; background-color: black;")
        self.camera_label_1.setAlignment(Qt.AlignCenter)
        self.camera_info_1 = QLabel("FPS: 0 | 狀態: 未啟動")
        left_layout.addWidget(self.camera_label_1)
        left_layout.addWidget(self.camera_info_1)
        
        # 右側：第二攝影機視窗
        right_layout = QVBoxLayout()
        self.camera_label_2 = QLabel("攝影機2")
        self.camera_label_2.setMinimumSize(640, 480)
        self.camera_label_2.setStyleSheet("border: 2px solid #FF9800; background-color: black;")
        self.camera_label_2.setAlignment(Qt.AlignCenter)
        self.camera_info_2 = QLabel("FPS: 0 | 狀態: 未啟動")
        right_layout.addWidget(self.camera_label_2)
        right_layout.addWidget(self.camera_info_2)
        
        display_layout.addLayout(left_layout)
        display_layout.addLayout(right_layout)
        
        # === 錄影列表區 ===
        list_layout = QVBoxLayout()
        list_label = QLabel("📼 已錄製影片")
        list_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.video_list = QListWidget()
        self.video_list.setMaximumHeight(150)
        self.video_list.itemDoubleClicked.connect(self.playSelectedVideo)
        
        self.refresh_btn = QPushButton("🔄 重新整理")
        self.refresh_btn.clicked.connect(self.refreshVideoList)
        
        list_layout.addWidget(list_label)
        list_layout.addWidget(self.video_list)
        list_layout.addWidget(self.refresh_btn)
        
        # 組合佈局
        main_layout.addLayout(control_layout)
        main_layout.addLayout(display_layout)
        main_layout.addLayout(list_layout)
        
        self.setLayout(main_layout)
        
    def initComponents(self):
        """初始化核心元件"""
        # 相機
        self.camera = Camera()
        self.camera_running = False
        
        # 骨架偵測器
        self.pose_estimater_1 = PoseEstimater(self.model)
        self.pose_estimater_2 = PoseEstimater(self.model)
        
        # 投球分析器
        self.pitching_analyzer = PitchingAnalyzer()
        
        # 計時器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.updateFrame)
        
        self.record_timer = None
        
        # 錄影相關
        self.is_recording = False
        self.is_detecting = False
        self.output_dir = None
        self.pitcher_id = "01"  # 預設投手編號
        
        # FPS計數
        self.frame_count = 0
        self.fps = 0
        
        # 載入影片列表
        self.refreshVideoList()
        
    def toggleCamera(self):
        """開關相機"""
        if not self.camera_running:
            # 開啟相機
            try:
                frame_width, frame_height, fps = self.camera.toggleCamera(True)
                self.model.setImageSize((frame_width, frame_height))
                self.camera_running = True
                self.update_timer.start(33)  # 約30 FPS更新
                
                self.camera_btn.setText("⏹ 關閉相機")
                self.camera_btn.setStyleSheet("font-size: 16px; padding: 10px; background-color: #f44336; color: white;")
                self.updateStatus("相機已開啟")
                
                # 啟用偵測選項
                self.detect_checkbox.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"無法開啟相機：{str(e)}")
        else:
            # 關閉相機
            self.camera.toggleCamera(False)
            self.camera_running = False
            self.update_timer.stop()
            
            self.camera_btn.setText("📹 開啟相機")
            self.camera_btn.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50; color: white;")
            self.updateStatus("相機已關閉")
            
            # 停用偵測
            self.detect_checkbox.setChecked(False)
            self.detect_checkbox.setEnabled(False)
            
            # 清空顯示
            self.camera_label_1.setText("攝影機1")
            self.camera_label_2.setText("攝影機2")
            
    def toggleDetection(self, state):
        """開關投球偵測"""
        self.is_detecting = (state == Qt.Checked)
        
        if self.is_detecting:
            self.updateStatus("投球偵測已啟用 - 等待動作...")
            self.pitching_analyzer.reset()
        else:
            self.updateStatus("投球偵測已停用")
            
    def updateFrame(self):
        """更新畫面（主迴圈）"""
        if not self.camera_running:
            return
            
        # 從相機取得幀
        if self.camera.frame_buffer.empty() or self.camera.frame_buffer_2.empty():
            return
            
        frame1 = self.camera.frame_buffer.get().copy()
        frame2 = self.camera.frame_buffer_2.get().copy()
        
        # 骨架偵測
        _, _, fps1 = self.pose_estimater_1.detectKpt(frame1, is_video=False)
        _, _, fps2 = self.pose_estimater_2.detectKpt(frame2, is_video=False)
        
        # 繪製骨架
        frame1_display = self.drawSkeleton(frame1, self.pose_estimater_1)
        frame2_display = self.drawSkeleton(frame2, self.pose_estimater_2)
        
        # 投球偵測
        if self.is_detecting and not self.is_recording:
            self.detectPitching()
        
        # 如果正在錄影，顯示倒數
        if self.record_timer is not None:
            remaining = self.record_timer.get_remaining_time()
            if remaining > 0:
                cv2.putText(frame1_display, f"RECORDING: {remaining}s", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(frame2_display, f"RECORDING: {remaining}s", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                # 錄影結束
                self.stopRecording()
        
        # 顯示畫面
        self.displayFrame(frame1_display, self.camera_label_1)
        self.displayFrame(frame2_display, self.camera_label_2)
        
        # 更新資訊
        self.camera_info_1.setText(f"FPS: {fps1:.0f} | 骨架: {len(self.pose_estimater_1.kpt_buffer)}")
        self.camera_info_2.setText(f"FPS: {fps2:.0f} | 骨架: {len(self.pose_estimater_2.kpt_buffer)}")
        
    def drawSkeleton(self, frame, pose_estimater):
        """簡單繪製骨架"""
        frame_copy = frame.copy()
        
        # 繪製所有偵測到的人
        for _, person in pose_estimater.kpt_buffer.iterrows():
            # 繪製邊界框
            x1, y1, x2, y2 = map(int, person['bbox'])
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 顯示ID
            cv2.putText(frame_copy, f"ID: {person['person_id']}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 繪製該人的關鍵點
            if 'keypoints' in person and person['keypoints'] is not None:
                kpts = person['keypoints']
                for i, kpt in enumerate(kpts):
                    if len(kpt) >= 3:
                        x, y, score = kpt[0], kpt[1], kpt[2]
                        if score > 0.5:
                            cv2.circle(frame_copy, (int(x), int(y)), 3, (0, 255, 255), -1)
        
        return frame_copy
        
    def detectPitching(self):
        """投球偵測邏輯"""
        # 自動選擇最大的人
        if self.pose_estimater_1.kpt_buffer.empty:
            return
            
        # 找最大的人
        max_area = 0
        max_person = None
        for _, row in self.pose_estimater_1.pre_person_df.iterrows():
            x1, y1, x2, y2 = map(int, row['bbox'])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                max_person = row
        
        if max_person is None:
            return
        
        # 取得該人的骨架關鍵點
        if 'keypoints' not in max_person or max_person['keypoints'] is None:
            return
            
        person_kpts = max_person['keypoints']
        
        # 使用 PitchingAnalyzer 分析
        signal = self.pitching_analyzer.process(person_kpts)
        
        if signal == "START":
            # 開始錄影
            self.startRecording()
            self.updateStatus("🔴 偵測到投球！開始錄影...")
            
        elif signal == "STOP":
            self.updateStatus("投球動作完成 - 等待錄影結束...")
            
    def startRecording(self):
        """開始錄影3秒"""
        if self.is_recording:
            return
            
        self.is_recording = True
        
        # 建立輸出目錄
        current_date = datetime.now().strftime("%Y%m%d")
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f'../../Db/Record/Pitcher{self.pitcher_id}_{current_date}/{current_time}'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 開始錄製
        video_path1 = os.path.join(self.output_dir, f'Camera1_{current_time}.mp4')
        video_path2 = os.path.join(self.output_dir, f'Camera2_{current_time}.mp4')
        
        self.camera.startRecording(video_path1, video_path2)
        
        # 啟動3秒計時器
        self.record_timer = Timer(3)
        self.record_timer.start()
        
    def stopRecording(self):
        """停止錄影"""
        if not self.is_recording:
            return
            
        self.camera.stop_recording()
        self.is_recording = False
        self.record_timer = None
        
        # 重置分析器
        self.pitching_analyzer.reset()
        
        # 更新影片列表
        self.refreshVideoList()
        
        # 詢問是否播放
        reply = QMessageBox.question(
            self, 
            "錄影完成", 
            "投球影片已儲存！\n\n是否立即播放？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 播放最新的影片
            self.playLatestVideo()
        
        self.updateStatus("✅ 錄影完成 - 等待下次投球...")
        
    def refreshVideoList(self):
        """重新載入影片列表"""
        self.video_list.clear()
        
        record_dir = "../../Db/Record"
        if not os.path.exists(record_dir):
            return
        
        # 掃描所有影片
        video_files = []
        for root, dirs, files in os.walk(record_dir):
            for file in files:
                if file.endswith('.mp4'):
                    full_path = os.path.join(root, file)
                    video_files.append(full_path)
        
        # 按時間排序（最新的在前）
        video_files.sort(reverse=True)
        
        # 加入列表
        for video_path in video_files[:20]:  # 只顯示最新20個
            video_name = os.path.basename(video_path)
            folder_name = os.path.basename(os.path.dirname(video_path))
            display_name = f"{folder_name} / {video_name}"
            
            item = self.video_list.addItem(display_name)
            # 儲存完整路徑作為資料
            self.video_list.item(self.video_list.count()-1).setData(Qt.UserRole, video_path)
            
    def playSelectedVideo(self, item):
        """播放選中的影片"""
        video_path = item.data(Qt.UserRole)
        self.playVideo(video_path)
        
    def playLatestVideo(self):
        """播放最新錄製的影片"""
        if self.video_list.count() > 0:
            item = self.video_list.item(0)
            video_path = item.data(Qt.UserRole)
            self.playVideo(video_path)
            
    def playVideo(self, video_path):
        """播放影片（使用系統預設播放器）"""
        try:
            import subprocess
            if sys.platform == 'win32':
                os.startfile(video_path)
            elif sys.platform == 'darwin':
                subprocess.call(['open', video_path])
            else:
                subprocess.call(['xdg-open', video_path])
                
            self.updateStatus(f"正在播放: {os.path.basename(video_path)}")
        except Exception as e:
            QMessageBox.warning(self, "播放失敗", f"無法播放影片：{str(e)}")
            
    def displayFrame(self, frame, label):
        """在QLabel上顯示影像"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
        
    def updateStatus(self, message):
        """更新狀態列"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_label.setText(f"[{timestamp}] {message}")
        
    def closeEvent(self, event):
        """關閉視窗時清理資源"""
        if self.camera_running:
            self.camera.toggleCamera(False)
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = SimplePitchSystem()
    window.show()
    
    print("=" * 60)
    print("投球偵測系統 - 最小可用版本 (MVP)")
    print("=" * 60)
    print("功能：")
    print("1. 點擊「開啟相機」啟動雙攝影機")
    print("2. 勾選「自動偵測投球」開始監控")
    print("3. 系統自動偵測投球動作並錄影3秒")
    print("4. 雙擊影片列表可播放已錄製的影片")
    print("=" * 60)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
