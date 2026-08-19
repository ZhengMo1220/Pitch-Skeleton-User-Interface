"""
簡易雙執行緒測試程式
目的：驗證影片播放 + 相機監控可以同時運行
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import queue
import time
from threading import Thread


class DualThreadTest(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initVariables()
        
    def initUI(self):
        """建立簡單的測試介面"""
        self.setWindowTitle("雙執行緒測試 - 影片 + 相機")
        self.setGeometry(100, 100, 1200, 600)
        
        # 主佈局
        main_layout = QVBoxLayout()
        
        # 控制按鈕區
        control_layout = QHBoxLayout()
        self.video_btn = QPushButton("播放測試影片")
        self.video_btn.clicked.connect(self.toggleVideo)
        
        self.camera_btn = QPushButton("啟動相機監控")
        self.camera_btn.clicked.connect(self.toggleCamera)
        
        self.background_checkbox = QCheckBox("背景監控模式")
        self.background_checkbox.stateChanged.connect(self.toggleBackgroundMode)
        
        control_layout.addWidget(self.video_btn)
        control_layout.addWidget(self.camera_btn)
        control_layout.addWidget(self.background_checkbox)
        
        # 顯示區域
        display_layout = QHBoxLayout()
        
        # 左側：影片顯示
        video_container = QVBoxLayout()
        self.video_label = QLabel("影片區域")
        self.video_label.setMinimumSize(560, 400)
        self.video_label.setStyleSheet("border: 2px solid blue; background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_info = QLabel("影片狀態: 未播放")
        video_container.addWidget(self.video_label)
        video_container.addWidget(self.video_info)
        
        # 右側：相機顯示
        camera_container = QVBoxLayout()
        self.camera_label = QLabel("相機區域")
        self.camera_label.setMinimumSize(560, 400)
        self.camera_label.setStyleSheet("border: 2px solid green; background-color: black;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_info = QLabel("相機狀態: 未啟動")
        camera_container.addWidget(self.camera_label)
        camera_container.addWidget(self.camera_info)
        
        display_layout.addLayout(video_container)
        display_layout.addLayout(camera_container)
        
        # 狀態資訊
        self.status_label = QLabel("系統狀態: 就緒")
        self.status_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f0f0f0;")
        
        # 組合佈局
        main_layout.addLayout(control_layout)
        main_layout.addLayout(display_layout)
        main_layout.addWidget(self.status_label)
        
        self.setLayout(main_layout)
        
    def initVariables(self):
        """初始化變數"""
        # 影片相關
        self.video_playing = False
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.updateVideoFrame)
        self.video_cap = None
        self.current_frame_num = 0
        self.total_frames = 0
        
        # 相機相關
        self.camera_running = False
        self.camera_thread = None
        self.camera_queue = queue.Queue(maxsize=10)
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.updateCameraFrame)
        self.camera_frame_count = 0
        
        # 背景監控模式
        self.background_mode = False
        
    def toggleVideo(self):
        """切換影片播放狀態"""
        if not self.video_playing:
            # 開始播放 - 使用生成的測試影片
            self.startVideo()
        else:
            # 停止播放
            self.stopVideo()
            
    def startVideo(self):
        """開始播放測試影片（生成假影片）"""
        # 創建測試影片幀（模擬影片）
        self.video_playing = True
        self.current_frame_num = 0
        self.total_frames = 300  # 模擬300幀
        self.video_timer.start(33)  # 約30 FPS
        self.video_btn.setText("停止影片")
        self.video_info.setText(f"影片狀態: 播放中 (0/{self.total_frames})")
        self.updateStatus("影片播放中")
        
    def stopVideo(self):
        """停止影片播放"""
        self.video_playing = False
        self.video_timer.stop()
        self.video_btn.setText("播放測試影片")
        self.video_info.setText("影片狀態: 已停止")
        self.video_label.setText("影片區域")
        self.updateStatus("影片已停止")
        
    def updateVideoFrame(self):
        """更新影片幀"""
        if self.current_frame_num >= self.total_frames:
            self.stopVideo()
            return
            
        # 生成測試幀（藍色漸變）
        frame = np.zeros((400, 560, 3), dtype=np.uint8)
        intensity = int((self.current_frame_num / self.total_frames) * 255)
        frame[:, :] = [intensity, 0, 255 - intensity]  # BGR
        
        # 繪製幀數
        cv2.putText(frame, f"Video Frame: {self.current_frame_num}", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 顯示
        self.displayImage(frame, self.video_label)
        self.video_info.setText(f"影片狀態: 播放中 ({self.current_frame_num}/{self.total_frames})")
        
        self.current_frame_num += 1
        
    def toggleCamera(self):
        """切換相機監控狀態"""
        if not self.camera_running:
            self.startCamera()
        else:
            self.stopCamera()
            
    def startCamera(self):
        """啟動相機監控（使用執行緒模擬）"""
        self.camera_running = True
        self.camera_frame_count = 0
        
        # 啟動背景執行緒產生幀
        self.camera_thread = Thread(target=self.cameraThreadWorker, daemon=True)
        self.camera_thread.start()
        
        # 啟動主執行緒的顯示更新
        self.camera_timer.start(33)  # 約30 FPS
        
        self.camera_btn.setText("停止相機")
        self.camera_info.setText("相機狀態: 執行中")
        self.updateStatus("相機監控啟動")
        
    def stopCamera(self):
        """停止相機監控"""
        self.camera_running = False
        self.camera_timer.stop()
        
        # 清空佇列
        while not self.camera_queue.empty():
            try:
                self.camera_queue.get_nowait()
            except:
                break
                
        self.camera_btn.setText("啟動相機監控")
        self.camera_info.setText("相機狀態: 已停止")
        self.camera_label.setText("相機區域")
        self.updateStatus("相機已停止")
        
    def cameraThreadWorker(self):
        """背景執行緒：持續產生相機幀"""
        while self.camera_running:
            # 生成測試幀（綠色變化，模擬動作偵測）
            frame = np.zeros((400, 560, 3), dtype=np.uint8)
            
            # 模擬動作偵測：每15幀變化一次
            if (self.camera_frame_count % 15) < 5:
                # 動作階段：亮綠色
                frame[:, :] = [0, 255, 0]
                status = "偵測到動作！"
            else:
                # 穩定階段：暗綠色
                frame[:, :] = [0, 100, 0]
                status = "監控中..."
            
            # 繪製資訊
            cv2.putText(frame, f"Camera Frame: {self.camera_frame_count}", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, status, 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # 放入佇列
            if not self.camera_queue.full():
                self.camera_queue.put((frame.copy(), self.camera_frame_count))
            
            self.camera_frame_count += 1
            time.sleep(0.033)  # 模擬相機幀率
            
    def updateCameraFrame(self):
        """從佇列取得並顯示相機幀"""
        if not self.camera_queue.empty():
            frame, frame_num = self.camera_queue.get()
            self.displayImage(frame, self.camera_label)
            
            # 模擬偵測
            if (frame_num % 15) < 5:
                self.camera_info.setText(f"相機狀態: 偵測到動作! (幀: {frame_num})")
                self.camera_info.setStyleSheet("color: red; font-weight: bold;")
            else:
                self.camera_info.setText(f"相機狀態: 監控中 (幀: {frame_num})")
                self.camera_info.setStyleSheet("color: black;")
                
    def toggleBackgroundMode(self, state):
        """切換背景監控模式"""
        self.background_mode = (state == Qt.Checked)
        
        if self.background_mode:
            self.updateStatus("背景監控模式: 已啟用 - 影片播放時同步監控相機")
            self.background_checkbox.setStyleSheet("color: green; font-weight: bold;")
            
            # 如果影片正在播放，自動啟動相機
            if self.video_playing and not self.camera_running:
                self.startCamera()
        else:
            self.updateStatus("背景監控模式: 已停用")
            self.background_checkbox.setStyleSheet("")
            
    def displayImage(self, frame, label):
        """顯示影像到 QLabel"""
        # 轉換為 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # 轉換為 QImage
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # 縮放以適應 label
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
        
    def updateStatus(self, message):
        """更新狀態列"""
        timestamp = time.strftime("%H:%M:%S")
        status_text = f"[{timestamp}] {message}"
        
        if self.background_mode and self.video_playing and self.camera_running:
            status_text += " | ⚡ 雙執行緒並行中"
            
        self.status_label.setText(status_text)
        
    def closeEvent(self, event):
        """關閉視窗時清理資源"""
        self.stopVideo()
        self.stopCamera()
        event.accept()


def main():
    app = QApplication(sys.argv)
    test_window = DualThreadTest()
    test_window.show()
    
    print("=" * 60)
    print("雙執行緒測試程式")
    print("=" * 60)
    print("功能說明：")
    print("1. 點擊「播放測試影片」- 在左側顯示模擬影片（藍色漸變）")
    print("2. 點擊「啟動相機監控」- 在右側顯示模擬相機（綠色閃爍=偵測動作）")
    print("3. 勾選「背景監控模式」- 影片播放時自動啟動相機監控")
    print("4. 觀察兩邊是否同時流暢運行，無卡頓或衝突")
    print("=" * 60)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
