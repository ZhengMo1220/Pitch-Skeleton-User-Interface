import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from cv_utils.cv_thread import VideoCaptureThread

class CalibWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("棒球系統 - 側面相機棋盤格校正工具")
        self.resize(800, 800)

        # --- 校正參數設定 ---
        self.pattern_size = (7, 7)  # 棋盤格內角點數量 (寬, 高)
        self.square_size = 100       # 每個格子的實際尺寸 (mm)
        
        # 存儲空間
        self.objpoints = []  # 3D 空間點
        self.imgpoints_side = []  # 側面相機 2D 點
        self.captured_count = 0

        self.initUI()
        
        # --- 使用 VideoCaptureThread 初始化 FLIR 相機 ---
        self.last_frame_s = None
        
        try:
            self.video_thread = VideoCaptureThread()
            self.video_thread.frame_ready.connect(self.on_frame_ready)
            self.video_thread.start_capture()
            self.log_area.append("✅ FLIR 相機初始化成功")
        except Exception as e:
            self.log_area.append(f"❌ FLIR 相機初始化失敗: {str(e)}")
            self.video_thread = None
        
        # 使用 QTimer 更新顯示（只顯示，不獲取幀）
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(30)

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 1. 畫面顯示區域
        self.lbl_side = QLabel("側面相機 (正在讀取...)")
        self.lbl_side.setAlignment(Qt.AlignCenter)
        self.lbl_side.setStyleSheet("border: 2px solid gray;")
        layout.addWidget(self.lbl_side)

        # 2. 控制按鈕區域
        btn_layout = QHBoxLayout()
        self.btn_capture = QPushButton("擷取樣本 (Space)")
        self.btn_calibrate = QPushButton("執行相機校正 (Calibrate)")
        self.btn_reset = QPushButton("清空樣本")
        
        self.btn_capture.clicked.connect(self.capture_sample)
        self.btn_calibrate.clicked.connect(self.run_calibration)
        self.btn_reset.clicked.connect(self.reset_samples)
        
        btn_layout.addWidget(self.btn_capture)
        btn_layout.addWidget(self.btn_calibrate)
        btn_layout.addWidget(self.btn_reset)
        layout.addLayout(btn_layout)

        # 3. 狀態記錄區
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        layout.addWidget(self.log_area)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_img).scaled(760, 600, Qt.KeepAspectRatio)

    def capture_sample(self):
        """從側面相機擷取棋盤格角點"""
        if self.last_frame_s is None:
            self.log_area.append("❌ 擷取失敗：無法獲取相機畫面")
            return
            
        gray_s = cv2.cvtColor(self.last_frame_s, cv2.COLOR_BGR2GRAY)
        ret_s, corners_s = cv2.findChessboardCorners(gray_s, self.pattern_size, None)

        if ret_s:
            # 定義 3D 世界座標 (Z=0)
            objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
            objp *= self.square_size

            self.objpoints.append(objp)
            self.imgpoints_side.append(corners_s)
            
            self.captured_count += 1
            self.log_area.append(f"✅ 成功擷取第 {self.captured_count} 組樣本")
        else:
            self.log_area.append("❌ 擷取失敗：相機未完整偵測到棋盤格")

    def run_calibration(self):
        if self.captured_count < 10:
            self.log_area.append("⚠️ 樣本不足，建議至少擷取 10 組以上不同的角度。")
            return

        self.log_area.append("⏳ 開始計算校正參數，請稍候...")
        QApplication.processEvents()

        h, w = self.last_frame_s.shape[:2]

        # 側面相機校正內參
        ret_s, mtx_s, dist_s, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_side, (w, h), None, None
        )

        self.log_area.append("🎉 校正完成！")
        self.log_area.append(f"RMS Error: {ret_s:.4f}")
        self.log_area.append(f"\n相機內參矩陣 (Camera Matrix):\n{mtx_s}")
        self.log_area.append(f"\n畸變係數 (Distortion Coefficients):\n{dist_s.ravel()}")
        
        # 儲存側面相機內參
        np.savez("side_camera_calib.npz", mtx=mtx_s, dist=dist_s)
        self.log_area.append("\n✅ 內參已儲存至 side_camera_calib.npz")

    def reset_samples(self):
        self.objpoints = []
        self.imgpoints_side = []
        self.captured_count = 0
        self.log_area.clear()
        self.log_area.append("已重設所有樣本。")


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.capture_sample()
        elif event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def on_frame_ready(self, frame_f, frame_s):
        """接收視頻線程發送的幀（只使用側面相機）"""
        self.last_frame_s = frame_f

    def update_display(self):
        """更新顯示內容"""
        if self.last_frame_s is not None:
            # 即時偵測棋盤格 (僅供視覺確認，不儲存)
            gray_s = cv2.cvtColor(self.last_frame_s, cv2.COLOR_BGR2GRAY)
            found_s, corners_s = cv2.findChessboardCorners(gray_s, self.pattern_size, None)
            frame_s_display = self.last_frame_s.copy()
            if found_s:
                cv2.drawChessboardCorners(frame_s_display, self.pattern_size, corners_s, found_s)

            # 更新 GUI 畫面
            self.lbl_side.setPixmap(self.convert_cv_qt(frame_s_display))

    def closeEvent(self, event):
        """窗口關閉時釋放相機資源"""
        if self.video_thread is not None:
            self.video_thread.stop_capture()
        self.timer.stop()
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    window = CalibWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
