"""
相機內參校正工具（單台相機，棋盤格法）

與外參校正分開：內參只跟「相機機身 + 鏡頭」的組合有關，
與安裝位置無關，因此一台一台單獨校正即可，不需要雙機同步或 GPIO 觸發線。

使用方式：
    cd Src\\UI_Control
    python calib.py

流程：下拉選單選相機 → 連接 → 確認棋盤格規格 → 擷取 10~15 組不同角度
      → 執行校正 → 自動存到 Db/Calibration/intrinsic/
      → 第 4 區指定正面/側面，寫入 Db/Calibration/camera_roles.json
"""
import os
import sys
from datetime import datetime

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QTextEdit,
                             QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QMessageBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

import PySpin
from camera_objects.single_camera.flir_camera_system import FlirCameraSystem
from cv_utils import calib_store

CONFIG_PATH = r".\camera_config\GH3_camera_config.yaml"


def list_cameras():
    """列舉目前連接的所有 FLIR 相機，回傳 [(序號, 型號), ...]"""
    result = []
    system = PySpin.System.GetInstance()
    try:
        cam_list = system.GetCameras()
        for i in range(cam_list.GetSize()):
            cam = cam_list.GetByIndex(i)
            try:
                tl = cam.TLDevice
                sn = tl.DeviceSerialNumber.GetValue()
                model = tl.DeviceModelName.GetValue()
            except Exception:
                sn, model = f"<index {i}>", "unknown"
            result.append((sn, model))
            del cam
        cam_list.Clear()
    finally:
        system.ReleaseInstance()
    return result


class CalibWindow(QMainWindow):
    # 預覽效能參數：全解析度做角點偵測太慢，預覽改用縮圖且降低偵測頻率。
    # 實際擷取樣本時仍使用全解析度，不影響校正精度。
    PREVIEW_SCALE = 0.4          # 偵測用縮圖比例
    PREVIEW_DETECT_EVERY = 3     # 每 N 次畫面更新才偵測一次

    def __init__(self):
        super().__init__()
        self.setWindowTitle("相機內參校正工具（單台）")
        self.resize(900, 900)

        self.camera = None
        self.last_frame = None
        self.objpoints = []
        self.imgpoints = []
        self.captured_count = 0
        self.image_size = None

        self._preview_tick = 0
        self._preview_found = False
        self._preview_corners = None

        self.initUI()
        self.refresh_cameras()
        self.refresh_role_combos()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(60)

    # ---------- UI ----------

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 相機選擇
        cam_group = QGroupBox("1. 選擇相機")
        cam_layout = QHBoxLayout(cam_group)
        self.cmb_camera = QComboBox()
        self.cmb_camera.setMinimumWidth(380)
        self.btn_refresh = QPushButton("重新掃描")
        self.btn_connect = QPushButton("連接")
        self.btn_disconnect = QPushButton("中斷連接")
        self.btn_refresh.clicked.connect(self.refresh_cameras)
        self.btn_connect.clicked.connect(self.connect_camera)
        self.btn_disconnect.clicked.connect(self.disconnect_camera)
        self.btn_disconnect.setEnabled(False)
        cam_layout.addWidget(self.cmb_camera)
        cam_layout.addWidget(self.btn_refresh)
        cam_layout.addWidget(self.btn_connect)
        cam_layout.addWidget(self.btn_disconnect)
        layout.addWidget(cam_group)

        # 棋盤格參數
        board_group = QGroupBox("2. 棋盤格規格（內角點數量 = 格子數 - 1）")
        board_layout = QHBoxLayout(board_group)
        self.spn_cols = QSpinBox()
        self.spn_cols.setRange(3, 30)
        self.spn_cols.setValue(8)
        self.spn_rows = QSpinBox()
        self.spn_rows.setRange(3, 30)
        self.spn_rows.setValue(11)
        self.spn_square = QDoubleSpinBox()
        self.spn_square.setRange(0.1, 1000.0)
        self.spn_square.setDecimals(2)
        self.spn_square.setValue(10.0)
        self.spn_square.setSuffix(" mm")
        board_layout.addWidget(QLabel("內角點 寬:"))
        board_layout.addWidget(self.spn_cols)
        board_layout.addWidget(QLabel("高:"))
        board_layout.addWidget(self.spn_rows)
        board_layout.addWidget(QLabel("每格邊長:"))
        board_layout.addWidget(self.spn_square)
        board_layout.addStretch()
        layout.addWidget(board_group)

        # 畫面
        self.lbl_view = QLabel("尚未連接相機")
        self.lbl_view.setAlignment(Qt.AlignCenter)
        self.lbl_view.setStyleSheet("border: 2px solid gray; background: #222; color: #ccc;")
        self.lbl_view.setMinimumHeight(420)
        layout.addWidget(self.lbl_view)

        # 操作按鈕
        act_group = QGroupBox("3. 擷取與校正")
        act_layout = QHBoxLayout(act_group)
        self.btn_capture = QPushButton("擷取樣本 (空白鍵)")
        self.btn_calibrate = QPushButton("執行校正")
        self.btn_reset = QPushButton("清空樣本")
        self.btn_capture.clicked.connect(self.capture_sample)
        self.btn_calibrate.clicked.connect(self.run_calibration)
        self.btn_reset.clicked.connect(self.reset_samples)
        act_layout.addWidget(self.btn_capture)
        act_layout.addWidget(self.btn_calibrate)
        act_layout.addWidget(self.btn_reset)
        layout.addWidget(act_group)

        # 角色指定：拍照時決定哪台當 primary，3D 計算時決定用哪台的內參
        role_group = QGroupBox("4. 指定正面 / 側面相機（拍外參照片與 3D 計算都依此設定）")
        role_layout = QHBoxLayout(role_group)
        self.cmb_front = QComboBox()
        self.cmb_front.setMinimumWidth(140)
        self.cmb_side = QComboBox()
        self.cmb_side.setMinimumWidth(140)
        self.btn_save_roles = QPushButton("儲存角色")
        self.btn_save_roles.clicked.connect(self.save_roles)
        role_layout.addWidget(QLabel("正面 (primary):"))
        role_layout.addWidget(self.cmb_front)
        role_layout.addWidget(QLabel("側面 (secondary):"))
        role_layout.addWidget(self.cmb_side)
        role_layout.addWidget(self.btn_save_roles)
        role_layout.addStretch()
        layout.addWidget(role_group)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMinimumHeight(160)
        layout.addWidget(self.log_area)

    def log(self, msg):
        # 最新訊息插在最上方，擷取成功與否不必捲動就看得到；多行訊息維持原本順序
        cursor = self.log_area.textCursor()
        cursor.movePosition(cursor.Start)
        cursor.insertText(f"[{datetime.now():%H:%M:%S}] {msg}\n")
        self.log_area.moveCursor(cursor.Start)

    # ---------- 相機 ----------

    def refresh_cameras(self):
        self.cmb_camera.clear()
        try:
            cams = list_cameras()
        except Exception as e:
            self.log(f"掃描相機失敗: {e}")
            return

        if not cams:
            self.log("未偵測到任何相機。請確認相機已接上、且 SpinView 等程式已關閉。")
            return

        for sn, model in cams:
            self.cmb_camera.addItem(f"SN={sn}   {model}", userData=sn)
        self.log(f"偵測到 {len(cams)} 台相機。")
        if hasattr(self, "cmb_front"):
            self.refresh_role_combos()

    def connect_camera(self):
        if self.camera is not None:
            self.log("已有相機連接中，請先中斷連接。")
            return

        sn = self.cmb_camera.currentData()
        if not sn:
            self.log("請先選擇相機。")
            return

        try:
            self.camera = FlirCameraSystem(CONFIG_PATH, sn)
            self.connected_sn = sn
            self.log(f"已連接相機 SN={sn}")
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
            self.cmb_camera.setEnabled(False)
            self.reset_samples()
        except Exception as e:
            self.camera = None
            self.log(f"連接失敗: {e}")

    def disconnect_camera(self):
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception as e:
                self.log(f"釋放相機時發生問題: {e}")
            self.camera = None
        self.last_frame = None
        self.lbl_view.setText("已中斷連接")
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.cmb_camera.setEnabled(True)
        self.log("已中斷連接。")

    # ---------- 畫面 ----------

    def update_display(self):
        if self.camera is None:
            return

        try:
            ok, raw = self.camera.get_grayscale_image()
        except Exception as e:
            self.log(f"取得畫面失敗: {e}")
            return

        if not ok or raw is None:
            return

        # 相機輸出為 BayerRG8，轉成 BGR 才能顯示與偵測
        frame = cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)
        self.last_frame = frame
        if self.image_size is None:
            h, w = frame.shape[:2]
            self.image_size = (w, h)

        display = frame
        self._preview_tick += 1

        # findChessboardCorners 在 1920x1084 全解析度下很慢，會拖垮畫面更新。
        # 預覽只需確認「板子有沒有被看到」，因此縮圖偵測 + 每 N 幀才做一次。
        if self._preview_tick % self.PREVIEW_DETECT_EVERY == 0:
            pattern = (self.spn_cols.value(), self.spn_rows.value())
            small = cv2.resize(frame, None, fx=self.PREVIEW_SCALE, fy=self.PREVIEW_SCALE,
                               interpolation=cv2.INTER_AREA)
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            found, corners_small = cv2.findChessboardCorners(
                gray_small, pattern,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK
            )
            self._preview_found = found
            # 角點座標換算回原始解析度，才能疊在全尺寸畫面上
            self._preview_corners = corners_small / self.PREVIEW_SCALE if found else None

        if self._preview_found and self._preview_corners is not None:
            display = frame.copy()
            pattern = (self.spn_cols.value(), self.spn_rows.value())
            cv2.drawChessboardCorners(display, pattern, self._preview_corners, True)

        self.lbl_view.setPixmap(self.to_pixmap(display))

    def to_pixmap(self, bgr):
        # 先用 OpenCV 縮到顯示尺寸再交給 Qt，避免每幀搬運整張 1920x1084 的資料
        target_w = max(self.lbl_view.width(), 1)
        target_h = max(self.lbl_view.height(), 1)
        h, w = bgr.shape[:2]
        scale = min(target_w / w, target_h / h)
        if scale < 1.0:
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_img.copy())

    # ---------- 校正 ----------

    def capture_sample(self):
        if self.camera is None or self.last_frame is None:
            self.log("尚未連接相機或畫面未就緒。")
            return

        pattern = (self.spn_cols.value(), self.spn_rows.value())
        square = self.spn_square.value()

        gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern, None)

        if not found:
            self.log("擷取失敗：未完整偵測到棋盤格。請調整角度/距離，確認整塊板子都在畫面內。")
            return

        # 亞像素精修，提升角點定位精度
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
        objp *= square

        self.objpoints.append(objp)
        self.imgpoints.append(corners)
        self.captured_count += 1
        self.log(f"已擷取第 {self.captured_count} 組樣本")

    def run_calibration(self):
        if self.captured_count < 10:
            self.log(f"樣本僅 {self.captured_count} 組，建議至少 10 組不同角度再校正。")
            return
        if self.image_size is None:
            self.log("尚未取得影像尺寸。")
            return

        self.log("開始計算校正參數...")
        QApplication.processEvents()

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.image_size, None, None
        )

        # 重投影誤差：逐張計算後取平均，比 calibrateCamera 回傳的 RMS 更直觀
        total_err = 0.0
        for i in range(len(self.objpoints)):
            proj, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            total_err += cv2.norm(self.imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        mean_err = total_err / len(self.objpoints)

        # 結果合成一則訊息，避免「最新在上」時多行順序顛倒
        self.log(
            f"校正完成 SN={self.connected_sn}\n"
            f"  RMS: {ret:.4f}\n"
            f"  平均重投影誤差: {mean_err:.4f} px  (< 0.5 為佳)\n"
            f"  fx={mtx[0,0]:.2f}  fy={mtx[1,1]:.2f}\n"
            f"  cx={mtx[0,2]:.2f}  cy={mtx[1,2]:.2f}\n"
            f"  畸變係數: {np.round(dist.ravel(), 6)}"
        )

        os.makedirs(calib_store.INTRINSIC_DIR, exist_ok=True)
        out_name = os.path.join(calib_store.INTRINSIC_DIR, f"calib_SN{self.connected_sn}.npz")
        np.savez(
            out_name,
            mtx=mtx, dist=dist,
            image_size=np.array(self.image_size),
            rms=ret, mean_reproj_err=mean_err,
            pattern=np.array([self.spn_cols.value(), self.spn_rows.value()]),
            square_size=self.spn_square.value(),
            serial_number=self.connected_sn,
            sample_count=self.captured_count,
        )
        self.log(f"已儲存至 {out_name}")

        self._append_to_json(mtx, dist, ret, mean_err)

        if mean_err > 1.0:
            QMessageBox.warning(
                self, "重投影誤差偏高",
                f"平均重投影誤差為 {mean_err:.3f} px，偏高。\n\n"
                "可能原因：棋盤格規格填錯、樣本角度不夠多樣、或板子不夠平整。\n"
                "建議清空樣本後重新擷取。"
            )

    def _append_to_json(self, mtx, dist, rms, mean_err):
        """把這台相機的內參累積寫入 intrinsics_all.json（以序號為鍵）。
        3D 計算時再依 camera_roles.json 取出正面/側面各自的內參。
        """
        entry = {
            "intrinsic_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist(),
            "fx": float(mtx[0, 0]),
            "fy": float(mtx[1, 1]),
            "cx": float(mtx[0, 2]),
            "cy": float(mtx[1, 2]),
            "image_size": list(self.image_size),
            "rms": float(rms),
            "mean_reproj_err": float(mean_err),
            "pattern": [self.spn_cols.value(), self.spn_rows.value()],
            "square_size_mm": self.spn_square.value(),
            "sample_count": self.captured_count,
            "calibrated_at": datetime.now().isoformat(timespec="seconds"),
        }

        try:
            count = calib_store.save_intrinsic(self.connected_sn, entry)
        except Exception as e:
            self.log(f"寫入 {calib_store.INTRINSICS_JSON} 失敗：{e}")
            return

        self.log(f"已累積寫入 {calib_store.INTRINSICS_JSON}（目前共 {count} 台）")
        self.refresh_role_combos()

    def refresh_role_combos(self):
        """更新正面/側面下拉選單：可選「已校正」或「目前連接」的相機，並預選現有角色"""
        try:
            sns = list(calib_store.load_intrinsics().keys())
        except Exception:
            sns = []
        for i in range(self.cmb_camera.count()):
            sn = self.cmb_camera.itemData(i)
            if sn and sn not in sns:
                sns.append(sn)

        roles = calib_store.load_roles()
        for cmb, role_idx in ((self.cmb_front, 0), (self.cmb_side, 1)):
            current = cmb.currentText() or (roles[role_idx] if roles else "")
            cmb.clear()
            cmb.addItems(sns)
            idx = cmb.findText(current)
            if idx >= 0:
                cmb.setCurrentIndex(idx)

    def save_roles(self):
        """寫入 camera_roles.json：拍外參照片時決定 primary，3D 計算時決定內參"""
        front_sn = self.cmb_front.currentText()
        side_sn = self.cmb_side.currentText()

        if not front_sn or not side_sn:
            self.log("請先選擇正面與側面相機。")
            return
        if front_sn == side_sn:
            self.log("正面與側面不能是同一台相機。")
            return

        calib_store.save_roles(front_sn, side_sn)
        intr = calib_store.load_intrinsics()
        missing = [sn for sn in (front_sn, side_sn) if sn not in intr]
        msg = (f"已儲存角色至 {calib_store.ROLES_JSON}\n"
               f"  正面 = SN{front_sn}\n"
               f"  側面 = SN{side_sn}")
        if missing:
            msg += f"\n  注意：{', '.join('SN' + s for s in missing)} 尚未做內參校正"
        self.log(msg)

    def reset_samples(self):
        self.objpoints = []
        self.imgpoints = []
        self.captured_count = 0
        self.log("已清空樣本。")

    # ---------- 其他 ----------

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.capture_sample()
        elif event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = CalibWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
