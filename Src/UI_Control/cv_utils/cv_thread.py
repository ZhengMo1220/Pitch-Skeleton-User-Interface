import os
import json
import logging

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QMutex

# 相機序號設定檔：放在 UI_Control 目錄下，格式為
#   {"front": "25462483", "side": "25462481"}
# 存在時優先採用，避免為了換相機而修改程式碼。
CAMERA_SN_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "camera_serials.json"
)

# 設定檔不存在且自動偵測失敗時的最後備援（原本寫死的值）
FALLBACK_SN1 = "25462483"
FALLBACK_SN2 = "25462481"


def detect_connected_serials():
    """列出目前實際連接的 FLIR 相機序號（依 PySpin 列舉順序）。"""
    try:
        import PySpin
    except ImportError:
        return []

    serials = []
    system = PySpin.System.GetInstance()
    try:
        cam_list = system.GetCameras()
        for i in range(cam_list.GetSize()):
            cam = cam_list.GetByIndex(i)
            try:
                sn = cam.TLDevice.DeviceSerialNumber.GetValue()
                if sn and sn != "unknown":
                    serials.append(sn)
            except Exception:
                pass
            del cam
        cam_list.Clear()
    except Exception as e:
        logging.warning("列舉相機失敗: %s", e)
    finally:
        system.ReleaseInstance()
    return serials


def resolve_camera_serials(sn1=None, sn2=None):
    """決定要使用的兩台相機序號。

    優先順序：明確指定 > camera_serials.json > 自動偵測 > 寫死的備援值。
    回傳 (sn1, sn2)。
    """
    if sn1 and sn2:
        return sn1, sn2

    # 設定檔
    if os.path.exists(CAMERA_SN_CONFIG):
        try:
            with open(CAMERA_SN_CONFIG, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            front = cfg.get("front")
            side = cfg.get("side")
            if front and side:
                logging.info("使用 camera_serials.json: front=%s side=%s", front, side)
                return front, side
        except Exception as e:
            logging.warning("讀取 %s 失敗: %s", CAMERA_SN_CONFIG, e)

    # 自動偵測：取前兩台
    detected = detect_connected_serials()
    if len(detected) >= 2:
        logging.info("自動偵測到相機: %s，採用前兩台", detected)
        return detected[0], detected[1]

    logging.warning("無法自動決定相機序號，使用預設值 %s / %s", FALLBACK_SN1, FALLBACK_SN2)
    return FALLBACK_SN1, FALLBACK_SN2


def normalize_fps(fps, default=30.0, min_fps=1.0, max_fps=240.0) -> float:
    try:
        value = float(fps)
    except (TypeError, ValueError):
        return float(default)

    if not np.isfinite(value) or value <= 0:
        return float(default)

    return float(max(min_fps, min(max_fps, value)))

class VideoToImagesThread(QThread):
    emit_signal = pyqtSignal([list, list, float, int])
    _run_flag=True
    def __init__(self,video_path,video_path_2):
        super(VideoToImagesThread, self).__init__()
        self.video_path=video_path
        self.video_path_2=video_path_2

    def video_to_frame(self, input_video_path):
        video_images = []
        vidcap = cv2.VideoCapture(input_video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        success, image = vidcap.read()
        count = 0
        while success:
            video_images.append(image)
            count += 1
            success, image = vidcap.read()
        vidcap.release()
        return video_images, fps, count

    def run(self):
        # capture from web cam
        video_images, fps_1, count = self.video_to_frame(self.video_path)
        video_images_2, fps_2, count_2 = self.video_to_frame(self.video_path_2)
        # 以第一支影片的 FPS 為主；若第一支無效則回退第二支
        fps = fps_1 if np.isfinite(fps_1) and fps_1 > 0 else fps_2
        # 以較小幀數避免越界
        count = min(count, count_2)
        _run_flag=False
        self.emit_signal.emit(video_images, video_images_2, fps, count)
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        # if self.cap!=None:
        #     self.cap.release()
        print("stop video to image thread")
    
    def isFinished(self):
        print(" finish thread")

from camera_objects import FlirCameraSystem,DualFlirSystem
class VideoCaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray,np.ndarray)

    def __init__(self, parent=None, sn1=None, sn2=None):
        super().__init__(parent)
        CONFIG = ".\camera_config\GH3_camera_config.yaml"
        CONFIG_2 = ".\camera_config\GH3_camera_config_2.yaml"

        # 序號不再寫死：可由呼叫端指定、camera_serials.json 設定、或自動偵測
        SN1, SN2 = resolve_camera_serials(sn1, sn2)
        self.serial_front = SN1
        self.serial_side = SN2
        print(f"[VideoCaptureThread] front={SN1}  side={SN2}")

        self.camera1 = FlirCameraSystem(CONFIG,SN1)
        self.camera2 = FlirCameraSystem(CONFIG_2,SN2)
        self.cameras = DualFlirSystem(self.camera1, self.camera2)
        # if not self.camera1.isOpened() or not self.camera2.isOpened():
        #     raise ValueError(f"Cannot open one or both camera.")
        self.running = False
    
    def start_capture(self):
        self.running = True
        if not self.isRunning():
            self.start()

    def stop_capture(self):
        self.running = False
        self.wait()
    
    def __del__(self):
        """確保資源被釋放"""
        if hasattr(self, 'running') and self.running:
            self.running = False
        if hasattr(self, 'cameras') and self.cameras is not None:
            try:
                self.cameras.release()
            except:
                pass

    def run(self):
        # import gc
        count = 0  # 移到迴圈外，避免每次都重置為0
        while self.running:    
            ret, frame1, frame2 = self.cameras.get_grayscale_images()
            if ret:
                count += 1  # 先遞增計數
                if count % 3 == 0:
                    # 只在發送時轉換為 BGR，不影響骨架偵測用的原始幀
                    rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BayerBG2BGR)
                    rgb_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BayerBG2BGR)
                    self.frame_ready.emit(rgb_frame1, rgb_frame2)  # 每3幀發送一次
            else:  # 例外處理
                print("Warning: Failed to capture frame")
                break
        print(f"[VideoCaptureThread] 停止: 總捕獲 {count} 幀")
        self.cameras.release()

class VideoWriterThread(QThread):
    # 使用 signal 傳遞狀態更新
    writeFinished = pyqtSignal()

    def __init__(self, filepath, filepath_2, frame_width, frame_height, fps=30, codec='mp4v'):
        super().__init__()
        self.filepath = filepath
        self.filepath_2 = filepath_2
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = normalize_fps(fps)
        self.codec = codec
        self.is_writing = False
        self.frame = None
        self.frame_2 = None
        self.lock = QMutex()

    def run(self):
        # 在 run 方法中進行視頻寫入
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(self.filepath, fourcc, self.fps, (self.frame_width, self.frame_height))
        self.writer_2 = cv2.VideoWriter(self.filepath_2, fourcc, self.fps, (self.frame_width, self.frame_height))
        # import gc
        # 持續寫入直到 is_writing 被設置為 False
        while self.is_writing:
            self.lock.lock()
            if self.frame is not None and self.frame_2 is not None:
                self.writer.write(self.frame)
                self.writer_2.write(self.frame_2)
            self.lock.unlock()

        # 錄製結束後釋放資源
        self.writer.release()
        self.writer_2.release()
        self.writeFinished.emit()

    def start_writing(self):
        self.is_writing = True
        self.start()

    def stop_writing(self):
        self.is_writing = False
        self.wait()

    def write_frame(self, frame, frame_2):
        # self.lock.lock()
        self.frame = frame
        self.frame_2 = frame_2
        # self.lock.unlock()

    def release(self):
        self.stop_writing()
        self.wait()  # 等待執行緒安全結束

class FramesToVideoWriterThread(QThread):
    """后台线程：将帧列表写入两部视频文件"""
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, frames_list, video_path_1, video_path_2, fps=60):
        super().__init__()
        self.frames_list = frames_list  # [(frame1, frame2), ...]
        self.video_path_1 = video_path_1
        self.video_path_2 = video_path_2
        self.fps = normalize_fps(fps)
        self._running = True

    def run(self):
        if not self.frames_list or len(self.frames_list) == 0:
            self.error.emit("No frames to write")
            return
        
        try:
            first_frame, first_frame_2 = self.frames_list[0]
            h, w = first_frame.shape[:2]
            h2, w2 = first_frame_2.shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer1 = cv2.VideoWriter(self.video_path_1, fourcc, self.fps, (w, h))
            writer2 = cv2.VideoWriter(self.video_path_2, fourcc, self.fps, (w2, h2))
            
            if not writer1.isOpened() or not writer2.isOpened():
                self.error.emit("Failed to open video writer")
                return
            
            for idx, (f, f2) in enumerate(self.frames_list):
                if not self._running:
                    break
                writer1.write(f)
                writer2.write(f2)
            
            writer1.release()
            writer2.release()
            print(f"[ThreadWrite] Saved {len(self.frames_list)} frames to {self.video_path_1}")
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"Error writing video: {str(e)}")

    def stop(self):
        self._running = False
        self.wait()