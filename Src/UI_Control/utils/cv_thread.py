import typing
import cv2
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, Qt, QThread
from .util import video_to_frame
import numpy as np
import sys
import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from .timer import Timer


class VideoToImagesThread(QThread):
    emit_signal = pyqtSignal([list,int,int])   
    _run_flag=True
    def __init__(self,video_path):
        super(VideoToImagesThread, self).__init__()
        self.video_path=video_path
    def run(self):
        # capture from web cam
        video_images, fps, count= video_to_frame(self.video_path)
        _run_flag=False
        self.emit_signal.emit(video_images, fps, count)
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        # if self.cap!=None:
        #     self.cap.release()
        print("stop video to image thread")
    
    def isFinished(self):
        print(" finish thread")

class VideoCaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, parent=None, camera_index=0):
        super().__init__(parent)
        self.timer = Timer()
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera with index {camera_index}")
        self.running = False

    def start_capture(self):
        self.running = True
        if not self.isRunning():
            self.start()

    def stop_capture(self):
        self.running = False
        self.wait()

    def run(self):
        while self.running:     
            ret, frame = self.cap.read()
            if ret:
                # if count % 6 ==0:
                self.frame_ready.emit(frame)  # 發送影像
            else:  # 例外處理
                print("Warning: Failed to capture frame")
                break

        self.cap.release()

class VideoWriterThread(QThread):
    def __init__(self, output_file, frame_queue, parent=None):
        super().__init__(parent)
        self.output_file = output_file
        self.frame_queue = frame_queue
        self.running = False
        self.out = None

    def start_writing(self, frame_width, frame_height, fps):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_file, fourcc, fps, (frame_width, frame_height))
        self.running = True
        if not self.isRunning():
            self.start()
        print(f"Started writing video to {self.output_file} at {fps} FPS with resolution {frame_width}x{frame_height}")

    def stop_writing(self):
        self.running = False
        self.wait()
        print("Stopped writing video.")

    def run(self):
        try:
            while self.running or not self.frame_queue.empty():
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    self.out.write(frame)
        except Exception as e:
            print(f"Error occurred in VideoWriterThread: {e}")
        finally:
            self.out.release()
            print("Released video writer resources.")

class VideoWriter:
    def __init__(self, filepath, frame_width, frame_height, fps=60, codec='mp4v'):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(filepath, fourcc, fps, (frame_width, frame_height))

    def write(self, frame):
        if self.writer is not None:
            self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None