import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
# from camera_widget import PoseCameraTabControl

from camera_widget_beta import PoseCameraTabControl
from video_widget_beta_two import PoseVideoTabControl
from pitch_widget import PosePitchTabControl
from main_window import Ui_MainWindow

from utils.set_parser import set_detect_parser, set_tracker_parser
from mmcv.transforms import Compose
from mmengine.logging import print_log
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "tracker"))
from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer
from mmpose.apis import init_model as init_pose_estimator
from mmpose.utils import adapt_mmdet_pipeline
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.model = self.init_model()
        self.init_tabs()
    
    def init_model(self):
        self.detect_args = set_detect_parser()
        self.tracker_args = set_tracker_parser()
        self.detector = init_detector(
            self.detect_args.det_config, self.detect_args.det_checkpoint, device=self.detect_args.device)
        self.detector.cfg.test_dataloader.dataset.pipeline[
            0].type = 'mmdet.LoadImageFromNDArray'
        self.detector_test_pipeline = Compose(self.detector.cfg.test_dataloader.dataset.pipeline)
        self.pose_estimator = init_pose_estimator(
            self.detect_args.pose_config,
            self.detect_args.pose_checkpoint,
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=self.detect_args.draw_heatmap)))
        )
        self.tracker = BoTSORT(self.tracker_args, frame_rate=30.0)
        
        
        return {
            "Detector": {
                        "args": self.detect_args,
                        "detector": self.detector,
                        "test_pipeline": self.detector_test_pipeline
                        },
            "Tracker":  {
                        "args": self.tracker_args,
                        "tracker": self.tracker
                        },
            "Pose Estimator": {
                                "pose estimator": self.pose_estimator
                            }
        }

    def init_tabs(self):
        self.camera_tab = PoseCameraTabControl(self.model)
        self.ui.Two_d_Tab.addTab(self.camera_tab, "2D 相機")
        self.video_tab = PoseVideoTabControl(self.model)
        self.ui.Two_d_Tab.addTab(self.video_tab, "2D 影片")
        self.pitch_tab = PosePitchTabControl(self.model)
        self.ui.Two_d_Tab.addTab(self.pitch_tab, "2D 投手")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec_())

