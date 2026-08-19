import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout

from camera_widget import PoseCameraTabControl
from video_widget_2 import PoseVideoTabControl
from pitch_widget import PosePitchTabControl
from video_widget_compare import PoseVideoCompareTabControl
from main_window import Ui_MainWindow
from utils.model import Model

class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.model = Model()
        self.init_tabs()

    def init_tabs(self):
        # self.camera_tab = PoseCameraTabControl()
        # self.ui.Two_d_Tab.addTab(self.camera_tab, "2D 相機")
        self.pitch_tab = PosePitchTabControl(self.model, self)
        self.ui.Two_d_Tab.addTab(self.pitch_tab, "2D")
        self.video_tab = PoseVideoTabControl(self.model)
        self.ui.Two_d_Tab.addTab(self.video_tab, "3D")    
        self.video_compare_tab = PoseVideoCompareTabControl(self.model)
        self.ui.Two_d_Tab.addTab(self.video_compare_tab, "回放比較")    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec_())

