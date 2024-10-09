import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout

from camera_widget import PoseCameraTabControl
from video_widget import PoseVideoTabControl
from pitch_widget import PosePitchTabControl
from main_window import Ui_MainWindow
from utils.model import Model

import sys
import os


class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.model = Model()
        self.init_tabs()

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

