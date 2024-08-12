import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
# from camera_widget import PoseCameraTabControl
from camera_widget_beta import PoseCameraTabControl
from video_widget_beta import PoseVideoTabControl
from main_window import Ui_MainWindow

class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_tabs()

    def init_tabs(self):
        self.camera_tab = PoseCameraTabControl()  # 正確初始化
        self.ui.Two_d_Tab.addTab(self.camera_tab, "2D 相機")  # 添加Tab
        self.video_tab = PoseVideoTabControl()
        self.ui.Two_d_Tab.addTab(self.video_tab, "2D 影片")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec_())

