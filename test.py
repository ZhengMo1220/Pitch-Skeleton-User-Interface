import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow

app = QApplication([])

# 创建一个主窗口
win = QMainWindow()
win.setWindowTitle('pyqtgraph Example')

# 创建一个 PlotWidget
plot_widget = pg.PlotWidget()

# 创建一个 ViewBox，并将其设置为左下角为原点
view = plot_widget.getViewBox()
view.setAspectLocked(False)

# 设置变换矩阵，调整 X 轴和 Y 轴的方向
view.invertX(False)  # 不反转X轴
view.invertY(True)   # 反转Y轴，使得Y轴向上增长

# 设置 X 轴和 Y 轴的范围
plot_widget.setXRange(0, 10)
plot_widget.setYRange(0, 10)

# 添加数据
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]
plot_widget.plot(x, y, pen='r')

# 显示图表
win.setCentralWidget(plot_widget)
win.show()

app.exec_()
