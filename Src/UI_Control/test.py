from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsTextItem
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
import sys
from PyQt5.QtWidgets import QApplication

class CountdownView(QGraphicsView):
    def __init__(self):
        super().__init__()

        # 設置場景
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # 初始化倒數時間
        self.countdown_value = 3

        # 創建倒數顯示的文字
        self.text_item = QGraphicsTextItem()
        self.text_item.setFont(QFont("Arial", 60))  # 設置字體和大小
        self.text_item.setDefaultTextColor(Qt.red)  # 設置字體顏色
        self.text_item.setPos(100, 100)  # 設置文字位置
        self.text_item.setTextWidth(200)
        self.text_item.setOpacity(0.5)  # 設置透明度為50%
        self.scene.addItem(self.text_item)

        # 更新顯示倒數
        self.updateCountdown()

        # 設置計時器，每秒更新一次
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateCountdown)
        self.timer.start(1000)  # 每1000ms = 1秒觸發一次

    def updateCountdown(self):
        # 更新倒數顯示
        if self.countdown_value > 0:
            self.text_item.setPlainText(str(self.countdown_value))
            self.countdown_value -= 1
        else:
            # 停止計時器
            self.text_item.setPlainText("Go!")
            self.timer.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CountdownView()
    window.show()
    sys.exit(app.exec_())
