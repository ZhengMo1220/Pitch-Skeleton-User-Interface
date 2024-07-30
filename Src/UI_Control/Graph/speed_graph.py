import numpy as np
import pyqtgraph as pg
from PyQt5.QtGui import QPainter, QPen, QColor, QImage, QPixmap, QFont


class Speedgraph:
    def __init__(self, speed_range, frame_ratio, length_ratio):
        self.speed_range = [0,12]
        self.frame_ratio = 1/120
        self.lenth_ratio = 1
        self.start_frame_num = 0
        self.speed_graph =  pg.PlotWidget()
     
    def init_speed_graph(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        title = "Speed (Average: 0.00m/s)"
        font = QFont()
        font.setPixelSize(15)
        self.speed_graph.addLegend(offset=(150, 5), labelTextSize="10pt")
        self.speed_graph.setLabel('left', 'Velocity (m/s)')
        self.speed_graph.setLabel('bottom', f'Frame (fps: {self.ui.fps_input.value()})')
        self.speed_graph.getAxis("bottom").setStyle(tickFont=font)
        self.speed_graph.getAxis("left").setStyle(tickFont=font)
        self.speed_graph.setXRange(0, self.total_images-1)
        self.speed_graph.setYRange(self.speed_range[0], self.speed_range[1])
        self.speed_graph.setWindowTitle(title)         
        self.speed_graph.setTitle(title)
        y_ticks = [(i, str(i)) for i in np.arange(0, 14, 2)]
        self.speed_graph.getPlotItem().getAxis('left').setTicks([y_ticks])
        self.show_graph(self.speed_graph, self.speed_scene, self.ui.speed_view)

    def obtain_speed(self, person_kpt): 
        if self.start_frame_num != 0: 
            person_kpt = person_kpt.to_numpy()
            l_x_kpt_datas = []
            l_y_kpt_datas = []
            pos_x = []
            pos_y = []
            v = []
            t = []
            l = self.ui.frame_slider.value() - self.start_frame_num
            for i in range(l):
                l_x_kpt_datas.append(person_kpt[i][5][0])
                l_y_kpt_datas.append(person_kpt[i][5][1])
            for i in range(len(l_x_kpt_datas)):
                mod = i % 30
                if mod == 0 :
                    pos_x.append([l_x_kpt_datas[i]])
                    pos_y.append([l_y_kpt_datas[i]])
            if len(pos_x) > 1 :
                for i in range(len(pos_x)):
                    if i > 0:
                        pos_f = np.array([pos_x[i-1], pos_y[i-1]])
                        pos_s = np.array([pos_x[i], pos_y[i]])
                        if pos_f[0] > self.end_line_pos:
                            length = np.linalg.norm(pos_f - pos_s)
                            temp_v = (length * self.length_ratio) / (30 * self.frame_ratio)
                            v.append(temp_v) 
                    else:
                        v.append(0)
            for i in range(len(v)):
                temp_t = self.start_frame_num + i * 30
                t.append(temp_t)
            t = t[1:]
            v = v[1:]
            return t,v
            if len(v) > 0:
                self.update_speed_graph(t, v)
    
    def update_speed_graph(self,person_kpt, start_frame):
        t = []
        v = []
        if len(person_kpt) > 0 and self.start_frame_num ==0 :
            self.start_frame_num = start_frame
        if self.start_frame_num != 0:
            t, v = self.obtain_speed(person_kpt)
        if len(v)>0 and len(t)>0:
            mean = np.round(np.mean(v[:max(t)]), 2)
            title = f"Speed (Average: {mean}m/s)"
            font = QFont()
            font.setPixelSize(15)
            self.speed_graph.setXRange(min(t), min(t)+200)
            self.speed_graph.setWindowTitle(title)         
            self.speed_graph.setTitle(title)
            self.speed_graph.plot(t, v, pen='r')    
            # 設置 x 軸刻度
            x_ticks = [(i, str(i)) for i in np.arange(min(t), min(t)+200, 30)]
            self.speed_graph.getPlotItem().getAxis('bottom').setTicks([x_ticks])
            for i, val in zip(t, v):
                formatted_val = "{:.2f}m/s".format(val)
                label = pg.TextItem(text=formatted_val, anchor=(0.5, 0.5), color=(0, 0, 0))
                label.setPos(i, val+1)
                self.speed_graph.addItem(label)