import pyqtgraph as pg
from PyQt5.QtGui import QPainter, QPen, QColor, QImage, QPixmap, QFont
import numpy as np

def init_graph(fps, speed_range, frame_range):
    graph =  pg.PlotWidget()
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    # title = "Speed (Average: 0.00m/s)"
    stride_title = f'<span style = "color: red; font-size: 15px">步幅 (平均長度: {0.00}m)</span>'
    speed_title = f'<span style = "color: blue; font-size: 15px">速度 (平均速率: {0.00}m/sec)</span>'
    graph.setTitle(f'{speed_title}<br>{stride_title}')
    
    font = QFont()
    font.setPixelSize(18)
    graph.addLegend(offset=(150, 5), labelTextSize="18pt")
    graph.setLabel('left', '<span style = "font-size: 18px" >速率 (m/sec)</span>', color= "blue")
    graph.setLabel('bottom', f'<span style = "font-size: 18px" >幀 (fps: {fps})</span>')
    graph.setLabel('right', '<span style = "font-size: 18px" >長度 (m)</span>', color = "red")
    graph.getAxis("bottom").setStyle(tickFont=font)
    graph.getAxis("left").setStyle(tickFont=font)
    graph.getAxis("right").setStyle(tickFont=font)
    graph.setXRange(0, frame_range-1)
    graph.setYRange(speed_range[0], speed_range[1])
    speed_y_ticks = [(i, str(i)) for i in np.arange(0, 16, 2)]
    distance_y_ticks = [(i, str(i/4)) for i in np.arange(0, 16, 2)]
    graph.getPlotItem().getAxis('left').setTicks([speed_y_ticks])
    graph.getPlotItem().getAxis('left').setPen(color=QColor("blue"))
    graph.getPlotItem().getAxis('left').setTextPen(color = QColor("blue"))
    graph.getPlotItem().getAxis('right').setTicks([distance_y_ticks])
    graph.getPlotItem().getAxis('right').setPen(color=QColor("red"))
    graph.getPlotItem().getAxis('right').setTextPen(color = QColor("red"))
    return graph

def update_graph(graph, analyze_information):
    graph.clear()
    stride_time = analyze_information['stride_time'][1:]
    stride_speed = analyze_information['stride_speed']
    stride_length = analyze_information['stride_length']
    stride_title = f'<font color = "red">步幅 (平均長度: {0.00}m)</font>'
    speed_title = f'<font color="blue">速度 (平均速率: {0.00}m/sec)</font>'
    
    if len(stride_time)>0:
        graph.setXRange(min(stride_time)-30 ,min(stride_time)+100)
        x_ticks = [(i, str(i)) for i in stride_time]
        graph.getPlotItem().getAxis('bottom').setTicks([x_ticks])
    if len(stride_length)> 0:
        mean_stride = np.round(np.mean(stride_length), 2)
        stride_title = f'<font color = "red">步幅 (平均長度: {mean_stride}m)</font>'
         # 添加條形圖
        md = [8*i for i in stride_length]
        # 設置步幅 x 軸刻度
        barItem = pg.BarGraphItem(x=stride_time, y=0, height=md, width=2, brush='r')
        graph.addItem(barItem)
        # 添加每個值方的標籤
        for t, val in zip(stride_time, stride_length):
            formatted_val = "{:.2f} m".format(np.round(val,2))
            label = pg.TextItem(text=formatted_val, anchor=(0.5, 0.5), color=(255, 0, 0))
            label.setPos(t, 12)
            graph.addItem(label)
    
    if len(stride_speed)> 0:
        mean_speed = np.round(np.mean(stride_speed), 2)
        speed_title = f'<font color="blue">速度 (平均速率: {mean_speed}m/sec)</font>'
         #speed
        graph.plot(stride_time, stride_speed, pen='b')    

        for t, val in zip(stride_time, stride_speed):
            formatted_val = str(np.round(val,2))
            text = f"{formatted_val} m/sec"
            label = pg.TextItem(text = text, anchor=(0.5, 0.5), color=(0, 0, 255))
            label.setPos(t, val-1)
            graph.addItem(label)

    graph.setTitle(f"{speed_title}<br>{stride_title}")
    # 創建字體對象
    return graph

   
   