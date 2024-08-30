import pyqtgraph as pg
from PyQt5.QtGui import QPainter, QPen, QColor, QImage, QPixmap, QFont
import numpy as np
import pandas as pd

def init_graph(frame_range, kpt_name = "右手肘"):
    graph =  pg.PlotWidget()
    title = f'<span style = "color: blue; font-size: 15px">{kpt_name}角度</span>'
    graph.setTitle(f'{title}')
    
    font = QFont()
    font.setPixelSize(18)
    graph.addLegend(offset=(150, 5), labelTextSize="18pt")
    graph.setLabel('left', '<span style = "font-size: 18px" >角度 (度)</span>', color= "blue")
    graph.setLabel('bottom', f'<span style = "font-size: 18px" >幀 (fps: 120)</span>')
    graph.getAxis("bottom").setStyle(tickFont=font)
    graph.getAxis("left").setStyle(tickFont=font)
    graph.setXRange(0, frame_range-1)
    graph.setYRange(0, 180)
    y_ticks = [(i, str(i)) for i in np.arange(0, 210, 30)]
    graph.getPlotItem().getAxis('left').setTicks([y_ticks])
    graph.getPlotItem().getAxis('left').setPen(color=QColor("blue"))
    graph.getPlotItem().getAxis('left').setTextPen(color = QColor("blue"))

    return graph

def update_graph(graph, angle_info, kpt_name = 'r_elbow_angle'):
    graph.clear()
    kpt_angle =  get_angle_info(angle_info, kpt_name)
    kpt_time = angle_info['frame_number'].unique()
    graph.plot(kpt_time, kpt_angle, pen='b')    

    return graph

def get_angle_info(analyze_info: pd.DataFrame, angle_name: str):
    angles = []

    for _, row in analyze_info.iterrows():
        angle_data = row['angle']
        if angle_name in angle_data:
            angles.append(angle_data[angle_name][0],)

    return angles
   
   