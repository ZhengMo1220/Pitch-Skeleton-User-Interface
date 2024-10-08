import pyqtgraph as pg
from PyQt5.QtGui import QFont, QColor
from .analyze import PoseAnalyzer
import numpy as np
import pandas as pd

class GraphPlotter():
    def __init__(self, pose_analyzer: PoseAnalyzer, angle_name: str = "右手肘"):
        """Initialize GraphPlotter with frame range and keypoint name."""
        pg.setConfigOptions(foreground=QColor(113,148,116), antialias = True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.angle_name = angle_name
        self.graph = pg.PlotWidget()
        self.pose_analyzer = pose_analyzer

    def _init_graph(self, frame_range: int) -> pg.PlotWidget:
        """Set up the plot widget with labels, axis, and initial configuration."""
        title = f'<span style="color: blue; font-size: 30px">{self.angle_name}角度 (度)</span>'
        self.graph.setTitle(f'{title}')

        # Setting up fonts and styles
        font = QFont()
        font.setPixelSize(30)
        self.graph.addLegend(offset=(150, 5), labelTextSize="30pt")
        self.graph.setLabel('left', '<span style="font-size: 30px">角度 (度)</span>', color="blue")
        self.graph.setLabel('bottom', '<span style="font-size: 30px">幀 (fps: 120)</span>')
        self.graph.getAxis("bottom").setStyle(tickFont=font)
        self.graph.getAxis("left").setStyle(tickFont=font)
        self.graph.setXRange(0, frame_range - 1)
        self.graph.setYRange(0, 180)
        
        # Setting Y-axis ticks
        y_ticks = [(i, str(i)) for i in np.arange(0, 210, 30)]
        self.graph.getPlotItem().getAxis('left').setTicks([y_ticks])
        self.graph.getPlotItem().getAxis('left').setPen(color=QColor("blue"))
        self.graph.getPlotItem().getAxis('left').setTextPen(color=QColor("blue"))

    def updateGraph(self, frame_num: int):
        """Update the graph with the latest angle data for the specified frame."""
      
        self.graph.clear()

        # Fetch current angle data for the specified frame
        _, angle_info = self.pose_analyzer.get_frame_angle_data(frame_num, self.angle_name)
        try:
            angle_value = int(angle_info[0])
            title = f'<span style="color: blue; font-size: 30px">{self.angle_name}角度({int(angle_value):03}度)</span>'
        except TypeError:
            # print(f"角度數據格式錯誤: {angle_value}")
            return
        self.graph.setTitle(f'{title}')
        # Plot angle data for all frames
        kpt_times, kpt_angles = self.pose_analyzer.get_frame_angle_data(angle_name=self.angle_name)
        self.graph.plot(kpt_times, kpt_angles, pen='b')


        # Create a scatter plot for the point
        scatter = pg.ScatterPlotItem([frame_num], [angle_value], size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255))
        
        # Add the scatter plot to the graph
        self.graph.addItem(scatter)
    
    def setAngleName(self, angle_name):
        self.angle_name = angle_name
    
    def reset(self):
        self.graph = pg.PlotWidget()

