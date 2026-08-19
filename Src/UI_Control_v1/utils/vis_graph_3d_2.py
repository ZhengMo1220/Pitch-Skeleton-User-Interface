import pyqtgraph as pg
from PyQt5.QtGui import QFont, QColor
import numpy as np
# from .analyze_3d import PoseAnalyzer # 假設這行在你的環境中是有效的

class GraphPlotter():
    def __init__(self, pose_analyzer, speed_name=None, title_names=None):
        """Initialize GraphPlotter with frame range and keypoint name."""
        # 設定整體配置
        pg.setConfigOptions(foreground=QColor(0, 0, 0), antialias=True)
        pg.setConfigOption('background', 'w')

        self.speed_names = speed_name if speed_name else []
        self.title_names = title_names if title_names else []
        self.graph = pg.GraphicsLayoutWidget()
        self.pose_analyzer = pose_analyzer
        
        # 儲存曲線參照，方便後續 update 使用 setData (比 clear 重畫更有效率)
        self.curves = {} 
        self.scatters = {}
        
        # 顏色庫
        self.curve_colors = ["#000000", "#800080", "#004BAC", "#C60000", '#00aa00']

    def _init_graph(self, frame_range: int):
        """
        建立單一圖表，但擁有雙 Y 軸 (左: 角度, 右: 速度)
        """
        self.graph.clear()
        self.curves = {}
        self.scatters = {}

        # --- 1. 建立主圖 (Left Axis - Angle) ---
        self.p1 = self.graph.addPlot(row=0, col=0, title="動作分析數據總覽")
        self.p1.setLabel('bottom', '幀 (Frame)')
        angle_label_html = '<span style="color:#800080;">膝蓋角度</span> <span style="color:#004BAC;">\肩峰連線與髖關節連線角度</span> <span style="color:#C60000;">\肩外旋角度</span>'
        self.p1.setLabel('left', angle_label_html)
        self.p1.setXRange(0, frame_range + 1, padding = 0.0)
        self.p1.setYRange(0, 180, padding = 0.0) # 設定角度的預設範圍
        self.p1.showGrid(x=True, y=False)
        
        # 加入圖例 (Legend)
        # self.legend = self.p1.addLegend(offset=(10, 10))

        # --- 2. 建立第二個 ViewBox (Right Axis - Speed) ---
        self.v2 = pg.ViewBox()
        self.p1.scene().addItem(self.v2)
        self.p1.getAxis('right').linkToView(self.v2) # 將右軸連接到 v2
        self.v2.setXLink(self.p1) # X軸連動
        
        # 設定右軸標籤
        speed_label_html = '<span style="color:#000000;">腳踝速度</span> <span style="color:#00aa00;">\手腕速度</span>'
        self.p1.getAxis('right').setLabel(speed_label_html)
        self.p1.showAxis('right')
        self.v2.setYRange(0, 40, padding=0.0) # 設定速度的預設範圍

        # --- 3. 處理視窗縮放同步 (關鍵步驟) ---
        # 當 p1 大小改變時，必須同步調整 v2 的大小
        def updateViews():
            self.v2.setGeometry(self.p1.vb.sceneBoundingRect())
            self.v2.linkedViewChanged(self.p1.vb, self.v2.XAxis)

        self.p1.vb.sigResized.connect(updateViews)
        updateViews() # 初始化執行一次

        # --- 4. 初始化所有曲線 ---
        for i, name in enumerate(self.speed_names):
            title = self.title_names[i]
            color = self.curve_colors[i % len(self.curve_colors)]
            
            # 判斷是角度還是速度 (簡單的字串判斷)
            is_angle = "角度" in title
            
            # 根據類型選擇畫布 (p1 或 v2)
            target_plot = self.p1 if is_angle else self.v2
            
            # 建立曲線物件 (先給空數據)
            # 注意: 如果是在 v2 (右軸) 畫圖，需要手動加到 legend，因為 addLegend 預設只抓 p1
            curve = pg.PlotCurveItem(pen=pg.mkPen(color=color, width=2), name=title)
            target_plot.addItem(curve)
            
            # 建立散點物件 (用於顯示當前幀位置)
            scatter = pg.ScatterPlotItem(size=8, brush=color)
            target_plot.addItem(scatter)
            
            # if not is_angle:
            #     # 手動將 v2 的曲線加入圖例 (pyqtgraph 的小眉角)
            #     self.legend.addItem(curve, title)

            # 存起來供 updateGraph 使用
            self.curves[name] = {'item': curve, 'is_angle': is_angle, 'title': title}
            self.scatters[name] = {'item': scatter}

    def updateGraph(self, frame_num: int):
        """
        更新所有曲線數據與圖例數值
        使用 setData 取代 clear+plot，效能更好且不會閃爍
        """
        for name in self.speed_names:
            if name not in self.curves:
                continue

            # 1. 獲取數據
            times, values = self.pose_analyzer.get_frame_angleOrSpeed_data(speed_name=name)
            try:
                times = np.array(times, dtype=float).flatten()
                values = np.array(values, dtype=float).flatten()
            except Exception as e:
                print(f"數據轉換錯誤 ({name}): {e}")
                times, values = [], []

            _, current_val = self.pose_analyzer.get_frame_angleOrSpeed_data(frame_num=frame_num, speed_name=name)
            try:
                current_val = float(current_val)
            except (TypeError, ValueError):
                current_val = 0.0

            # 2. 更新曲線數據
            self.curves[name]['item'].setData(times, values)
            
            # 3. 更新當前幀的散點
            self.scatters[name]['item'].setData([frame_num], [current_val])

            # 4. 更新圖例名稱顯示數值 (因為只有一個標題欄，改用圖例顯示數值更清楚)
            base_title = self.curves[name]['title']
            new_name = f"{base_title}: {current_val:.2f}"
            pass

        # 選項：在主標題顯示當前幀數
        self.p1.setTitle(f"Frame: {frame_num}")

    def setPlotVisibility(self, speed_name: str, visible: bool):
        """控制特定曲線顯示/隱藏"""
        if speed_name in self.curves:
            self.curves[speed_name]['item'].setVisible(visible)
            self.scatters[speed_name]['item'].setVisible(visible)

    def setMotionPhases(self, phases_data: list):
        """
        接收動作階段數據 (例如: [(start, end, color, label)])，並在圖表上繪製分段區域。
        此函式應在影片分析完成後呼叫。
        """
        # 儲存新的分段數據
        self.motion_phases = phases_data
        
        # 迭代數據並繪製分段
        for start, end, color in self.motion_phases:
            shaded_color = QColor(color)
            shaded_color.setAlpha(100) # 255 * 0.5 ≈ 128
            # 創建垂直區域
            region = pg.LinearRegionItem(
                values=[start, end], 
                orientation='vertical',
                brush=pg.mkBrush(shaded_color), # 顏色 + 50% 透明度
                movable=False
            )
            region.setZValue(-100) # 確保陰影在曲線後面
            
            self.p1.addItem(region)
            self.regions.append(region)
            
        # 確保圖表更新
        self.p1.update()

    def resize_graph(self, width, height):
        self.graph.resize(width, height)

    def reset(self):
        self.graph.clear()
        self.regions = [] # 清除引用
        self.curves = {} 
        self.scatters = {}