# PosePitchTabControl 系統類別架構說明

## 系統概述
`PosePitchTabControl` 是一個用於投球動作分析的 PyQt5 應用程式，整合了影像處理、骨架偵測、動作分析等功能。

---

## 核心類別架構

### 1. **主控制類別**

#### `PosePitchTabControl` (QWidget)
- **檔案位置**: [pitch_widget_copy_v2.py](pitch_widget_copy_v2.py)
- **功能**: 主要控制介面，整合所有子系統
- **主要職責**:
  - UI 控制與事件綁定
  - 攝影機/影片播放控制
  - 協調各子系統運作
  - 投球動作自動偵測與錄製

---

## 子系統類別

### 2. **影像擷取與處理模組** (cv_utils/)

#### `Camera`
- **檔案位置**: [cv_utils/cv_control.py](cv_utils/cv_control.py)
- **功能**: 雙攝影機控制
- **主要屬性**:
  - `frame_buffer: queue.Queue` - 攝影機1幀緩衝區
  - `frame_buffer_2: queue.Queue` - 攝影機2幀緩衝區
  - `video_thread: VideoCaptureThread` - 影像擷取執行緒
  - `video_writer: VideoWriterThread` - 影片錄製執行緒
- **主要方法**:
  - `open_camera()` - 開啟雙攝影機
  - `close_camera()` - 關閉攝影機
  - `startRecording(filename, filename_2)` - 開始錄製
  - `stop_recording()` - 停止錄製
  - `buffer_frame(frame, frame_2)` - 接收並緩衝影像幀

#### `VideoLoader`
- **檔案位置**: [cv_utils/cv_control.py](cv_utils/cv_control.py)
- **功能**: 影片載入與管理
- **主要屬性**:
  - `video_frames: List` - 影片1所有幀資料
  - `video_frames_2: List` - 影片2所有幀資料
  - `total_frames: int` - 總幀數
  - `image_drawer: ImageDrawer` - 影像繪製器1
  - `image_drawer_2: ImageDrawer` - 影像繪製器2
- **主要方法**:
  - `loadVideo(path1, path2)` - 載入雙影片
  - `getVideoImage(frame_num)` - 取得指定幀
  - `saveVideo(output_dir)` - 儲存處理後影片
  - `reset()` - 重置載入狀態

---

### 3. **骨架偵測模組** (skeleton/)

#### `PoseEstimater`
- **檔案位置**: [skeleton/detect_skeleton.py](skeleton/detect_skeleton.py)
- **功能**: 人體骨架偵測與追蹤
- **依賴**: `Model` (提供 YOLO 偵測器、MMPose 姿態估測器、BoTSORT 追蹤器)
- **主要屬性**:
  - `model: Model` - 深度學習模型集合
  - `person_df: pd.DataFrame` - 所有幀的人物資料
  - `person_id: int` - 選中的人物ID
  - `kpt_id: int` - 選中的關鍵點ID
  - `kpt_buffer: List` - 關鍵點軌跡緩衝
  - `joints: dict` - 關節定義 (COCO/HAPLE格式)
- **主要方法**:
  - `detectKpt(frame, frame_num, is_video)` - 偵測骨架關鍵點
  - `getPersonDf(frame_num, is_select, is_kpt)` - 取得人物資料
  - `setPersonId(id)` - 設定追蹤目標
  - `setKptId(id)` - 設定追蹤關鍵點
  - `reset()` - 重置偵測狀態

---

### 4. **動作分析模組** (utils/)

#### `PoseAnalyzer`
- **檔案位置**: [utils/analyze.py](utils/analyze.py)
- **功能**: 骨架角度分析
- **依賴**: `PoseEstimater`
- **主要屬性**:
  - `pose_estimater: PoseEstimater` - 骨架偵測器
  - `angle_dict: dict` - 角度計算定義
  - `analyze_df: pd.DataFrame` - 分析結果資料框
  - `processed_frames: set` - 已處理幀集合
- **主要方法**:
  - `addAnalyzeInfo(frame_num)` - 新增該幀分析資料
  - `_calculate_angle(A, B, C)` - 計算三點角度
  - `get_frame_angle_data(frame_num, angle_name)` - 取得角度資料
  - `reset()` - 重置分析資料

#### `ImageDrawer`
- **檔案位置**: [utils/vis_image.py](utils/vis_image.py)
- **功能**: 影像視覺化繪製
- **依賴**: `PoseEstimater`, `PoseAnalyzer`
- **主要屬性**:
  - `pose_estimater: PoseEstimater` - 骨架偵測器
  - `pose_analyzer: PoseAnalyzer` - 動作分析器
  - `show_skeleton: bool` - 顯示骨架
  - `show_bbox: bool` - 顯示邊界框
  - `show_grid: bool` - 顯示網格
  - `show_traj: bool` - 顯示軌跡
  - `show_countdown: bool` - 顯示倒數計時
- **主要方法**:
  - `drawInfo(img, frame_num, kpt_buffer, countdown_time)` - 繪製所有資訊
  - `drawPointsandSkeleton(img, person_df, skeleton_links)` - 繪製骨架
  - `drawBbox(img, person_df)` - 繪製邊界框
  - `drawTraj(img, kpt_buffer)` - 繪製軌跡
  - `drawCountdown(img, countdown_time)` - 繪製倒數
  - `reset()` - 重置繪製狀態

---

### 5. **選擇器模組** (utils/)

#### `PersonSelector`
- **檔案位置**: [utils/selector.py](utils/selector.py)
- **功能**: 選擇目標人物
- **主要屬性**:
  - `selected_id: int` - 選中的人物ID
- **主要方法**:
  - `select(x, y, search_person_df)` - 根據座標或最大bbox選人
  - `reset()` - 重置選擇

#### `KptSelector`
- **檔案位置**: [utils/selector.py](utils/selector.py)
- **功能**: 選擇目標關鍵點
- **主要屬性**:
  - `selected_id: int` - 選中的關鍵點ID
- **主要方法**:
  - `select(x, y, search_person_df)` - 根據座標選擇最近關鍵點
  - `reset()` - 重置選擇

---

### 6. **模型管理模組** (utils/)

#### `Model`
- **檔案位置**: [utils/model.py](utils/model.py)
- **功能**: 深度學習模型初始化與管理
- **主要屬性**:
  - `detector` - MMYOLO 人物偵測器 (YOLOv8)
  - `pose_estimator` - MMPose 姿態估測器 (ViTPose)
  - `tracker` - BoTSORT 多目標追蹤器
  - `image_size: tuple` - 影像尺寸
- **主要方法**:
  - `setDetectParser()` - 設定偵測器參數
  - `setPoseParser()` - 設定姿態估測參數
  - `setTrackerParser()` - 設定追蹤器參數
  - `setImageSize(size)` - 設定影像尺寸

---

### 7. **計時器模組** (utils/)

#### `Timer`
- **檔案位置**: [utils/timer.py](utils/timer.py)
- **功能**: 倒數計時器
- **主要屬性**:
  - `duration: int` - 總時長(秒)
  - `start_time: float` - 開始時間
- **主要方法**:
  - `start()` - 開始計時
  - `is_time_up()` - 檢查時間是否到
  - `get_remaining_time()` - 取得剩餘時間
  - `reset()` - 重置計時器

#### `FPSTimer`
- **檔案位置**: [utils/timer.py](utils/timer.py)
- **功能**: FPS 計算器
- **主要方法**:
  - `tic()` - 開始計時
  - `toc(average)` - 結束計時並返回FPS
  - `clear()` - 清除記錄

---

## 類別依賴關係圖

```
PosePitchTabControl (主控制器)
├── Model (深度學習模型)
│   ├── YOLO Detector (YOLOv8)
│   ├── Pose Estimator (ViTPose)
│   └── Tracker (BoTSORT)
│
├── Camera (雙攝影機控制)
│   ├── VideoCaptureThread (影像擷取)
│   └── VideoWriterThread (影片錄製)
│
├── VideoLoader (影片管理)
│   ├── ImageDrawer (視覺化繪製1)
│   └── ImageDrawer (視覺化繪製2)
│
├── PoseEstimater x2 (骨架偵測)
│   ├── Model (共用模型)
│   ├── FPSTimer (FPS計算)
│   └── OneEuroFilter (平滑濾波)
│
├── PoseAnalyzer x2 (動作分析)
│   └── PoseEstimater (骨架資料來源)
│
├── ImageDrawer x2 (視覺化繪製)
│   ├── PoseEstimater (骨架資料)
│   └── PoseAnalyzer (分析資料)
│
├── PersonSelector x2 (人物選擇)
├── KptSelector (關鍵點選擇)
│
└── Timer (倒數計時器)
```

---

## 資料流程

### 即時攝影機模式
```
Camera → VideoCaptureThread → frame_buffer/frame_buffer_2
                                      ↓
                            PoseEstimater.detectKpt()
                                      ↓
                            PoseAnalyzer.addAnalyzeInfo()
                                      ↓
                            ImageDrawer.drawInfo()
                                      ↓
                            QGraphicsView 顯示
```

### 影片播放模式
```
VideoLoader.loadVideo() → 載入所有幀到記憶體
                               ↓
frame_num → VideoLoader.getVideoImage()
                               ↓
           PoseEstimater.detectKpt(frame, frame_num)
                               ↓
           PoseAnalyzer.addAnalyzeInfo(frame_num)
                               ↓
           ImageDrawer.drawInfo(frame, frame_num)
                               ↓
           QGraphicsView 顯示
```

### 自動投球偵測流程
```
攝影機模式 → 關節穩定偵測 (pitherAnaylze)
                  ↓
        檢測關節穩定幀數 >= 15
                  ↓
        檢測腳踝移動 > 閾值
                  ↓
        啟動倒數計時器 (3秒)
                  ↓
        自動開始錄製
                  ↓
        計時結束後自動播放錄製影片
```

---

## UI 事件綁定

### CheckBox 控制
- **cameraCheckBox** → `toggleCamera()` - 開啟/關閉雙攝影機
- **recordCheckBox** → `toggleRecord()` - 開始/停止錄製
- **selectCheckBox** → `toggleSelect()` - 選擇追蹤目標
- **showSkeletonCheckBox** → `toggleShowSkeleton()` - 顯示骨架
- **selectKptCheckBox** → `toggleKptSelect()` - 選擇關鍵點並顯示軌跡
- **showBboxCheckBox** → `toggleShowBbox()` - 顯示邊界框
- **showLineCheckBox** → `toggleShowGrid()` - 顯示網格線
- **startPitchCheckBox** → `togglePitching()` - 啟動自動投球偵測
- **skeletonVideoCheckBox** → `toggleSkeletonVideo()` - 處理並儲存骨架影片

### Slider/Input 控制
- **frameSlider** → `analyzeFrame()` - 更新影片幀
- **redRatioSlider** → `onRedRatioChanged()` - 調整紅色白平衡
- **blueRatioSlider** → `onBlueRatioChanged()` - 調整藍色白平衡
- **gainSlider** → `onGainChanged()` - 調整增益
- **cameraIdInput** → `changeCamera()` - 切換攝影機ID
- **pitchInput** → `changePitcher()` - 切換左/右投

### Button 控制
- **playBtn** → `playBtnClicked()` - 播放/暫停影片
- **backKeyBtn** → 影片往前一幀
- **forwardKeyBtn** → 影片往後一幀

### Tree 控制
- **videoTree** → `play_video_from_item()` - 選擇並播放兩部影片

---

## 關鍵特性

### 1. 雙視角同步
- 同時處理兩個攝影機/影片
- 每個視角獨立的 `PoseEstimater`、`PoseAnalyzer`、`ImageDrawer`
- 同步顯示與分析

### 2. 自動投球偵測
- 透過關節穩定性判斷投球預備姿勢
- 腳踝移動觸發自動錄製
- 倒數計時視覺化提示

### 3. 模型共用
- 多個 `PoseEstimater` 共用同一個 `Model` 實例
- 降低GPU記憶體使用
- 提高效能

### 4. 即時與離線雙模式
- 即時攝影機模式：低延遲顯示
- 離線影片模式：完整分析與重播

---

## 主要工作流程

### 系統初始化
1. 建立 `Model` (載入 YOLO、MMPose、Tracker)
2. 初始化 `Camera`、`VideoLoader`
3. 建立雙份 `PoseEstimater`、`PoseAnalyzer`、`ImageDrawer`
4. 綁定 UI 事件
5. 載入影片列表到 Tree

### 即時攝影機工作流
1. 使用者勾選 `cameraCheckBox`
2. `Camera.open_camera()` 啟動 `VideoCaptureThread`
3. `timer` 每 1ms 觸發 `analyzeFrame()`
4. 從 `frame_buffer` 取得最新幀
5. `PoseEstimater.detectKpt()` 偵測骨架
6. 若啟用投球模式，執行 `pitherAnaylze()` 檢測
7. `ImageDrawer.drawInfo()` 繪製視覺化
8. 更新 `QGraphicsView` 顯示

### 影片分析工作流
1. 從 Tree 選擇兩部影片
2. `VideoLoader.loadVideo()` 載入所有幀
3. 使用者拖動 `frameSlider` 或點擊播放
4. `analyzeFrame()` 取得指定幀
5. 若勾選 `skeletonVideoCheckBox`:
   - 對每一幀執行骨架偵測
   - 累積分析資料到 `PoseAnalyzer`
   - 播放結束後儲存骨架影片
6. 更新顯示

---

## 檔案結構總覽

```
Src/UI_Control/
├── pitch_widget_copy_v2.py          # 主控制器
├── pitch_ui_2tabs_beta_sync_v3.py   # UI 定義 (Qt Designer生成)
├── playVideo_2.py                   # 全螢幕播放對話框
│
├── cv_utils/
│   ├── cv_control.py                # Camera, VideoLoader
│   └── cv_thread.py                 # 影像執行緒
│
├── skeleton/
│   └── detect_skeleton.py           # PoseEstimater
│
├── utils/
│   ├── model.py                     # Model (深度學習模型管理)
│   ├── analyze.py                   # PoseAnalyzer
│   ├── vis_image.py                 # ImageDrawer
│   ├── selector.py                  # PersonSelector, KptSelector
│   ├── timer.py                     # Timer, FPSTimer
│   ├── one_euro_filter.py           # OneEuroFilter (平滑濾波)
│   └── vis_graph.py                 # GraphPlotter (圖表繪製)
│
└── ui_utils/
    ├── table_control.py             # KeypointTable
    └── graphicview_control.py       # FrameView
```

---

## 依賴套件

### 深度學習框架
- **PyTorch** - 深度學習基礎
- **MMDetection** - 物件偵測 (YOLO)
- **MMPose** - 姿態估測 (ViTPose)
- **MMEngine** - 訓練/推論引擎

### 影像處理
- **OpenCV (cv2)** - 影像處理
- **NumPy** - 數值計算
- **Pillow (PIL)** - 影像繪製 (支援中文字型)
- **PySpin** - FLIR 工業相機SDK

### UI 框架
- **PyQt5** - GUI 框架
- **PyQtGraph** - 圖表繪製

### 資料處理
- **Pandas** - 資料框架
- **SciPy** - 訊號平滑 (Savgol filter)

### 其他
- **BoTSORT** - 多目標追蹤演算法

---

## 效能考量

### 記憶體管理
- 影片模式：將所有幀載入記憶體 (高記憶體使用)
- 攝影機模式：使用 Queue 緩衝區 (低記憶體)
- 模型共用：避免重複載入權重

### 處理速度
- GPU 加速：YOLO、MMPose 使用 CUDA
- FPS 控制：攝影機模式可設定跳幀
- 非同步錄製：`VideoWriterThread` 獨立執行緒

### 資料平滑
- `OneEuroFilter` 平滑關鍵點軌跡
- `savgol_filter` 平滑角度曲線

---

## 未來擴展建議

### 功能增強
1. 支援更多關節角度分析
2. 3D 姿態重建 (使用雙視角)
3. 投球速度估算
4. 投球動作評分系統

### 效能優化
1. 影片懶載入 (按需載入幀)
2. 多執行緒骨架偵測
3. 模型量化加速

### 使用者體驗
1. 拖放式影片載入
2. 可自訂角度定義
3. 匯出分析報告 (PDF/Excel)

---

## 版本資訊
- **系統名稱**: Pitch Skeleton User Interface
- **版本**: v7
- **最後更新**: 2026-01-12

---

## 授權與貢獻
本系統整合多個開源專案：
- **MMDetection** - Apache 2.0 License
- **MMPose** - Apache 2.0 License
- **BoTSORT** - MIT License
