# 開發者筆記

給要重建開發環境、除錯、或接手維護這個專案的人看。使用者安裝步驟請見 [README.md](../README.md)；AI 助手的協作規範與問題追蹤表請見 [TODO.md](TODO.md)。這份文件記錄「為什麼」——每個已知問題的成因與排查過程，方便日後遇到類似狀況時參考。

最後更新：2026-09-21

---

## 目前狀況一覽（2026-09-21）

**可以正常使用的功能**
- 環境已建置完成（conda 環境 `Pitcher`），程式可正常啟動
- 雙相機即時擷取（需接 GPIO 同步線）、自動投球偵測、自動錄影
- 2D 分頁「雙影片模式」回放分析（同時選 CF + CS 兩支影片）

**目前不能用 / 有風險的功能**
- 2D 分頁「單影片模式」：跑完分析會跳錯誤視窗，無法完成（問題編號 P1-002）
- 2D 分頁單獨選「正面 CF」影片：系統會拿正面影片去做側面專用的分析，**算出來的角度數據沒有意義，但畫面不會有任何警告**（問題編號 P1-001）
- 3D 分頁的 3D 重建：使用的相機校正參數有兩套互相矛盾的版本，精度存疑（問題編號 P2-001）

**2D 分頁該怎麼操作（暫行建議）**
1. 執行 `cd Src\UI_Control` 然後 `python main.py`
2. 切到「2D」分頁，右側選「回放」頁籤
3. 在影片樹狀清單中，**同時選取同一次投球的 CF + CS 兩支影片**（按住 Ctrl 點選），這是目前唯一能正常完成分析的方式
4. 勾選「檢視骨架影片」開始分析
5. 分析完成後，下方「前導腳著地 / 肩最大外旋 / 球離手」三個按鈕會變成可點擊，按下可跳到對應畫面
6. 播放控制：空白鍵播放/暫停、A 上一幀、D 下一幀

> 為什麼一定要選兩支？因為系統的生物力學分析（肩髖分離角、肩外旋角等）設計上只從**側面**視角計算。選兩支時系統會自動用 CS（側面）那支做分析；只選一支時系統會把它同時當成正面和側面用，若選到 CF 就會分析錯對象。

**接下來的工作順序**
1. 修好 2D 分頁的兩個問題（P1-001、P1-002）
2. 比對新舊兩版骨架偵測程式（`detect_skeleton.py` vs `detect_skeleton_new.py`），決定要怎麼收拾這個未完成的重構
3. 相機內參重新校正（因應 9/17 場勘後的新 8 台相機架構）
4. GB10 主機的相機連接測試（等實體機器到手）

---

## 環境建置已知問題

### SSL 憑證驗證失敗（`CERTIFICATE_VERIFY_FAILED`）

**症狀**：`pip install` / `conda install` 全部失敗，錯誤訊息包含 `unable to get local issuer certificate`。

**成因**：部分防毒軟體（例如 Avast）的「網頁守衛 / HTTPS 掃描」功能會攔截所有 HTTPS 連線，用自己生成的根憑證重新簽發，藉此掃描流量內容。這張憑證會被加進 Windows 系統信任庫（瀏覽器、`curl` 因此不受影響），但 Python 生態系（`pip`、`conda`）預設只信任 `certifi` 套件內建、與系統信任庫完全獨立的公開憑證清單，不認得防毒軟體臨時生成的憑證，因而拒絕連線。

**解法**：到防毒軟體設定裡關閉 HTTPS/SSL 掃描（Avast 路徑：主畫面齒輪圖示 →「設定」→「隱私」→「網頁守衛」→ 關閉「啟用 HTTPS 掃描」）。這是最徹底的解法，關閉後所有 Python 工具鏈都會恢復正常，不需要額外的憑證合併或環境變數設定。

### `libiomp5md.dll` 不可刪除

**症狀**：`ImportError` / `OSError: Error loading "...\nvfuser_codegen.dll"`，torch 完全無法 import。

**成因**：舊版 README 曾指示刪除 `anaconda3\envs\<env>\Lib\site-packages\torch\lib\libiomp5md.dll` 以消除一則 OpenMP 重複載入的 warning。這個指示可能是針對更早期的 torch 版本寫的——當時 torch 內部沒有用到 nvfuser，刪除該檔案只會消除警告。但目前使用的 torch 2.0.1 版本中，`nvfuser_codegen.dll` 依賴這個檔案，刪除後會導致 torch 整個 import 失敗。

**解法**：不要刪除這個檔案。若已誤刪，執行 `pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118 --force-reinstall --no-deps` 重新安裝即可恢復。

### `opencv-python-headless` 覆蓋 GUI 版 opencv

**症狀**：`cv2.imshow`、`cv2.namedWindow` 等視窗顯示功能消失或報錯，但 `import cv2` 本身不會出錯。

**成因**：`mmpretrain` 的可選相依套件 `albumentations`（資料增強函式庫）依賴 `opencv-python-headless`（無 GUI 功能的精簡版），pip 安裝時會用它覆蓋掉先前裝好的 `opencv-python`（完整版）。這套系統的 GUI（`cv_utils/cv_thread.py`、`pitch_widget.py` 等）需要完整版才能正常運作，且核心功能不依賴 `albumentations`。

**解法**：
```
pip uninstall -y opencv-python-headless
pip install --force-reinstall --no-deps opencv-python==4.8.1.78
```

### vispy / PySpin DLL 衝突

**症狀**：`RuntimeError: Could not import backend "PyQt5": DLL load failed while importing QtOpenGL`。

**成因**：若在 `import PySpin`（FLIR 相機 SDK）之後才呼叫 `vispy.use('pyqt5')` 鎖定 Qt 後端，PySpin 附帶的第三方 DLL 會污染 Windows 的 DLL 搜尋路徑，導致 Qt 之後載入 `QtOpenGL` 時抓到不相容的版本。已透過二分法排查確認：問題不是 PySpin 本身，而是「PyQt5 已載入 + PySpin 載入」這個組合疊加時才會觸發。

**解法**：`vispy.use('pyqt5')` 必須在檔案最上方、任何會匯入 PySpin 的模組之前執行。目前 `main.py` 已依此順序修正。若新增其他程式進入點（entry point）也需要遵循相同順序。

### 執行路徑錯誤導致 `FileNotFoundError`

**症狀**：`FileNotFoundError: [Errno 2] No such file or directory: '..\mmyolo_main\configs\...'`

**成因**：`utils\model.py` 中的模型設定檔路徑使用相對路徑（如 `..\mmyolo_main\configs\...`），這些路徑假設程式是從 `Src\UI_Control` 目錄本身執行的。若從專案根目錄執行 `python Src\UI_Control\main.py`，相對路徑解析基準不同，會找不到檔案。

**解法**：務必先 `cd Src\UI_Control` 再執行 `python main.py`。

### FLIR SDK（PySpin）為隱性必要相依套件

`camera_objects/__init__.py` 設計上會在沒有 PySpin 時優雅跳過 FLIR 相關類別的匯出，但 `cv_utils/cv_thread.py` 目前無條件匯入 `FlirCameraSystem`，因此實務上 PySpin 是必要相依套件，即使不接 FLIR 相機也需要安裝，否則整個 GUI 無法啟動。

若沒有官方安裝檔，且電腦上其他 conda 環境已裝過 PySpin，可直接複製該環境的 `Lib\site-packages\PySpin\` 與對應的 `spinnaker_python-*.dist-info` 資料夾過來，前提是兩邊 Python 版本一致。

## 相機硬體相關

### 相機序號寫死在程式碼中

`Src\UI_Control\cv_utils\cv_thread.py` 第 66-71 行：
```python
SN1 = "25462483"
SN2 = "25462481"
self.camera1 = FlirCameraSystem(CONFIG, SN1)
self.camera2 = FlirCameraSystem(CONFIG_2, SN2)
```
接上不同的實體 FLIR 相機時，需要直接修改此處的序號常數，目前沒有設定檔或環境變數的動態帶入機制。

### `camera_widget.py` 的 `model` 參數已移除

`PoseCameraTabControl.__init__` 目前簽名為 `def __init__(self, parent=None)`，不再接受 `model` 參數（第 17-18 行殘留了被註解掉的 `self.model = model` 與 `self.init_pose_estimater()`，是重構未完成的痕跡）。目前相機分頁純粹負責畫面預覽與錄影，不涉及即時骨架推論，呼叫端（`main.py`）需對應使用 `PoseCameraTabControl()`（不帶參數）。

### 相機規格參考（`camera_config/*.yaml`）

- 解析度 1920×1084，179fps，`BayerRG8` 原始格式（需程式端做色彩還原，對應 UI 上的曝光/紅色/藍色調色滑桿）
- 透過 GPIO 硬體訊號線做多相機同步觸發：`camera1`（正面）在 `DualFlirSystem.__init__` 中被設為 Primary（`configure_gpio_primary()`，輸出 `ExposureActive` 訊號），`camera2`（側面）被設為 Secondary（`configure_gpio_secondary()`，由 `Line3` 接收觸發），非純軟體時間戳記同步
- **已實測驗證**：僅接 USB 資料線、未接 GPIO 同步線時，Secondary 相機會因等不到觸發訊號而逾時（`Failed waiting for EventData on NEW_BUFFER_DATA event`）；接上同步線後雙機可正常同步擷取畫面
- 目前程式僅支援兩台相機（正面 + 側面各一），無第二支側面相機（論文原始架構為 2 側面 + 1 正面）的處理邏輯

## 背景：系統設計依據

此系統的演算法設計對應學長姐碩士論文：《基於2D-to-3D Diffusion-Transformer網路之棒球投球骨架動作切分及動作分析》（施邑穎，國立成功大學人工智慧科技碩士學位學程，指導教授連震杰）。

- 原始硬體架設：3 部高速工業相機（2 側面 + 1 正面）+ Rapsodo Pro 2.0 投球追蹤系統（獨立設備，量測結果未納入本系統演算法）
- 核心流程：雙攝影機校正（Zhang's method）→ YOLOv11+ViTPose 2D 骨架偵測 → 三角測量重建 3D 骨架 → 投球四階段切分（Foot Contact / Maximum External Rotation / Ball Release）→ 生物力學參數計算（肩髖分離角、肩外旋角、手腕速度等）
- 系統架構採「即時擷取（Live）」與「離線分析（Replay）」分離設計，避免即時運算拖慢影像擷取效能

### 鏡頭規格（論文表 22）

| 位置 | 型號 | 焦距 | 光圈 | 用途 |
|---|---|---|---|---|
| c1, c2（側面 ×2） | KOWA LM8HC | 8mm | F/1.8 | 視角較大，完整涵蓋投手從抬腿到跟隨動作的整個過程 |
| c3（正面） | KOWA LM50HC | 50mm | F/1.4 | 較高人體解析度，提升關節辨識品質 |

> 曾誤判正面鏡頭為 75mm——那是設備交接清單裡的庫存/歷史紀錄，非論文正式採用規格。查證鏡頭焦距等硬體規格時，以論文本文為準，交接清單僅供財產追蹤參考。

### GUI 分頁與即時操作的實際對應關係

依交接文件《交接程式操作文件_Ui》第 10 節：**`Src/UI_Control/camera_widget.py`（「2D 相機」分頁）目前入口未啟用**，是獨立於主流程之外的備用實作。真正的雙攝影機即時操作、自動投球偵測、自動錄影，是整合在 **「2D」分頁**（`PosePitchTabControl`，即 `pitch_widget.py`）裡的 `cameraCheckBox`（開啟相機）/`startPitchCheckBox`（開始投球）等控制項。

雙視角畫面的左右對應（追蹤自 `cv_thread.py` → `DualFlirSystem.get_grayscale_images()` → `pitch_widget.py updateFrame()`，並經論文原文「介面左上方提供正面攝影機與側面攝影機之同步播放畫面」交叉驗證）：

| 畫面位置 | UI 元件 | 對應相機 | 序號 |
|---|---|---|---|
| 左邊 | `FrameView` | 正面攝影機（camera1） | `25462483` |
| 右邊 | `FrameView_2` | 側面攝影機（camera2） | `25462481` |

## Git 歷史說明

本專案原始 git 歷史（47 個 commit，2024-07-30 ～ 2024-10-12）為前一位貢獻者所建立，commit 訊息多為概略性的 `fix bug`。約 2024-10 之後至此份程式碼被複製交接為止的開發進度未曾提交版本控制，是透過 USB 直接複製取得，因此這段時間的變更沒有逐次的 commit 紀錄可供追溯。
