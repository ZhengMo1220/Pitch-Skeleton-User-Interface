# Pitch-Skeleton-User-Interface

## Usage
### Installation

> **開始前**：若你的電腦有防毒軟體開啟 HTTPS 流量掃描（例如 Avast 的「網頁守衛 / HTTPS 掃描」），pip/conda 會因為憑證驗證失敗而無法下載套件（`CERTIFICATE_VERIFY_FAILED`）。請先到防毒軟體設定裡關閉 HTTPS 掃描，否則以下所有步驟都可能失敗。

1. Clone this repo.
2. Setup conda environment:
    ```
    conda create -n Pitcher python=3.8 -y
    conda activate Pitcher
    pip install -r requirements.txt
    # CUDA 11.8
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    conda install -c conda-forge faiss-gpu
    pip install cython_bbox
    ```
    > ⚠️ **不要**刪除 `anaconda3\envs\<your_env_name>\Lib\site-packages\torch\lib\libiomp5md.dll`。這個檔案是 torch 2.0.1 正常運作所需要的依賴，刪除後會導致 `OSError: Error loading "...\nvfuser_codegen.dll"`，torch 完全無法 import。舊版 README 曾建議刪除以消除 warning，但在目前的 torch 版本下會直接讓環境壞掉。

3. setup openmim
    ``` 
    pip install -U openmim
    mim install mmcv-full
    mim install "mmcv==2.0.1"
    mim install "mmdet==3.1.0"
    ``` 
5. Setup mmpose environment:
    ```
    cd Src\mmpose_main
    pip install -r requirements.txt
    pip install -v -e .
    cd ..\..\
    ```
6. Setup mmyolo environment:
    ```
    cd Src\mmyolo_main
    pip install -r requirements.txt
    pip install -v -e .
    cd ..\..\
    ```
7. Setup mmengine_main environment:
    ```
    mim install mmengine
    ```
    > 若上一步 `mmcv==2.0.1` 已安裝成功，mmengine 通常已隨其相依套件一併裝好，可用 `python -c "import mmengine; print(mmengine.__version__)"` 確認後決定是否需要重跑此步驟。

8. Setup mmpretrain_main environment:
    ```
    cd Src\mmpretrain_main
    pip install -r requirements.txt
    pip install -v -e .
    cd ..\..\
    ```
    > ⚠️ 這一步會連帶安裝 `albumentations`，其相依套件 `opencv-python-headless` 會覆蓋掉前面裝好的 `opencv-python`（含 GUI 功能的完整版），導致 `cv2.imshow` 等視窗顯示功能失效。這套系統的 GUI（`cv_utils/cv_thread.py`、`pitch_widget.py` 等）需要完整版 opencv，請在此步驟之後執行：
    ```
    pip uninstall -y opencv-python-headless
    pip install --force-reinstall --no-deps opencv-python==4.8.1.78
    ```

9. Setup vispy（3D 骨架視覺化所需，原版 requirements 未列出）:
    ```
    pip install vispy
    ```

10. Setup FLIR camera SDK（僅在需要接實體 FLIR 相機時需要）：
    - 至 [FLIR/Teledyne 官網](https://www.teledynevisionsolutions.com) 登入下載 **Spinnaker SDK**（對應 Windows、Python 3.8、系統位元數），安裝完成後 `PySpin` 套件會位於 Python site-packages 中
    - 若沒有官方安裝檔，可從另一台已安裝過的環境直接複製 `Lib\site-packages\PySpin\` 資料夾與對應的 `spinnaker_python-*.dist-info` 資料夾到 `Pitcher` 環境，前提是兩邊 Python 版本一致（3.8）
    - 沒有安裝 PySpin 時，`camera_objects` 模組會自動跳過 FLIR 相關類別，但 `cv_utils/cv_thread.py` 目前無條件匯入 `FlirCameraSystem`，仍會拋出 `ImportError`，因此 PySpin 實質上是必要相依套件

### 已知環境問題與修正紀錄

- **SSL 憑證問題**：見上方「開始前」提示。
- **`libiomp5md.dll` 不可刪除**：見步驟 2 提示。
- **opencv 版本被覆蓋**：見步驟 8 提示。
- **vispy/PySpin DLL 衝突**：若在 `import PySpin` 之後才呼叫 `vispy.use('pyqt5')`，會出現 `RuntimeError: Could not import backend "PyQt5": DLL load failed while importing QtOpenGL`。`main.py` 目前已將 `vispy.use('pyqt5')` 移到檔案最上方、任何會匯入 PySpin 的模組之前執行，修正此問題。若新增其他進入點（entry point）也需要遵循相同順序。

### Data Preparation
To obtain the vitpose、yolo and fast-reid wights, it can be downloaded from the https://drive.google.com/drive/folders/1D7Q5bTnTAfKkfLuppqUo4_8W4t0wrCmP?usp=sharing. The resulting data directory should look like this:
    
    ${POSE_ROOT}
    |-- Db
    -- |-- pretrain
            |-- vitpose_Sk26.pth
            |-- yolov7_x_syncbn_fast_8x16b-300e_coco_20221124_215331-ef949a68.pth
    -- |-- Record (for output data)
            |-- {video_name}.mp4 (原始影片)
            |-- {video_name}_Sk26.mp4 (將原始影片畫上骨架資訊)
            |-- {video_name}.json (將偵測出來的結果紀錄，裡面包含了人物的bounding box info. 和 26 個關節點位置)
            
    |-- Src

### Demo
1. Demo command

    > ⚠️ 必須先切換到 `Src\UI_Control` 目錄再執行，不能從專案根目錄直接執行 `python Src\UI_Control\main.py`。程式內部（例如 `utils\model.py`）使用相對路徑（如 `..\mmyolo_main\configs\...`）尋找設定檔，這些路徑是以 `Src\UI_Control` 為基準解析的，從錯誤的工作目錄執行會導致 `FileNotFoundError`。

    ```
    cd Src\UI_Control
    python main.py
    ```
2. 2D 相機
    ```
    利用相機去進行骨架偵測
    ```
3. 2D 影片
    ```
    利用影片進行骨架偵測
    ```
    
