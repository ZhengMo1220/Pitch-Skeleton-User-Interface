# Pitch-Skeleton-User-Interface

> 開發者/維護者請另外參考 [DEVELOPMENT.md](docs/DEVELOPMENT.md)（環境問題排查、硬體相關細節、系統設計背景）。

## Usage
### Installation

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
8. Setup mmpretrain_main environment:
    ```
    cd Src\mmpretrain_main
    pip install -r requirements.txt
    pip install -v -e .
    cd ..\..\
    ```
9. Setup vispy:
    ```
    pip install vispy
    ```
10. Setup FLIR camera SDK（僅在需要接實體 FLIR 相機時需要）

    **必須使用 Spinnaker SDK 4.0.0.116**，不要自行到官網下載最新版。原因見下方說明。

    安裝包內含六個檔案，請向專案維護者索取（檔案約 195MB，未納入 git）：

    ```text
    TeledyneCommonComponentsSetup.exe
    Spinnaker_GenICam_v140_x64.msi
    Spinnaker_Binaries_v140_x64.msi
    Spinnaker_GenTL_v140_x64.msi
    Spinnaker_Drivers_x64.msi
    spinnaker_python-4.0.0.116-cp38-cp38-win_amd64.whl
    ```

    **安裝步驟：**

    a. 先確認 Python 版本（必須是 3.8 且為 64-bit，否則 wheel 無法安裝）：
    ```powershell
    conda activate Pitcher
    python -c "import sys,struct; print(sys.version); print('Bits:',struct.calcsize('P')*8)"
    ```

    b. 關閉相機程式、拔除 FLIR 相機，以系統管理員身分**依序**安裝前五個檔案：
    `TeledyneCommonComponentsSetup.exe` → `Spinnaker_GenICam_v140_x64.msi` → `Spinnaker_Binaries_v140_x64.msi` → `Spinnaker_GenTL_v140_x64.msi` → `Spinnaker_Drivers_x64.msi`
    （若顯示已安裝，選 `Repair`；詢問是否安裝驅動程式時選允許）

    c. **重新啟動 Windows**（udev/驅動權限需重開機才生效）

    d. 安裝 PySpin wheel：
    ```powershell
    conda activate Pitcher
    python -m pip install "<解壓縮路徑>\spinnaker_python-4.0.0.116-cp38-cp38-win_amd64.whl"
    ```

    e. 驗證版本一致（SDK 與 PySpin 必須同為 4.0.0.116）：
    ```powershell
    python -c "import PySpin; s=PySpin.System.GetInstance(); v=s.GetLibraryVersion(); print(f'Spinnaker SDK: {v.major}.{v.minor}.{v.type}.{v.build}'); s.ReleaseInstance()"
    ```

    f. 接上相機後確認能偵測到：
    ```powershell
    python -c "import PySpin; s=PySpin.System.GetInstance(); cs=s.GetCameras(); print('Camera count:',cs.GetSize()); cs.Clear(); s.ReleaseInstance()"
    ```

    > **為什麼鎖定 4.0.0.116**：PySpin 與 Spinnaker SDK 的版本必須嚴格一致，混搭會出現 `DLL load failed while importing PySpin`。而 PySpin 的 wheel 同時也綁定 Python 版本（`cp38` = Python 3.8），本專案的 OpenMMLab 生態系（mmcv 2.0.1 / mmdet 3.1.0 / torch 2.0.1）建構於 Python 3.8 之上，較新的 Spinnaker 版本已逐步不再提供 cp38 的 wheel，因此升級 SDK 會連帶要求升級 Python 與整套 AI 套件。在完成該項評估前，請維持此版本。

    > **相機序號**：專案預設使用序號 `25462483`（正面）與 `25462481`（側面），寫死於 `Src\UI_Control\cv_utils\cv_thread.py`。若你的相機序號不同，即使 SDK 安裝成功仍會開不了相機，需修改該檔案。

> 安裝過程中若遇到任何錯誤，請先查閱 [DEVELOPMENT.md](docs/DEVELOPMENT.md) 的「環境建置已知問題」章節，裡面記錄了目前已知的問題成因與解法。

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

    > 必須先切換到 `Src\UI_Control` 目錄再執行，不能從專案根目錄執行。

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
    
