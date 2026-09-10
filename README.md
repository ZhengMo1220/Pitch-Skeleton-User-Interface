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
10. Setup FLIR camera SDK（僅在需要接實體 FLIR 相機時需要）：
    - 至 [FLIR/Teledyne 官網](https://www.teledynevisionsolutions.com) 登入下載 **Spinnaker SDK**（對應 Windows、Python 3.8、系統位元數）

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
    
