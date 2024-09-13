# Pitch-Skeleton-User-Interface

## Usage
### Installation
1. Clone this repo.
2. Setup conda environment:
    ```
    conda create -n Pitcher python=3.7 -y
    conda activate Pitcher
    pip install -r requirements.txt
    # CUDA 11.7
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    # 需要去環境中將anaconda3\envs\your_envs_name\libiomp5md.dll刪除，不然會跳warning
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
    cd Src\mmengine_main
    pip install -r requirements.txt
    pip install -v -e .
    cd ..\..\
    ```
8. Setup mmpretrain_main environment:
    ```
    cd Src\mmpretrain_main
    pip install -r requirements.txt
    pip install -v -e .
    cd ..\..\
    ```
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
    ```
    python Src\UI_Control\main.py
    ```
2. 2D 相機
    ```
    利用相機去進行骨架偵測
    ```
3. 2D 影片
    ```
    利用影片進行骨架偵測
    ```
    
