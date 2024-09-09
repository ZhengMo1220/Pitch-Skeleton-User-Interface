# pose_live

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
    # 需要去環境中將libiomp5md.dll刪除，不然會跳warning
    conda install -c conda-forge faiss-gpu

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
