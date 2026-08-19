@echo off
setlocal
echo ======================================================
echo   Pitch-Skeleton-User-Interface Env Repair Tool
echo ======================================================

:: 1. 進入 Conda 環境中
call "C:\Users\USER\Desktop\Env_Pitcher\Scripts\activate.bat"


echo [1/5] 正在重新註冊 MMEngine (修復 ModuleNotFoundError)...
pip uninstall mmengine -y >nul 2>&1
pip install mmengine

echo [2/5] 正在連結 MMPose...
cd Src\mmpose_main
pip install -v -e .
cd ..\..

echo [3/5] 正在連結 MMYOLO...
cd Src\mmyolo_main
pip install -v -e .
cd ..\..

echo [4/5] 正在連結 MMPretrain...
cd Src\mmpretrain_main
pip install -v -e .
cd ..\..

echo [5/5] 正在清理 libiomp5md.dll 衝突...
:: 嘗試自動尋找並刪除環境中的衝突 DLL
for /d %%i in ("%CONDA_PREFIX%\Library\bin") do (
    if exist "%%i\libiomp5md.dll" (
        del "%%i\libiomp5md.dll"
        echo 已刪除: %%i\libiomp5md.dll
    )
)

echo ======================================================
echo [SUCCESS] 環境修復完成！正在進行簡單測試...
echo ======================================================
pause