@echo off
<<<<<<< HEAD
REM === 初始化 Conda（依照你的安裝位置）===
call "C:\Users\USER\Desktop\Env_Pitcher\Scripts\activate.bat"


REM === 切換到專案資料夾 ===
cd /d C:/Users/USER/Desktop/Pitch-Skeleton-User-Interface_v7/Src/UI_Control

REM === 執行 Python 主程式 ===
python main.py

REM === 避免視窗結束後自動關閉（可刪）===
pause
=======
REM Double-click to launch the Pitcher main program.
REM Requires Anaconda/Miniconda and the "Pitcher" environment (see README).

set "CONDA_ACTIVATE="
for %%p in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3" "%LOCALAPPDATA%\anaconda3") do (
    if not defined CONDA_ACTIVATE if exist "%%~p\Scripts\activate.bat" set "CONDA_ACTIVATE=%%~p\Scripts\activate.bat"
)
if not defined CONDA_ACTIVATE (
    echo Anaconda not found. Open "Anaconda Prompt" and run:
    echo     conda activate Pitcher
    echo     cd Src\UI_Control
    echo     python main.py
    pause
    exit /b 1
)

call "%CONDA_ACTIVATE%" Pitcher
REM The program loads configs and models by relative path, so it must run from Src\UI_Control.
cd /d "%~dp0Src\UI_Control"
python main.py
pause
>>>>>>> 776fe9a91c6aab9aa3b75aaa5aba367b3b445aff
