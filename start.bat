@echo off
REM === 初始化 Conda（依照你的安裝位置）===
call "C:\Users\USER\Desktop\Env_Pitcher\Scripts\activate.bat"


REM === 切換到專案資料夾 ===
cd /d C:/Users/USER/Desktop/Pitch-Skeleton-User-Interface_v7/Src/UI_Control

REM === 執行 Python 主程式 ===
python main.py

REM === 避免視窗結束後自動關閉（可刪）===
pause