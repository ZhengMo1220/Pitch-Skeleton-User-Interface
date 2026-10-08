@echo off
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
