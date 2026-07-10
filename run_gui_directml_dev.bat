@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo Missing .venv. Create it first with: py -3.12 -m venv .venv
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
REM Force DirectML to use the discrete Radeon GPU on Ryzen+iGPU systems.
REM Change to 0 only if your RX card is listed as adapter 0.
set "VIDEOCR_DIRECTML_DEVICE_INDEX=1"
REM Larger stitched grids reduce per-image overhead and can keep a fast Radeon GPU busier.
REM These are also available in the GUI Advanced Settings tab.
set "VIDEOCR_DIRECTML_GRID_MAX_WIDTH=2400"
set "VIDEOCR_DIRECTML_GRID_MAX_HEIGHT=2400"
python VideOCR.py
pause
