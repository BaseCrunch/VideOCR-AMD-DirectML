@echo off
setlocal

if "%~1"=="" (
    echo Drag and drop a short video file onto this BAT, or run:
    echo tools\run_directml_cli_test.bat "C:\path\to\video.mp4"
    pause
    exit /b 1
)

set "VIDEO=%~1"
set "OUT=%~dpn1.directml.en.srt"
REM Force DirectML to use the discrete Radeon GPU on Ryzen+iGPU systems.
set "VIDEOCR_DIRECTML_DEVICE_INDEX=1"

python CLI\videocr_cli.py ^
  --video_path "%VIDEO%" ^
  --output "%OUT%" ^
  --ocr_engine easyocr_directml ^
  --lang en ^
  --use_gpu true ^
  --time_start 0:00 ^
  --time_end 0:30 ^
  --frames_to_skip 1

echo.
echo Output: %OUT%
pause
