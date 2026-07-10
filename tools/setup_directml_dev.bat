@echo off
setlocal
cd /d "%~dp0\.."

echo ==========================================
echo VideOCR AMD DirectML dev setup
 echo ==========================================
echo.

echo Checking for Python 3.12...
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.12 was not found.
    echo Install it first with:
    echo   winget install -e --id Python.Python.3.12
    echo.
    echo Then close this window, reopen CMD, and run this file again.
    pause
    exit /b 1
)

if exist .venv (
    echo Removing old .venv...
    rmdir /s /q .venv
)

echo Creating .venv with Python 3.12...
py -3.12 -m venv .venv
if errorlevel 1 goto fail

call .venv\Scripts\activate.bat
if errorlevel 1 goto fail

echo Upgrading pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto fail

echo Installing VideOCR DirectML dependencies...
python -m pip install -e ".[directml]"
if errorlevel 1 goto fail

echo Testing DirectML on DirectML adapter index 1...
REM Force DirectML to use the discrete Radeon GPU on Ryzen+iGPU systems.
set "VIDEOCR_DIRECTML_DEVICE_INDEX=1"
python tools\test_directml.py
if errorlevel 1 goto fail

echo.
echo SUCCESS. Now run:
echo   .venv\Scripts\activate.bat
echo   python VideOCR.py
echo.
pause
exit /b 0

:fail
echo.
echo SETUP FAILED. Please copy the error above and send it.
pause
exit /b 1
