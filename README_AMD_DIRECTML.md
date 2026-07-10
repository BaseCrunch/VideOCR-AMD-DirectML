# VideOCR AMD DirectML experimental backend

This fork patch adds an experimental **EasyOCR DirectML** engine for Windows systems with AMD GPUs such as the RX 7900 XTX.

The existing VideOCR GPU packages are CUDA/NVIDIA builds. They cannot run on AMD hardware. This patch does not try to fake CUDA support. Instead, it adds a separate OCR path:

```text
VideOCR frame extraction / crop / SSIM filtering
        ↓
EasyOCR
        ↓
torch-directml
        ↓
AMD / DirectX 12 GPU
```

## What changed

- Added a new CLI engine: `easyocr_directml`
- Added a new GUI option: `EasyOCR DirectML (AMD GPU)`
- Added a new backend file: `CLI/videocr/easyocr_directml.py`
- Avoided the old NVIDIA `nvidia-smi` check when using EasyOCR DirectML
- Added optional Python dependencies: `easyocr` and `torch-directml`
- Added a `gpu-directml` build target in `build.py`

## Important limitations

This is an experimental backend, not an official upstream VideOCR release.

- I could syntax-check the patched Python files, but I could not test DirectML on an RX 7900 XTX in this Linux container.
- EasyOCR may download model files on first run.
- Some PyTorch operations may fall back or fail on DirectML depending on your installed `torch-directml`, driver, and Windows version.
- The DirectML backend currently uses EasyOCR detection + recognition in one pass. It does not use PaddleOCR's CUDA backend.


## Required Python version

Use **Python 3.12 x64** for this DirectML patch. Do not use Python 3.13.

`torch-directml` currently publishes Windows wheels for CPython 3.8, 3.9, 3.10, 3.11, and 3.12, but not 3.13. If `python --version` shows 3.13, create the virtual environment with `py -3.12` instead.

Quick setup:

```bat
winget install -e --id Python.Python.3.12
cd /d C:\Users\bweak\Downloads\VideOCR-master\VideOCR-master
rmdir /s /q .venv 2>nul
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[directml]"
python tools\test_directml.py
python VideOCR.py
```

## Install for source/development testing

From the patched project folder:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[directml]"
```

Then test DirectML itself:

```bat
python tools\test_directml.py
```

Expected result:

```text
DirectML test passed.
Device: privateuseone:0
Result: 3.0
```

## Run CLI test

Use a short clip first. Subtitle crop is strongly recommended.

```bat
python CLI\videocr_cli.py ^
  --video_path "C:\path\to\test.mp4" ^
  --output "C:\path\to\test.en.srt" ^
  --ocr_engine easyocr_directml ^
  --lang en ^
  --use_gpu true ^
  --time_start 0:00 ^
  --time_end 0:30 ^
  --frames_to_skip 1 ^
  --crop_x 0 ^
  --crop_y 700 ^
  --crop_width 1920 ^
  --crop_height 380
```

For a 1080p anime video, a good starting subtitle crop is usually:

```text
x = 0
y = 650 to 760
width = 1920
height = 260 to 430
```

## Run GUI

After installing the dependencies:

```bat
python VideOCR.py
```

Select:

```text
OCR Engine: EasyOCR DirectML (AMD GPU)
Subtitle Language: English
Enable GPU Usage: checked
```

## Build DirectML target

From the patched project folder:

```bat
py -3.12 -m venv .venv-build
.venv-build\Scripts\activate
python -m pip install --upgrade pip
python -m pip install ".[directml]"
python -m pip install nuitka requests
python build.py --target gpu-directml --windows-installer true --archive true
```

The build script names the output like:

```text
VideOCR-GPU-v1.5.1-DirectML
videocr-cli-GPU-v1.5.1-DirectML
```

## If DirectML fails

Try these checks in order:

1. Update AMD Adrenalin drivers.
2. Run `python tools\test_directml.py`.
3. In VideOCR, keep `EasyOCR DirectML (AMD GPU)` selected but uncheck `Enable GPU Usage`. This confirms EasyOCR CPU mode works.
4. Try English first before Japanese/Chinese/Korean.
5. Use a subtitle crop instead of full-frame OCR.

If CPU EasyOCR works but DirectML fails, the patch is wired correctly and the issue is probably `torch-directml` operator support, driver, or package version compatibility.

## Fix for `videocr-cli not found` in source/dev mode

If the GUI opens but Run shows:

```text
Error: videocr-cli not found. Please check the path.
```

apply the v3 changed files. The GUI now falls back to:

```bat
python CLI\videocr_cli.py
```

using the same Python interpreter that launched `VideOCR.py`. This means you can run the GUI directly from the source folder without building `videocr-cli.exe` first.

Make sure you start the GUI from the activated venv:

```bat
cd /d C:\Users\bweak\Downloads\VideOCR-master\VideOCR-master
call .venv\Scripts\activate.bat
python VideOCR.py
```


## v4 dependency/import fix

If the GUI reaches `Running EasyOCR DirectML pass...` and then says EasyOCR is missing even though pip installed it, run:

```bat
python -m pip install --force-reinstall "numpy==1.26.4" "opencv-python-headless==4.10.0.84"
python -m pip install --upgrade --force-reinstall "easyocr==1.7.2" "torch-directml==0.2.5.dev240914"
python tools\diagnose_easyocr_directml.py
python VideOCR.py
```

v4 also reports the real EasyOCR import exception instead of hiding it behind a generic missing-dependency message.

## v5 DirectML runtime patch

v5 adds a runtime EasyOCR patch for DirectML. EasyOCR's normal GPU path assumes CUDA-style `torch.nn.DataParallel`. That path can fail on AMD DirectML. This patch loads EasyOCR's detector and recognizer weights on CPU, strips DataParallel prefixes when needed, moves the plain FP32 models to the DirectML device, and prints full tracebacks if DirectML startup or OCR reading fails.

After copying v5 over the project, run inside your active `.venv`:

```bat
python -m pip install --force-reinstall --no-cache-dir "sympy==1.13.3" "mpmath==1.3.0"
python tools\diagnose_easyocr_directml.py
python VideOCR.py
```

If the diagnostic reader passes, run the same short video test again.


## v6 note

DirectML mode now uses a hybrid pipeline: EasyOCR text detection runs on DirectML when supported, while EasyOCR text recognition is forced to CPU because torch-directml currently crashes on EasyOCR's BiLSTM/LSTM recognizer path (`aten::_thnn_fused_lstm_cell`). This prevents the 19:00 crash and should produce an SRT instead of failing after Step 1.

## v7 RX 7900 XTX adapter selection fix

On Ryzen systems with integrated graphics plus an RX 7900 XTX, DirectML may choose adapter `0`, which is often the integrated `AMD Radeon(TM) Graphics`. v7 changes the DirectML picker so it prefers adapter `1` when multiple adapters are visible, then falls back to adapter `0` if needed.

To force the RX 7900 XTX manually before launching the GUI:

```bat
cd /d C:\Users\bweak\Downloads\VideOCR-master\VideOCR-master
call .venv\Scripts\activate.bat
set VIDEOCR_DIRECTML_DEVICE_INDEX=1
python VideOCR.py
```

The helper `run_gui_directml_dev.bat` now sets this automatically.

When it is working, the console should print:

```text
Selected DirectML adapter index: 1
Using EasyOCR DirectML device: privateuseone:1
```

If Task Manager still shows the integrated GPU being used, try:

```bat
set VIDEOCR_DIRECTML_DEVICE_INDEX=0
python VideOCR.py
```

Only use `0` if Windows/torch-directml reports the RX card as adapter 0 on your machine.
