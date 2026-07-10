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
- Added v10 AMD Max Support controls for GPU selection, performance presets, and recognition mode

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

## v9 GPU selector + v8 performance / GPU-load tuning

v8 adds practical tuning controls for the working AMD DirectML hybrid backend:

- GUI DirectML GPU dropdown with Refresh button
- CLI setting for `directml_device_index`
- GUI/CLI settings for `directml_grid_max_width` and `directml_grid_max_height`
- DirectML stitched-grid logging
- Performance timing logs for Step 1 and Step 2
- Better EasyOCR DirectML progress output

Recommended RX 7900 XTX starting values:

```text
DirectML GPU: GPU 1: AMD Radeon RX 7900 XTX
DirectML Grid Max Width: 2400
DirectML Grid Max Height: 2400
Frames to Skip: 2 for speed, 1 for accuracy
Max OCR Image Width: 720 for speed, 960 for accuracy
Full Frame OCR: off
Crop: subtitle area only
```

CLI example:

```bat
python CLI\videocr_cli.py ^
  --video_path "C:\path\to\episode.mp4" ^
  --output "C:\path\to\episode.en.srt" ^
  --ocr_engine easyocr_directml ^
  --lang en ^
  --use_gpu true ^
  --directml_device_index 1 ^
  --directml_grid_max_width 2400 ^
  --directml_grid_max_height 2400 ^
  --frames_to_skip 2 ^
  --ocr_image_max_width 720 ^
  --crop_x 0 ^
  --crop_y 760 ^
  --crop_width 1920 ^
  --crop_height 300
```

The current DirectML backend is still hybrid: detection runs on the selected DirectML adapter, while recognition stays on CPU to avoid EasyOCR's unsupported LSTM path on `torch-directml`. Raising the grid size can reduce per-image overhead and keep the GPU busier, but it will not make the GPU run at 100% because recognition and subtitle merging are still CPU-side.

## AMD Max Support / v10 notes

This fork now includes extra DirectML controls intended to support more AMD GPU setups and to let users push their GPU harder when their driver and `torch-directml` stack can handle it.

### DirectML GPU selection

The GUI includes a DirectML GPU selector under Advanced Settings. On many Ryzen desktops:

```text
GPU 0 = integrated AMD Radeon(TM) Graphics
GPU 1 = discrete AMD Radeon RX card
```

The CLI also supports:

```bat
--directml_device_index 1
```

Auto mode now tries to prefer the high-performance/discrete adapter by checking Windows video-controller names and adapter RAM before falling back to the older GPU 1 -> GPU 0 order.

### AMD Performance Preset

The GUI and CLI now support:

```bat
--directml_performance_preset compatibility
--directml_performance_preset balanced
--directml_performance_preset max
--directml_performance_preset manual
```

Preset behavior:

| Preset | Grid target | Intended use |
|---|---:|---|
| `compatibility` | 1600x1600 | Older/lower-VRAM GPUs, stability first |
| `balanced` | 2400x2400 | Recommended default |
| `max` | 4096x4096 | Larger batches to feed high-end AMD GPUs harder |
| `manual` | Uses the grid width/height fields | Manual tuning |

### DirectML Recognition Mode

The GUI and CLI now support:

```bat
--directml_recognition_mode stable
--directml_recognition_mode auto
--directml_recognition_mode experimental
```

Mode behavior:

| Mode | Detection | Recognition | Notes |
|---|---|---|---|
| `stable` | DirectML GPU | CPU | Safest; avoids EasyOCR LSTM DirectML crash |
| `auto` | DirectML GPU | Try DirectML GPU, then CPU fallback | Best “max AMD” test mode |
| `experimental` | DirectML GPU | Try DirectML GPU | Still falls back for known LSTM compatibility failures |

Current known limitation: EasyOCR recognition uses a BiLSTM path. Some `torch-directml` versions cannot run that operator on DirectML reliably. `auto` mode tries the GPU recognizer first, then falls back to the stable CPU recognizer if the known LSTM failure appears.

### New diagnostics

```bat
python tools\list_directml_adapters.py
python tools\diagnose_easyocr_directml.py
```

These show detected Windows GPU adapters, the likely preferred high-performance adapter, the selected DirectML device, and whether the patched EasyOCR reader can start.


## v10.1 Hotfix

AMD Max Auto now catches the known EasyOCR / torch-directml LSTM recognition failure (`aten::_thnn_fused_lstm_cell`) and automatically retries the current OCR image in Stable Hybrid mode. This keeps DirectML text detection on the selected AMD GPU while moving text recognition back to CPU when required, instead of crashing the whole subtitle job.

## v11 Experimental AMD Frame Scan Mode

v11 adds an experimental Step 1 frame-scan mode for AMD/DirectML users:

```bat
--directml_frame_scan_mode cpu_ssim
--directml_frame_scan_mode directml_ssim
```

| Mode | Frame similarity scan | Notes |
|---|---|---|
| `cpu_ssim` | CPU SSIM | Safest and most compatible |
| `directml_ssim` | DirectML global SSIM-style comparison | Experimental; moves Step 1 similarity comparisons onto the selected AMD/DirectML adapter |

Important limitation: PyAV still decodes video and performs crop/scale operations on the CPU. The new DirectML frame-scan mode offloads the per-frame similarity comparison after the crop/scale step. A true fully GPU-resident pipeline would require a larger rewrite around FFmpeg/D3D11VA/Direct3D textures or an ONNX/DirectML video preprocessing pipeline.

Recommended experimental high-load AMD settings:

```text
DirectML GPU: GPU 1: AMD Radeon RX 7900 XTX
AMD Performance Preset: Max AMD GPU Load
DirectML Recognition Mode: Stable Hybrid or AMD Max Auto
AMD Frame Scan Mode: AMD DirectML SSIM (experimental)
Frames to Skip: 1 or 2
Max OCR Image Width: 720 or 960
Use Full Frame OCR: unchecked
```

## v12 Experimental AMD Hardware Decode / Frame Scan Prototype

v12 adds a new frame scan mode for users who want to push more of the early video pipeline toward the AMD GPU:

```text
AMD FFmpeg D3D11VA Decode + DirectML SSIM (prototype)
```

This mode asks FFmpeg to use Windows D3D11VA hardware acceleration while decoding the video, then streams the cropped subtitle region into VideOCR's existing stitched OCR grid pipeline. When SSIM filtering is enabled, the similarity comparison is also attempted through DirectML on the selected adapter.

Recommended v12 test settings for a Radeon RX 7900 XTX:

```text
OCR Engine: EasyOCR DirectML (AMD GPU)
DirectML GPU: GPU 1: AMD Radeon RX 7900 XTX
AMD Performance Preset: Max AMD GPU Load
DirectML Recognition Mode: Stable Hybrid (recommended)
AMD Frame Scan Mode: AMD FFmpeg D3D11VA Decode + DirectML SSIM (prototype)
Frames to Skip: 1 or 2
Max OCR Image Width: 720
Use Full Frame OCR: unchecked
```

Important limitations:

- FFmpeg must be installed and available on PATH.
- The v12 hardware decode prototype currently supports the common single-zone subtitle crop workflow first.
- The decoded/cropped frames still have to be copied back to CPU memory before EasyOCR can process the stitched images.
- EasyOCR recognition may still use CPU fallback because the PyTorch DirectML LSTM recognizer can hit unsupported operator paths.
- Higher GPU usage is not guaranteed if CPU-side image handling, Python overhead, or recognition remains the bottleneck.

A helper benchmark is included:

```bash
python tools/benchmark_amd_decode.py "C:\path\to\video.mp4" --crop 1920:287:0:793 --scale 720:-2 --frames-to-skip 1 --seconds 60
```

Use this to check whether FFmpeg D3D11VA decode is working on your system before relying on the prototype mode for full episodes.

## v13 ONNX Runtime DirectML experimental OCR

v13 adds a new experimental OCR engine:

```text
ONNX Runtime DirectML (AMD GPU Experimental)
```

CLI value:

```bat
--ocr_engine onnx_directml
```

This backend tries to use RapidOCR through ONNX Runtime with the DirectML provider. If the ONNX stack is not installed, does not expose `DmlExecutionProvider`, or fails during processing, VideOCR falls back to the working EasyOCR DirectML Hybrid path for that run so the subtitle job can still finish.

The DirectML optional dependency group now includes:

```text
onnxruntime-directml
rapidocr-onnxruntime
```

v13 also adds a GUI **Last Run Benchmark** panel. The CLI emits `[Perf]` and `[Bench]` lines with step timing, end-to-end runtime, and speed vs real-time.

## v14 - ONNX DirectML tuning and benchmark compare

v14 adds ONNX-specific tuning because the ONNX Runtime DirectML backend can reserve very large VRAM when OCR grids are too large.

New GUI settings:

- ONNX DirectML Tuning
  - Low VRAM / safer: 1600x1600 stitched OCR grids
  - Balanced ONNX (recommended): 2048x2048 stitched OCR grids
  - Max Throughput / higher VRAM: 3072x3072 stitched OCR grids
  - Manual Grid Size: uses the visible DirectML Grid Max Width/Height boxes

- Benchmark Compare ONNX vs EasyOCR sample
  - Optional. Runs a small EasyOCR Hybrid sample after the ONNX run and prints [BenchCompare] lines.
  - This will make the run longer, so leave it off for normal processing.

New CLI options:

```bat
--onnx_directml_tuning balanced
--benchmark_compare_engine false
--benchmark_compare_sample_grids 3
```

Recommended v14 ONNX test settings for RX 7900 XTX:

```text
OCR Engine: ONNX Runtime DirectML (AMD GPU Experimental)
DirectML GPU: GPU 1: AMD Radeon RX 7900 XTX
AMD Performance Preset: Max AMD GPU Load
ONNX DirectML Tuning: Balanced ONNX (recommended)
AMD Frame Scan Mode: AMD FFmpeg D3D11VA Decode + DirectML SSIM (prototype)
DirectML Recognition Mode: Stable Hybrid (recommended)
Frames to Skip: 1 or 2
Max OCR Image Width: 720
Benchmark Compare: off for normal runs, on only when comparing
```

Optional standalone benchmark compare tool:

```bat
python tools\benchmark_ocr_compare.py "C:\path\to\video.mp4" --crop 0:793:1920:287 --frames-to-skip 1 --onnx-directml-tuning balanced
```
