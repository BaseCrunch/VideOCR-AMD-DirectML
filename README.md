English | [中文](README_ch.md)

<p align="center">
<img src="Pictures/VideOCR.png" alt="VideOCR Icon" width="128">
  <h1 align="center">VideOCR AMD DirectML Fork</h1>
  <p align="center">
    Extract hardcoded subtitles from videos with a simple GUI, 200+ languages, and experimental AMD DirectML GPU support.
    <br />
  </p>
</p>

<br>

## ℹ About

VideOCR extracts hardcoded / burned-in subtitles from videos and exports them as `.srt` subtitle files.

This fork adds an experimental **AMD GPU acceleration path for Windows** using:

- **DirectML**
- **torch-directml**
- **EasyOCR**
- **Hybrid OCR mode**

It keeps the original VideOCR features while adding an AMD-friendly OCR backend for systems that do not have NVIDIA CUDA GPUs.

Original VideOCR supports:

- Local OCR with **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)**
- Hybrid cloud recognition with **Google Lens**
- GUI usage
- CLI usage
- 200+ OCR languages depending on selected OCR engine

This fork additionally supports:

- `easyocr_directml` OCR engine
- AMD GPU text detection through DirectML
- CPU-safe text recognition fallback for stability
- DirectML adapter selection for systems with both integrated AMD graphics and a discrete AMD GPU

## AMD DirectML Fork Status

This fork has been tested on:

- **Windows**
- **AMD Radeon RX 7900 XTX**
- **Python 3.12**
- **torch-directml**
- **EasyOCR 1.7.2**

Current AMD backend:

| Stage | Device |
|---|---|
| Video decoding / frame filtering | CPU |
| Image preprocessing / stitching | CPU |
| EasyOCR text detection | AMD GPU through DirectML |
| EasyOCR text recognition | CPU fallback |
| Subtitle merging / SRT generation | CPU |

The recognition stage currently stays on CPU because EasyOCR's LSTM/CRNN recognizer can hit DirectML operator compatibility issues, such as:

```text
aten::_thnn_fused_lstm_cell
```

The hybrid mode is intentional. It avoids the crash while still using the AMD GPU for the text-detection part of the OCR pipeline.

## Important Notes

- The AMD DirectML path is **experimental**.
- DirectML mode is currently intended for **Windows AMD GPUs**.
- NVIDIA users should still use the normal CUDA builds.
- CPU mode still works.
- Google Lens mode still works where available.
- EasyOCR may download model files the first time it runs.
- Python **3.12** is recommended.
- Python **3.13** is not recommended for this fork because `torch-directml` may not provide compatible wheels for it.

## Setup

### Windows CPU / CUDA / Normal Use

You can either install VideOCR with the setup installer or download a folder containing the executable and required files, then unzip it to your desired location.

### Windows AMD DirectML Development Setup

Use this setup if you want to run the AMD DirectML fork directly from source.

Open CMD in the repository folder and run:

```bat
py -3.12 -m venv .venv --upgrade-deps
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[directml]"
python -m pip install --force-reinstall --no-cache-dir "sympy==1.13.3" "mpmath==1.3.0"
```

If your PC has both integrated AMD graphics and a discrete AMD GPU, force the DirectML adapter.

For RX 7900 XTX systems where the discrete GPU is adapter index `1`:

```bat
set VIDEOCR_DIRECTML_DEVICE_INDEX=1
```

Then test DirectML:

```bat
python tools\test_directml.py
python tools\diagnose_easyocr_directml.py
```

Expected result:

```text
DirectML tensor test: 3.0
DirectML device: privateuseone:1
Reader OK
All DirectML diagnostics passed.
```

Then start the GUI:

```bat
python VideOCR.py
```

You can also use the helper:

```bat
run_gui_directml_dev.bat
```

This helper sets:

```bat
set VIDEOCR_DIRECTML_DEVICE_INDEX=1
```

before launching the GUI.

### Windows Graphics Preference

If Windows still sends DirectML workloads to the integrated GPU, force Python to use the high-performance GPU:

```text
Windows Settings
→ System
→ Display
→ Graphics
→ Add desktop app
→ Select:
  .venv\Scripts\python.exe
→ Options
→ High performance
→ AMD Radeon RX 7900 XTX
```

Then launch VideOCR again with:

```bat
set VIDEOCR_DIRECTML_DEVICE_INDEX=1
python VideOCR.py
```

### Linux

Download the tarball archive from the releases page and unzip it to your desired location.

Optionally, you can add VideOCR to your app menus. Open a terminal where you unpacked the archive and run:

```bash
./install_videocr.sh
```

This creates a shortcut for VideOCR.

You can remove it with:

```bash
./uninstall_videocr.sh
```

### Docker

The VideOCR CLI can also be run within a Docker container.

#### Requirements

- **[Docker](https://docs.docker.com/get-docker/)** installed on your system.
- **For CUDA GPU acceleration:** An NVIDIA GPU with the **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** installed on your host machine.

> AMD DirectML mode is Windows/DirectML-based and is not currently supported through the Docker flow.

#### Option A: Download from GitHub Container Registry

Pre-built images are automatically generated and hosted on GitHub.

CPU version:

```bash
docker pull ghcr.io/timminator/videocr-cli-cpu:latest
```

GPU version, CUDA 11.8 / NVIDIA 10 Series graphics cards:

```bash
docker pull ghcr.io/timminator/videocr-cli-gpu-cuda11.8:latest
```

GPU version, CUDA 12.9 / NVIDIA 16 - 50 Series graphics cards:

```bash
docker pull ghcr.io/timminator/videocr-cli-gpu-cuda12.9:latest
```

#### Option B: Build Locally

Clone the repository and use the provided Dockerfile. You can specify the hardware target with `BUILD_TARGET`.

Supported Docker build targets:

```text
cpu
gpu-cuda11.8
gpu-cuda12.9
```

Example CUDA 12.9 GPU build:

```bash
docker build --build-arg BUILD_TARGET=gpu-cuda12.9 -t videocr-cli-gpu:latest .
```

Example CPU build:

```bash
docker build --build-arg BUILD_TARGET=cpu -t videocr-cli-cpu:latest .
```

## GUI Usage

Import a video and seek through the video using the timeline. You can also use the left and right arrow keys.

Draw a crop box over the subtitle area using click and drag. After selecting the subtitle area, start subtitle extraction with the **Run** button.

For AMD DirectML mode, recommended GUI settings are:

```text
OCR Engine: EasyOCR DirectML / AMD GPU
GPU Usage: checked
Full Frame: unchecked
Crop Area: subtitle area only
Language: English, or your subtitle language
```

Recommended 1080p anime episode settings:

```text
Frames to Skip: 2
OCR Image Max Width: 720
SSIM Threshold: 92
Confidence Threshold: 65–75
Max Merge Gap: 0.1–0.3
```

If subtitles are missed, increase accuracy:

```text
Frames to Skip: 1
OCR Image Max Width: 960
```

If processing is too slow, increase speed:

```text
Frames to Skip: 3
OCR Image Max Width: 720
```

![GUI screenshot](Pictures/GUI.png)

## CLI Usage

There is also a CLI version available. Open a terminal in the VideOCR folder and run:

### Windows

```bat
.\videocr-cli.exe -h
```

When running from source:

```bat
python CLI\videocr_cli.py -h
```

### Linux

```bash
./videocr-cli.bin -h
```

### Example Usage: Windows Executable

```bat
.\videocr-cli.exe --video_path "Path\to\your\video\example.mp4" --output "Path\to\your\desired\subtitle\location\example.srt" --lang en --time_start "18:40" --use_gpu true
```

### Example Usage: AMD DirectML Source Mode

```bat
set VIDEOCR_DIRECTML_DEVICE_INDEX=1

python CLI\videocr_cli.py ^
  --video_path "C:\Path\To\video.mp4" ^
  --output "C:\Path\To\output.en.srt" ^
  --ocr_engine easyocr_directml ^
  --lang en ^
  --use_gpu true ^
  --use_fullframe false ^
  --crop_x 36 ^
  --crop_y 793 ^
  --crop_width 1862 ^
  --crop_height 273 ^
  --frames_to_skip 2 ^
  --ssim_threshold 92 ^
  --ocr_image_max_width 720
```

### Example Usage: Docker

When running the Docker container, use Docker volumes with `-v` to mount your local video folder into the container's `/data` directory.

GPU example:

```bash
docker run --rm -it --gpus all \
-v /path/to/your/local/videos:/data \
ghcr.io/timminator/videocr-cli-gpu-cuda12.9:latest \
--video_path /data/my_video.mp4 \
--output /data/my_subtitle.srt \
--use_gpu true
```

CPU example:

```bash
docker run --rm -it \
-v /path/to/your/local/videos:/data \
ghcr.io/timminator/videocr-cli-cpu:latest \
--video_path /data/my_video.mp4 \
--output /data/my_subtitle.srt
```

Any CLI parameters listed below can be appended to the Docker command.

## Performance

### AMD DirectML v12 frame-scan prototype

This fork includes an experimental frame scan mode intended for AMD GPUs on Windows:

```text
AMD FFmpeg D3D11VA Decode + DirectML SSIM (prototype)
```

This mode asks FFmpeg to use D3D11VA hardware decode, then feeds the cropped subtitle area into the EasyOCR DirectML pipeline. It is best tested with a single subtitle crop area and FFmpeg available on PATH. For reliable use, keep DirectML Recognition Mode on `Stable Hybrid`; use `AMD Max Auto` only for experiments.

A quick diagnostic is available:

```bash
python tools/benchmark_amd_decode.py "C:\path\to\video.mp4" --crop 1920:287:0:793 --scale 720:-2 --frames-to-skip 1 --seconds 60
```


Local OCR processing can be slow on CPU. Using a GPU is recommended when available.

This fork provides three practical performance paths:

| Mode | Best For | Notes |
|---|---|---|
| `paddleocr` CPU | Compatibility | Fully local but can be slow |
| `paddleocr` CUDA | NVIDIA GPUs | Fastest official local GPU path |
| `google_lens` | Accuracy / cloud recognition | Requires internet |
| `easyocr_directml` | AMD Windows GPUs | Experimental hybrid DirectML mode |

### AMD DirectML Performance Notes

GPU usage may appear lower than gaming or rendering workloads. This is normal.

The AMD DirectML backend processes OCR in bursts:

1. CPU reads and filters frames.
2. CPU creates stitched OCR image grids.
3. RX 7900 XTX performs text detection through DirectML.
4. CPU performs text recognition for compatibility.
5. CPU merges subtitle lines and writes the `.srt`.

Because only part of the OCR pipeline runs on the GPU, GPU usage around `10–40%` can be normal. This does not mean the GPU is unused.

For RX-class GPUs, v8 adds DirectML grid tuning and v9 adds a friendlier DirectML GPU dropdown. Larger stitched grids reduce per-image overhead and can keep the GPU busier during the detection pass. Good starting values are:

```text
DirectML GPU: GPU 1: AMD Radeon RX 7900 XTX
DirectML Grid Max Width: 2400
DirectML Grid Max Height: 2400
Frames to Skip: 2 for speed, 1 for accuracy
OCR Image Max Width: 720 for speed, 960 for accuracy
```

The GUI DirectML GPU selector writes the selected adapter index to the CLI automatically. The GUI also prints performance lines such as Step 1 time, Step 2 EasyOCR DirectML time, filtered frame count, stitched grid count, and average frames per grid.

## Tips

When cropping, leave a bit of buffer space above and below the subtitle text to improve detection, but do not make the crop box too large.

A tight crop box around the subtitle area is usually much faster and more accurate than full-frame OCR.

### Quick Configuration Cheatsheet

| Option | More Speed | More Accuracy | Notes |
|---|---|---|---|
| Input video quality | Use lower quality | Use higher quality | Cropping reduces the performance cost of high resolution |
| `frames_to_skip` | Higher number | Lower number | For perfectly accurate timestamps, set this to `0` |
| `ssim_threshold` | Lower threshold | Higher threshold | Lower values reduce the number of images sent to OCR |
| `ocr_image_max_width` | Lower value | Higher value | Lower values are faster; higher values help small text |
| `directml_grid_max_width` / `directml_grid_max_height` | Larger values can reduce overhead | Smaller values use less memory | AMD DirectML mode only |
| Crop area | Tighter crop | Slight buffer around text | Avoid full-frame OCR unless needed |

## Command Line Parameters

### `video_path`

Path to the video where subtitles should be extracted from.

### `output`

Path where the `.srt` subtitle file should be stored.

### `ocr_engine`

Select the OCR engine to use for text detection and recognition.

Valid values include:

```text
paddleocr
google_lens
easyocr_directml
```

`paddleocr` uses local processing for both text detection and recognition.

`google_lens` uses hybrid processing where local detection is combined with Google Lens recognition. This mode requires an active internet connection.

`easyocr_directml` uses the AMD DirectML fork backend. In the current stable hybrid mode, EasyOCR text detection runs on DirectML / AMD GPU and recognition falls back to CPU.

### `lang`

Language of the subtitles.

Supported languages depend on the selected OCR engine.

- For `paddleocr`: see the PaddleOCR documentation.
- For `google_lens`: see the Google Lens / Vision language documentation.
- For `easyocr_directml`: use EasyOCR-supported language codes, such as `en`.

### `subtitle_position`

Specifies the alignment of subtitles in the video and allows for better text recognition.

### `conf_threshold`

Confidence threshold for word predictions. Words with lower confidence than this value are discarded.

Default value:

```text
75
```

Make it lower if you get too few words in each line.

Make it higher if there are too many extra words in each line.

### `sim_threshold`

Similarity threshold for subtitle lines. Subtitle lines with larger Levenshtein ratios than this threshold are merged together.

Default value:

```text
80
```

Make it lower if there are too many duplicated subtitle lines.

Make it higher if too few subtitle lines are being generated.

### `ssim_threshold`

If the SSIM between consecutive frames exceeds this threshold, the frame is considered similar and discarded during initial frame filtering in Step 1.

A lower value can greatly reduce the number of images OCR needs to process.

On tight subtitle crop boxes, good results may be possible around:

```text
85–92
```

### `post_processing`

Adds a post-processing step for detected text. This can analyze detected text for missing spaces and insert them automatically.

Currently available for:

```text
English
Spanish
Portuguese
German
Italian
French
```

### `max_merge_gap`

Maximum allowed time gap in seconds between two subtitles to be considered for merging if they are similar.

Default value:

```text
0.09
```

Increase this if the output SRT contains repeated subtitle lines that should have been merged.

### `time_start` and `time_end`

Extract subtitles from only part of the video.

Subtitle timestamps are still calculated according to the full video timeline.

### `use_fullframe`

By default, the specified crop area is used for OCR. If no crop is specified, the bottom third of the frame is used.

Set this to `True` to OCR the entire frame.

### `crop_x`, `crop_y`, `crop_width`, `crop_height`

Specify the bounding area in pixels used for OCR.

![Crop example](Pictures/crop_example.png)

### `crop_x2`, `crop_y2`, `crop_width2`, `crop_height2`

Specify a second bounding area in pixels for OCR when needed.

### `subtitle_alignment` and `subtitle_alignment2`

Subtitle alignment values for ASS / Advanced SubStation Alpha positioning.

Valid values:

```text
bottom-left
bottom-center
bottom-right
middle-left
middle-center
middle-right
top-left
top-center
top-right
```

### `ocr_image_max_width`

Downscales the cropped image frame so its width does not exceed this value before passing it to OCR.

Lower values improve speed.

Higher values may improve accuracy.

### `use_gpu`

Set to `True` to perform OCR with GPU acceleration where supported.

For AMD DirectML mode, also set:

```bat
set VIDEOCR_DIRECTML_DEVICE_INDEX=1
```

if your discrete AMD GPU is adapter index `1`.

### `use_angle_cls`

Set to `True` if classification should be enabled.

For PaddleOCR only.

### `brightness_threshold`

If set, pixels whose brightness is less than the threshold are blackened out.

Valid brightness values range from:

```text
0
```

to:

```text
255
```

This can help improve accuracy when OCR is performed on videos with white subtitles.

### `frames_to_skip`

Number of frames to skip before sampling a frame for OCR.

For 1080p anime episodes, a good starting point is:

```text
2
```

Increase for speed.

Decrease for more accurate timing.

### `min_subtitle_duration`

Subtitles shorter than this threshold are omitted from the final subtitle file.

### `normalize_to_simplified_chinese`

Traditional Chinese characters are converted to Simplified Chinese before processing.

Only active for Chinese & English.

### `use_server_model`

Enables server models for OCR.

This may improve detection at the cost of more processing power.

Primarily for PaddleOCR GPU usage.

## AMD DirectML Environment Variables

### `VIDEOCR_DIRECTML_DEVICE_INDEX`

Selects the DirectML adapter index.

Examples:

```bat
set VIDEOCR_DIRECTML_DEVICE_INDEX=0
```

usually selects the first DirectML adapter.

```bat
set VIDEOCR_DIRECTML_DEVICE_INDEX=1
```

may select the discrete GPU on systems with integrated graphics plus an RX 7900 XTX.

### `VIDEOCR_DIRECTML_GRID_MAX_WIDTH` / `VIDEOCR_DIRECTML_GRID_MAX_HEIGHT`

Optional helper values used by the development launcher. The GUI and CLI settings are preferred.

Recommended RX 7900 XTX starting point:

```bat
set VIDEOCR_DIRECTML_GRID_MAX_WIDTH=2400
set VIDEOCR_DIRECTML_GRID_MAX_HEIGHT=2400
```

### `VIDEOCR_EASYOCR_RECOGNITION_DEVICE`

Optional advanced setting.

Supported values:

```text
cpu
directml
```

Default:

```text
cpu
```

Keep this on `cpu` unless you are testing experimental full DirectML recognition. DirectML recognition may fail on some EasyOCR models due to unsupported LSTM-related operators.

## Build and Compile Instructions

### Requirements

- Python 3.9 or higher
- Python 3.12 recommended for AMD DirectML mode

Windows:

- C++ Build Tools, for example Visual Studio with **Desktop development with C++**
- 7-Zip available in PATH
- Tkinter, included with the default Python installation on Windows

Linux:

- 7-Zip
- Tkinter
- Working dbus installation is recommended

### Clone Repository

```bash
git clone https://github.com/BaseCrunch/VideOCR.git
cd VideOCR
```

If you rename this fork, use your new repository URL instead.

### Install Dependencies

Standard install:

```bash
python -m pip install --upgrade pip
pip install . --group all
```

AMD DirectML source install:

```bat
py -3.12 -m venv .venv --upgrade-deps
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[directml]"
python -m pip install --force-reinstall --no-cache-dir "sympy==1.13.3" "mpmath==1.3.0"
```

### Build

CPU build:

```bash
python build.py --target cpu
```

CUDA 11.8 build:

```bash
python build.py --target gpu-cuda11.8
```

CUDA 12.9 build:

```bash
python build.py --target gpu-cuda12.9
```

AMD DirectML build, if enabled in this fork:

```bash
python build.py --target gpu-directml
```

More information:

```bash
python build.py -h
```

## Troubleshooting

### `No suitable Python runtime found`

Install Python 3.12:

```bat
winget install -e --id Python.Python.3.12
```

Then open a new CMD and verify:

```bat
py -0p
```

### `No module named pip`

Repair pip inside the active venv:

```bat
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
```

### `No module named av`

Install project dependencies inside the active venv:

```bat
python -m pip install -e ".[directml]"
```

### `No module named sympy.core`

Repair SymPy:

```bat
python -m pip install --force-reinstall --no-cache-dir "sympy==1.13.3" "mpmath==1.3.0"
```

### DirectML uses integrated GPU instead of RX 7900 XTX

Set the adapter index:

```bat
set VIDEOCR_DIRECTML_DEVICE_INDEX=1
```

Then run:

```bat
python tools\test_directml.py
python tools\diagnose_easyocr_directml.py
python VideOCR.py
```

Also set Windows graphics preference for:

```text
.venv\Scripts\python.exe
```

to **High performance**.

### EasyOCR DirectML fails on LSTM / CRNN recognition

Keep hybrid mode enabled.

Use:

```text
Detection: DirectML
Recognition: CPU
```

Do not force recognition to DirectML unless testing.

## Credits

This fork is based on the original **VideOCR** project by `timminator`.

Original project:

```text
https://github.com/timminator/VideOCR
```

AMD DirectML fork changes include:

- `easyocr_directml` backend
- DirectML adapter selection
- RX 7900 XTX development helper scripts
- DirectML diagnostics
- Hybrid EasyOCR mode for stable AMD GPU usage

## AMD DirectML / Max GPU Support

This fork adds experimental Windows AMD GPU acceleration through **EasyOCR + torch-directml**. It is intended for AMD Radeon GPUs such as the RX 7900 XTX, while still keeping CPU fallback paths for compatibility.

Recommended 1080p anime settings:

```text
OCR Engine: EasyOCR DirectML (AMD GPU)
DirectML GPU: your discrete Radeon card, e.g. GPU 1: AMD Radeon RX 7900 XTX
AMD Performance Preset: Balanced or Max AMD GPU Load
DirectML Recognition Mode: Stable Hybrid, or AMD Max Auto for testing
Frames to Skip: 1–2
Max OCR Image Width: 720–960
Use Full Frame OCR: unchecked
Crop: subtitle area only
```

CLI options added by this fork:

```bat
--ocr_engine easyocr_directml ^
--directml_device_index 1 ^
--directml_performance_preset max ^
--directml_recognition_mode auto
```

`stable` recognition mode runs detection on DirectML and recognition on CPU for maximum compatibility. `auto` tries DirectML recognition first and falls back to CPU recognition if the EasyOCR LSTM path is unsupported by the installed DirectML stack.

For diagnostics:

```bat
python tools\list_directml_adapters.py
python tools\diagnose_easyocr_directml.py
```

See `README_AMD_DIRECTML.md` for full setup and troubleshooting notes.
