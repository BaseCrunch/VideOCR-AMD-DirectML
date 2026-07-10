from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

from . import utils

# EasyOCR uses different language identifiers than PaddleOCR.
# Keep the map deliberately conservative so unsupported values fail with a clear message.
EASYOCR_LANG_MAP: dict[str, list[str]] = {
    "en": ["en"],
    "ja": ["ja", "en"],
    "japan": ["ja", "en"],
    "ko": ["ko", "en"],
    "korean": ["ko", "en"],
    "ch": ["ch_sim", "en"],
    "zh": ["ch_sim", "en"],
    "zh-CN": ["ch_sim", "en"],
    "chinese_cht": ["ch_tra", "en"],
    "zh-TW": ["ch_tra", "en"],
    "de": ["de", "en"],
    "german": ["de", "en"],
    "fr": ["fr", "en"],
    "es": ["es", "en"],
    "it": ["it", "en"],
    "pt": ["pt", "en"],
    "ru": ["ru", "en"],
    "ar": ["ar", "en"],
    "id": ["id", "en"],
    "th": ["th", "en"],
    "vi": ["vi", "en"],
}


def normalize_easyocr_lang(lang: str) -> list[str]:
    """Return the EasyOCR language list for a VideOCR language code."""
    key = (lang or "en").strip()
    if key in EASYOCR_LANG_MAP:
        return EASYOCR_LANG_MAP[key]

    # Many EasyOCR language IDs are already two-letter ISO codes.
    if len(key) == 2:
        return [key]

    supported = ", ".join(sorted(EASYOCR_LANG_MAP.keys()))
    raise ValueError(
        f"EasyOCR DirectML does not have a language mapping for '{lang}'. "
        f"Use one of these VideOCR language codes or add a mapping in easyocr_directml.py: {supported}"
    )



def _windows_video_adapters() -> list[dict[str, Any]]:
    """Return Windows video-controller names/RAM in likely DirectML order.

    This is best-effort only. It helps Auto mode prefer a discrete/high-VRAM
    Radeon GPU instead of a Ryzen iGPU when torch-directml exposes multiple
    adapters but not friendly adapter names.
    """
    if sys.platform != "win32":
        return []

    ps_cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
    ]
    try:
        result = subprocess.run(ps_cmd, capture_output=True, text=True, timeout=5)
        payload = (result.stdout or "").strip()
        if payload:
            data = json.loads(payload)
            if isinstance(data, dict):
                data = [data]
            adapters: list[dict[str, Any]] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("Name") or "").strip()
                ram_raw = item.get("AdapterRAM") or 0
                try:
                    ram = int(ram_raw)
                except Exception:
                    ram = 0
                if name:
                    adapters.append({"name": name, "ram": ram})
            if adapters:
                return adapters
    except Exception:
        pass

    wmic_cmd = ["wmic", "path", "win32_VideoController", "get", "name,adapterram"]
    try:
        result = subprocess.run(wmic_cmd, capture_output=True, text=True, timeout=5)
        adapters = []
        for raw in (result.stdout or "").splitlines()[1:]:
            line = " ".join(raw.split())
            if not line:
                continue
            parts = line.split(" ", 1)
            ram = 0
            name = line
            if parts and parts[0].isdigit():
                try:
                    ram = int(parts[0])
                except Exception:
                    ram = 0
                name = parts[1] if len(parts) > 1 else "Unknown GPU"
            adapters.append({"name": name.strip(), "ram": ram})
        return adapters
    except Exception:
        return []


def _score_video_adapter(adapter: dict[str, Any]) -> int:
    """Score adapters so Auto mode prefers the fastest discrete AMD card."""
    name = str(adapter.get("name") or "").lower()
    ram = int(adapter.get("ram") or 0)
    score = ram // (256 * 1024 * 1024)

    # Prefer discrete/high-performance cards. Keep NVIDIA/Intel Arc positive too
    # because DirectML is vendor-neutral, while still strongly favoring AMD/Radeon.
    if "radeon rx" in name or "rx " in name:
        score += 1000
    elif "radeon" in name or "amd" in name:
        score += 600
    elif "geforce" in name or "nvidia" in name or "rtx" in name:
        score += 500
    elif "arc" in name:
        score += 350

    # Common integrated-GPU names should not win unless they are the only adapter.
    if "integrated" in name or "radeon(tm) graphics" in name or "graphics" == name.strip():
        score -= 400
    if "microsoft basic" in name or "remote" in name:
        score -= 1000
    return score


def _preferred_high_performance_adapter_index() -> int | None:
    adapters = _windows_video_adapters()
    if not adapters:
        return None
    scored = [(idx, _score_video_adapter(adapter), adapter) for idx, adapter in enumerate(adapters)]
    scored.sort(key=lambda item: item[1], reverse=True)
    best_idx, best_score, _ = scored[0]
    if best_score <= -500:
        return None
    return int(best_idx)


def get_directml_recognition_mode() -> str:
    """Return DirectML recognition strategy.

    stable: detector on DirectML, recognizer on CPU. Safest.
    auto: try GPU recognition, fall back to CPU if DirectML lacks an op.
    experimental: force GPU recognition attempt, still falls back on known LSTM failure.
    """
    mode = os.environ.get("VIDEOCR_DIRECTML_RECOGNITION_MODE", "stable").strip().lower()
    aliases = {
        "cpu": "stable",
        "hybrid": "stable",
        "safe": "stable",
        "dml": "experimental",
        "gpu": "experimental",
        "full": "experimental",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"stable", "auto", "experimental"}:
        mode = "stable"
    return mode


def _is_lstm_directml_failure(e: BaseException) -> bool:
    text = _format_exception(e).lower()
    needles = (
        "_thnn_fused_lstm_cell",
        "lstm",
        "not currently supported on the dml backend",
        "could not run 'aten::",
    )
    return any(n in text for n in needles)


def _safe_directml_device_count(torch_directml: Any) -> int | None:
    """Return the DirectML adapter count when the installed torch-directml exposes it."""
    for name in ("device_count", "get_device_count"):
        fn = getattr(torch_directml, name, None)
        if callable(fn):
            try:
                count = int(fn())
                if count >= 0:
                    return count
            except Exception:
                pass
    return None


def _build_directml_candidate_indices(torch_directml: Any) -> list[int | None]:
    """Pick DirectML adapter candidates.

    Many AMD desktops expose the Ryzen iGPU as DirectML adapter 0 and the real
    Radeon dGPU as adapter 1. The default torch_directml.device() therefore can
    land on the integrated GPU. VIDEOCR_DIRECTML_DEVICE_INDEX lets the user force
    the correct adapter. Without an env override, prefer adapter 1 when multiple
    adapters are visible, then fall back to adapter 0/default.
    """
    candidates: list[int | None] = []

    forced = os.environ.get("VIDEOCR_DIRECTML_DEVICE_INDEX", "").strip()
    if forced:
        try:
            candidates.append(int(forced))
        except ValueError:
            raise RuntimeError(
                "VIDEOCR_DIRECTML_DEVICE_INDEX must be a number, for example:\n"
                "  set VIDEOCR_DIRECTML_DEVICE_INDEX=1"
            )

    prefer_high_perf = os.environ.get("VIDEOCR_DIRECTML_AUTO_PREFER_HIGH_PERFORMANCE", "1").strip().lower() not in {"0", "false", "no", "off"}
    preferred = _preferred_high_performance_adapter_index() if prefer_high_perf else None
    if preferred is not None:
        candidates.append(preferred)

    count = _safe_directml_device_count(torch_directml)
    if count is not None:
        # On systems like Ryzen 7800X3D + RX 7900 XTX, adapter 0 is commonly the
        # integrated AMD Radeon Graphics and adapter 1 is the RX 7900 XTX. Try
        # the detected high-performance adapter first, then the common dGPU
        # index, then all visible adapters.
        if count > 1:
            candidates.append(1)
        candidates.extend(range(count))
    else:
        # Older torch-directml builds may not expose a count. Try the common dGPU
        # index first, then default/0.
        candidates.extend([1, 0, None])

    candidates.append(None)

    deduped: list[int | None] = []
    for value in candidates:
        if value not in deduped:
            deduped.append(value)
    return deduped


def get_directml_device() -> Any:
    """Create a torch-directml device and run a tiny sanity check."""
    if sys.platform != "win32":
        raise RuntimeError("EasyOCR DirectML mode is Windows-only. Use CPU mode or CUDA/ROCm on other platforms.")

    try:
        import torch  # type: ignore
        import torch_directml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "EasyOCR DirectML dependencies are missing. Install them with:\n"
            "  python -m pip install easyocr torch-directml\n"
            "Then run VideOCR again."
        ) from e

    errors: list[str] = []
    for index in _build_directml_candidate_indices(torch_directml):
        try:
            if index is None:
                device = torch_directml.device()
                label = "default"
            else:
                device = torch_directml.device(index)
                label = str(index)

            x = torch.tensor([1.0]).to(device)
            _ = (x + 1).cpu().item()

            try:
                device._videocr_directml_index = index
            except Exception:
                pass

            print(f"Selected DirectML adapter index: {label}", flush=True)
            return device
        except Exception as e:
            errors.append(f"adapter {index if index is not None else 'default'}: {type(e).__name__}: {e}")

    raise RuntimeError(
        "torch-directml is installed, but DirectML could not initialize on any tested adapter. "
        "Update your AMD driver and verify DirectX 12 support.\n\n"
        "Tried:\n  " + "\n  ".join(errors)
    )



def _format_exception(e: BaseException) -> str:
    """Return a compact but useful chained exception traceback."""
    import traceback
    return "".join(traceback.format_exception(type(e), e, e.__traceback__)).strip()


def _is_known_directml_recognition_failure(e: BaseException) -> bool:
    """Return True for the known EasyOCR/torch-directml recognizer crash.

    EasyOCR's recognition model uses a BiLSTM path. On the current Windows
    torch-directml stack this may surface as an unsupported/fallback failure
    around aten::_thnn_fused_lstm_cell. In that case v10.1 should retry in
    Stable Hybrid mode instead of aborting the whole job.
    """
    text = _format_exception(e)
    needles = (
        "aten::_thnn_fused_lstm_cell",
        "not currently supported on the DML backend",
        "PrivateUse1",
        "LSTM",
        "lstm",
    )
    return any(needle in text for needle in needles)


_DML_PATCH_APPLIED = False


def patch_easyocr_for_directml(easyocr_module: Any) -> None:
    """
    EasyOCR's CUDA path wraps both CRAFT and recognizer models in
    torch.nn.DataParallel. torch-directml does not support that path well.

    Patch EasyOCR at runtime so DirectML mode loads weights on CPU, removes the
    optional DataParallel/module prefix, and then moves the plain model to the
    DirectML device.
    """
    global _DML_PATCH_APPLIED
    if _DML_PATCH_APPLIED:
        return

    try:
        import importlib
        from collections import OrderedDict

        import torch  # type: ignore
        import easyocr.detection as detection  # type: ignore
        import easyocr.easyocr as easyocr_core  # type: ignore
        import easyocr.recognition as recognition  # type: ignore
        from easyocr.craft import CRAFT  # type: ignore
        from easyocr.utils import CTCLabelConverter  # type: ignore
        from easyocr.detection import copyStateDict  # type: ignore
    except Exception as e:
        raise RuntimeError("Could not prepare EasyOCR DirectML runtime patch:\n" + _format_exception(e)) from e

    def _is_directml_device(device: Any) -> bool:
        return str(device).startswith("privateuseone")

    def dml_get_detector(trained_model: str, device: Any = "cpu", quantize: bool = True, cudnn_benchmark: bool = False):
        if not _is_directml_device(device):
            return _ORIG_GET_DETECTOR(trained_model, device=device, quantize=quantize, cudnn_benchmark=cudnn_benchmark)

        net = CRAFT()
        # DirectML map_location during torch.load is unreliable. Load on CPU first.
        state = torch.load(trained_model, map_location="cpu", weights_only=False)
        net.load_state_dict(copyStateDict(state))
        net = net.to(device)
        net.eval()
        return net

    def dml_get_recognizer(recog_network: str, network_params: dict[str, Any], character: str,
                           separator_list: dict[str, Any], dict_list: dict[str, str], model_path: str,
                           device: Any = "cpu", quantize: bool = True):
        if not _is_directml_device(device):
            return _ORIG_GET_RECOGNIZER(recog_network, network_params, character,
                                        separator_list, dict_list, model_path,
                                        device=device, quantize=quantize)

        converter = CTCLabelConverter(character, separator_list, dict_list)
        num_class = len(converter.character)

        if recog_network == "generation1":
            model_pkg = importlib.import_module("easyocr.model.model")
        elif recog_network == "generation2":
            model_pkg = importlib.import_module("easyocr.model.vgg_model")
        else:
            model_pkg = importlib.import_module(recog_network)

        model = model_pkg.Model(num_class=num_class, **network_params)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

        # EasyOCR recognition weights are usually saved from DataParallel with
        # a leading "module." prefix. Strip it if present.
        cleaned = OrderedDict()
        for key, value in state_dict.items():
            cleaned[key[7:] if key.startswith("module.") else key] = value

        model.load_state_dict(cleaned)

        recognition_mode = get_directml_recognition_mode()
        if recognition_mode in {"auto", "experimental"}:
            # Try to place EasyOCR's recognizer on DirectML as well. Some
            # torch-directml/PyTorch versions still lack the BiLSTM op EasyOCR
            # uses; run_easyocr_on_stitched_images will catch that known failure
            # and recreate the reader in stable CPU-recognition mode.
            model = model.to(device)
            try:
                model._videocr_directml_recognizer_experimental = True
            except Exception:
                pass
        else:
            # Stable hybrid mode: DirectML detector + CPU recognizer.
            model = model.to("cpu")
            try:
                model._videocr_cpu_recognizer_for_directml = True
            except Exception:
                pass
        model.eval()
        return model, converter

    # Preserve originals once.
    if not hasattr(detection, "_videocr_orig_get_detector"):
        detection._videocr_orig_get_detector = detection.get_detector
    if not hasattr(recognition, "_videocr_orig_get_recognizer"):
        recognition._videocr_orig_get_recognizer = recognition.get_recognizer
    if not hasattr(easyocr_core.Reader, "_videocr_orig_recognize"):
        easyocr_core.Reader._videocr_orig_recognize = easyocr_core.Reader.recognize

    _ORIG_GET_DETECTOR = detection._videocr_orig_get_detector
    _ORIG_GET_RECOGNIZER = recognition._videocr_orig_get_recognizer
    _ORIG_RECOGNIZE = easyocr_core.Reader._videocr_orig_recognize

    def dml_hybrid_recognize(self: Any, *args: Any, **kwargs: Any):
        """
        Keep EasyOCR text recognition on CPU when Reader.device is DirectML.

        torch-directml currently cannot run EasyOCR's BiLSTM recognizer reliably;
        it falls through aten::_thnn_fused_lstm_cell and crashes. Detection can
        still use DirectML, then recognition runs on CPU with normal PyTorch.
        """
        current_device = getattr(self, "device", "cpu")
        if not _is_directml_device(current_device):
            return _ORIG_RECOGNIZE(self, *args, **kwargs)

        # Stable mode intentionally forces recognition to CPU. Auto and
        # experimental mode try DirectML recognition first so newer DirectML/
        # PyTorch stacks can use more of the GPU.
        if get_directml_recognition_mode() in {"auto", "experimental"}:
            return _ORIG_RECOGNIZE(self, *args, **kwargs)

        old_device = current_device
        old_recognizer = getattr(self, "recognizer", None)
        try:
            self.device = "cpu"
            if old_recognizer is not None:
                self.recognizer = old_recognizer.to("cpu")
            return _ORIG_RECOGNIZE(self, *args, **kwargs)
        finally:
            self.device = old_device
            # Keep the recognizer on CPU. Moving it back to DirectML would just
            # re-trigger the unsupported LSTM path on the next subtitle frame.

    detection.get_detector = dml_get_detector
    recognition.get_recognizer = dml_get_recognizer

    # Reader.__init__ imports these into easyocr.easyocr / instance attributes,
    # so patch both module locations.
    easyocr_core.get_recognizer = dml_get_recognizer
    easyocr_core.Reader.recognize = dml_hybrid_recognize

    _DML_PATCH_APPLIED = True


def create_easyocr_reader(lang: str, use_gpu: bool) -> Any:
    """Create an EasyOCR reader using DirectML when requested."""
    try:
        import easyocr  # type: ignore
    except Exception as e:
        import traceback

        original_error = "".join(traceback.format_exception_only(type(e), e)).strip()
        raise RuntimeError(
            "EasyOCR could not be imported. It may be missing, or one of its dependencies "
            "may be incompatible with torch-directml.\n\n"
            f"Original import error:\n{original_error}\n\n"
            "Recommended Windows/AMD fix inside your active .venv:\n"
            "  python -m pip install --force-reinstall \"numpy==1.26.4\" \"opencv-python-headless==4.10.0.84\"\n"
            "  python -m pip install --upgrade --force-reinstall \"easyocr==1.7.2\" \"torch-directml==0.2.5.dev240914\""
        ) from e

    langs = normalize_easyocr_lang(lang)

    if use_gpu:
        device = get_directml_device()
        recognition_mode = get_directml_recognition_mode()
        print(f"Using EasyOCR DirectML device: {device}", flush=True)
        if recognition_mode == "stable":
            print("Using stable hybrid mode: DirectML detector + CPU text recognizer.", flush=True)
        elif recognition_mode == "auto":
            print("Using AMD Max Auto mode: trying DirectML detector + DirectML recognizer, with CPU fallback if needed.", flush=True)
        else:
            print("Using experimental DirectML recognizer mode. CPU fallback will be used for known LSTM compatibility failures.", flush=True)
        try:
            patch_easyocr_for_directml(easyocr)
            # quantize=False avoids CPU-only quantization paths; DirectML runs
            # the plain FP32 detector model. EasyOCR's LSTM recognizer is kept
            # on CPU by the runtime patch above.
            return easyocr.Reader(langs, gpu=device, verbose=False, quantize=False)
        except Exception as e:
            raise RuntimeError(
                "EasyOCR failed to start on DirectML after applying the VideOCR DirectML patch.\n\n"
                "Original startup error:\n"
                f"{_format_exception(e)}\n\n"
                "Try CPU mode once to confirm the OCR models downloaded correctly. If CPU mode works but "
                "DirectML still fails, the remaining issue is likely a torch-directml operator compatibility problem."
            ) from e

    print("Using EasyOCR CPU mode.", flush=True)
    return easyocr.Reader(langs, gpu=False, verbose=False)


def _normalize_box(box: Any) -> list[list[float]]:
    """Normalize EasyOCR's quadrilateral into VideOCR's [[x, y], ...] format."""
    points: list[list[float]] = []
    for point in box:
        if len(point) < 2:
            continue
        points.append([float(point[0]), float(point[1])])

    if len(points) != 4:
        raise ValueError(f"Unexpected EasyOCR box format: {box!r}")

    return points


def run_easyocr_on_stitched_images(
        input_dir: str,
        stitch_map: dict[str, list[dict[str, Any]]],
        lang: str,
        use_gpu: bool,
        directml_recognition_mode: str = "stable") -> dict[tuple[int, int], list[Any]]:
    """
    Run EasyOCR on VideOCR's already-filtered stitched image grids.

    Returns a map keyed by (frame_idx, zone_idx). Each value is compatible with
    PredictedFrames: [[box, (text, confidence)], ...].
    """
    if use_gpu:
        os.environ["VIDEOCR_DIRECTML_RECOGNITION_MODE"] = str(directml_recognition_mode or "stable")

    reader = create_easyocr_reader(lang, use_gpu)
    dml_fallback_done = False

    filenames = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
    )

    outputs: dict[tuple[int, int], list[Any]] = {}
    total = len(filenames)

    if total == 0:
        return outputs

    engine_name = "EasyOCR DirectML hybrid" if use_gpu else "EasyOCR"
    print(f"Starting {engine_name}...", flush=True)
    if use_gpu:
        mode = get_directml_recognition_mode()
        if mode == "stable":
            print("DirectML hybrid note: detector runs on the selected AMD/DirectML adapter; EasyOCR text recognition stays on CPU for LSTM compatibility.", flush=True)
        else:
            print(f"DirectML recognition mode: {mode}. VideOCR will fall back to stable CPU recognition if the DirectML LSTM path is unsupported.", flush=True)

    for index, filename in enumerate(filenames, 1):
        image_path = os.path.join(input_dir, filename)
        mapping = stitch_map.get(filename)
        if not mapping:
            continue

        try:
            results = reader.readtext(image_path, detail=1, paragraph=False)
        except Exception as e:
            mode = get_directml_recognition_mode() if use_gpu else "cpu"
            if (
                use_gpu
                and not dml_fallback_done
                and mode in {"auto", "experimental"}
                and _is_known_directml_recognition_failure(e)
            ):
                print(
                    "\nDirectML recognizer hit the known EasyOCR LSTM compatibility issue. "
                    "Falling back to Stable Hybrid mode: DirectML detector + CPU recognizer.",
                    flush=True,
                )
                os.environ["VIDEOCR_DIRECTML_RECOGNITION_MODE"] = "stable"
                dml_fallback_done = True
                try:
                    reader = create_easyocr_reader(lang, use_gpu)
                    results = reader.readtext(image_path, detail=1, paragraph=False)
                except Exception as retry_error:
                    raise RuntimeError(
                        f"EasyOCR failed while reading {filename} after DirectML recognition fallback:\n"
                        f"Original DirectML recognition error:\n{_format_exception(e)}\n\n"
                        f"Stable Hybrid retry error:\n{_format_exception(retry_error)}"
                    ) from retry_error
            else:
                raise RuntimeError(f"EasyOCR failed while reading {filename}:\n{_format_exception(e)}") from e

        for item in results:
            if len(item) < 3:
                continue

            box_raw, text_raw, conf_raw = item[0], item[1], item[2]
            text = str(text_raw).strip()
            if not text:
                continue

            try:
                box = _normalize_box(box_raw)
                confidence = float(conf_raw)
            except Exception:
                continue

            for adjusted_poly, meta in utils.unstitch_polygon(box, mapping):
                key = (int(meta["frame_idx"]), int(meta["zone_idx"]))
                outputs.setdefault(key, []).append([adjusted_poly, (text, confidence)])

        print(f"\rStep 2/3: Performing Text-Detection on image {index} of {total}", end="", flush=True)

    print()
    return outputs
