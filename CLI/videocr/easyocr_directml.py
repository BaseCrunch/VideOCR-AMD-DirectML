from __future__ import annotations

import os
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

    count = _safe_directml_device_count(torch_directml)
    if count is not None:
        # On systems like Ryzen 7800X3D + RX 7900 XTX, adapter 0 is commonly the
        # integrated AMD Radeon Graphics and adapter 1 is the RX 7900 XTX.
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

        # Hybrid DirectML mode:
        # EasyOCR's recognizer uses BiLSTM. On torch-directml 0.2.5 / torch 2.4.1,
        # the LSTM path can fall back to a broken CPU operator path
        # (aten::_thnn_fused_lstm_cell). Keep recognition on real CPU while the
        # detector can still use DirectML. This avoids crashes and still gives
        # AMD users a working GPU-assisted backend instead of a hard failure.
        model = model.to("cpu")
        model.eval()
        try:
            model._videocr_cpu_recognizer_for_directml = True
        except Exception:
            pass
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
        print(f"Using EasyOCR DirectML device: {device}", flush=True)
        print("Using hybrid mode: DirectML detector + CPU text recognizer.", flush=True)
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
        use_gpu: bool) -> dict[tuple[int, int], list[Any]]:
    """
    Run EasyOCR on VideOCR's already-filtered stitched image grids.

    Returns a map keyed by (frame_idx, zone_idx). Each value is compatible with
    PredictedFrames: [[box, (text, confidence)], ...].
    """
    reader = create_easyocr_reader(lang, use_gpu)

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
        print("DirectML hybrid note: detector runs on the selected AMD/DirectML adapter; EasyOCR text recognition stays on CPU for LSTM compatibility.", flush=True)

    for index, filename in enumerate(filenames, 1):
        image_path = os.path.join(input_dir, filename)
        mapping = stitch_map.get(filename)
        if not mapping:
            continue

        try:
            results = reader.readtext(image_path, detail=1, paragraph=False)
        except Exception as e:
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
