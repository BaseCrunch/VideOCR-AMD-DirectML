from __future__ import annotations

import os
import traceback
from typing import Any

from . import utils


def _format_exception(e: BaseException) -> str:
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))


def _normalize_box(box: Any) -> list[list[float]]:
    points: list[list[float]] = []
    for point in box:
        if len(point) < 2:
            continue
        points.append([float(point[0]), float(point[1])])
    if len(points) != 4:
        raise ValueError(f"Unexpected ONNX OCR box format: {box!r}")
    return points


def _load_rapidocr_engine() -> tuple[Any | None, str]:
    """Create a RapidOCR ONNXRuntime engine when available.

    RapidOCR packages have changed constructor signatures over time, so this
    intentionally tries the lowest-common-denominator path first. The DirectML
    provider is selected through onnxruntime when the installed package exposes
    it; otherwise this returns a reason and VideOCR falls back to EasyOCR
    DirectML Hybrid instead of crashing the job.
    """
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        return None, f"onnxruntime-directml is not importable: {e}"

    providers = []
    try:
        providers = list(ort.get_available_providers())
    except Exception:
        providers = []

    if "DmlExecutionProvider" not in providers:
        return None, f"ONNX Runtime is installed, but DmlExecutionProvider is not available. Providers: {providers}"

    requested_index = os.environ.get("VIDEOCR_DIRECTML_DEVICE_INDEX", "").strip()
    if requested_index:
        # ORT DirectML uses this environment variable on many builds.
        os.environ["ORT_DML_DEVICE_ID"] = requested_index

    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except Exception as e:
        return None, f"rapidocr-onnxruntime is not importable: {e}"

    last_error: BaseException | None = None
    for kwargs in (
        {"providers": ["DmlExecutionProvider", "CPUExecutionProvider"]},
        {},
    ):
        try:
            return RapidOCR(**kwargs), f"ONNX Runtime providers: {providers}"
        except TypeError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue

    reason = _format_exception(last_error) if last_error else "unknown RapidOCR initialization error"
    return None, f"RapidOCR ONNXRuntime could not initialize: {reason}"


def _run_rapidocr_on_grid(engine: Any, image_path: str) -> list[tuple[list[list[float]], str, float]]:
    raw = engine(image_path)

    # Common RapidOCR return shapes:
    #   (result, elapse) where result is list[[box, text, score], ...]
    #   result directly as list[[box, text, score], ...]
    if isinstance(raw, tuple):
        result = raw[0]
    else:
        result = raw

    if result is None:
        return []

    normalized: list[tuple[list[list[float]], str, float]] = []
    for item in result:
        if not item or len(item) < 3:
            continue
        try:
            box = _normalize_box(item[0])
            text = str(item[1]).strip()
            conf = float(item[2])
        except Exception:
            continue
        if text:
            normalized.append((box, text, conf))
    return normalized


def run_onnx_directml_on_stitched_images(
        input_dir: str,
        stitch_map: dict[str, list[dict[str, Any]]],
        lang: str,
        use_gpu: bool,
        directml_recognition_mode: str = "stable") -> dict[tuple[int, int], list[Any]]:
    """Experimental ONNXRuntime DirectML OCR pass.

    When RapidOCR + ONNXRuntime DirectML are available, this reads the already
    stitched VideOCR grids with ONNXRuntime. If the optional ONNX stack is not
    available or cannot initialize on DirectML, it falls back to the proven
    EasyOCR DirectML Hybrid backend so the run still finishes.
    """
    filenames = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
    )
    outputs: dict[tuple[int, int], list[Any]] = {}
    total = len(filenames)
    if total == 0:
        return outputs

    if not use_gpu:
        print("ONNX Runtime DirectML OCR was selected, but GPU usage is disabled. Falling back to EasyOCR CPU/Hybrid path.", flush=True)
        from .easyocr_directml import run_easyocr_on_stitched_images
        return run_easyocr_on_stitched_images(input_dir, stitch_map, lang, use_gpu, directml_recognition_mode)

    print("Starting ONNX Runtime DirectML OCR (experimental)...", flush=True)
    engine, reason = _load_rapidocr_engine()
    print(f"ONNX DirectML status: {reason}", flush=True)

    if engine is None:
        print(
            "ONNX DirectML OCR is not ready on this install. Falling back to EasyOCR DirectML Hybrid for this run.",
            flush=True,
        )
        from .easyocr_directml import run_easyocr_on_stitched_images
        return run_easyocr_on_stitched_images(input_dir, stitch_map, lang, use_gpu, directml_recognition_mode)

    try:
        for index, filename in enumerate(filenames, 1):
            image_path = os.path.join(input_dir, filename)
            mapping = stitch_map.get(filename)
            if not mapping:
                continue

            for box, text, confidence in _run_rapidocr_on_grid(engine, image_path):
                for adjusted_poly, meta in utils.unstitch_polygon(box, mapping):
                    key = (int(meta["frame_idx"]), int(meta["zone_idx"]))
                    outputs.setdefault(key, []).append([adjusted_poly, (text, confidence)])

            print(f"\rStep 2/3: Performing ONNX DirectML OCR on image {index} of {total}", end="", flush=True)
        print()
        return outputs
    except Exception as e:
        print(
            "\nONNX DirectML OCR failed during processing. Falling back to EasyOCR DirectML Hybrid.\n"
            f"ONNX error:\n{_format_exception(e)}",
            flush=True,
        )
        from .easyocr_directml import run_easyocr_on_stitched_images
        return run_easyocr_on_stitched_images(input_dir, stitch_map, lang, use_gpu, directml_recognition_mode)
