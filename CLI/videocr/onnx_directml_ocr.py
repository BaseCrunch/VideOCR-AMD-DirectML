from __future__ import annotations

import os
import time
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


def _make_session_options(tuning: str) -> tuple[Any | None, str]:
    """Build conservative ONNX Runtime session options when possible.

    DirectML can reserve a lot of VRAM with very large OCR grids. These options
    do not force true batching, but they reduce extra memory patterns/arenas and
    make the selected tuning visible in the logs. RapidOCR versions vary, so the
    loader tries these options first and then gracefully falls back.
    """
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return None, "session options unavailable"

    tuning = (tuning or "balanced").strip().lower()
    so = ort.SessionOptions()

    try:
        if tuning in ("low_vram", "balanced"):
            so.enable_mem_pattern = False
            so.enable_cpu_mem_arena = False
        if tuning == "low_vram":
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
        elif tuning == "balanced":
            so.intra_op_num_threads = 2
        elif tuning == "max":
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        else:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    except Exception:
        pass

    return so, f"session_options={tuning}"


def _load_rapidocr_engine() -> tuple[Any | None, str]:
    """Create a RapidOCR ONNXRuntime engine when available.

    v14 adds tuning-aware startup, more provider reporting, and safer fallback
    attempts for RapidOCR versions with different constructor signatures.
    """
    tuning = os.environ.get("VIDEOCR_ONNX_DIRECTML_TUNING", "balanced").strip().lower() or "balanced"

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
    provider_options: list[dict[str, Any]] = []
    if requested_index:
        # ORT DirectML uses this environment variable on many builds.
        os.environ["ORT_DML_DEVICE_ID"] = requested_index
        provider_options = [{"device_id": int(requested_index)}, {}] if requested_index.isdigit() else []

    # Keep OpenMP from spinning hard on the CPU while DirectML is the target.
    if tuning in ("low_vram", "balanced"):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except Exception as e:
        return None, f"rapidocr-onnxruntime is not importable: {e}"

    session_options, session_note = _make_session_options(tuning)
    provider_list = ["DmlExecutionProvider", "CPUExecutionProvider"]

    attempts: list[dict[str, Any]] = []
    if session_options is not None and provider_options:
        attempts.append({"providers": provider_list, "provider_options": provider_options, "sess_options": session_options})
        attempts.append({"providers": provider_list, "provider_options": provider_options, "session_options": session_options})
    if session_options is not None:
        attempts.append({"providers": provider_list, "sess_options": session_options})
        attempts.append({"providers": provider_list, "session_options": session_options})
    if provider_options:
        attempts.append({"providers": provider_list, "provider_options": provider_options})
    attempts.append({"providers": provider_list})
    attempts.append({})

    last_error: BaseException | None = None
    last_kwargs: dict[str, Any] | None = None
    for kwargs in attempts:
        try:
            engine = RapidOCR(**kwargs)
            selected = ", ".join(kwargs.keys()) if kwargs else "package defaults"
            return engine, (
                f"ONNX Runtime providers: {providers}; selected provider order: {provider_list}; "
                f"tuning: {tuning}; {session_note}; RapidOCR kwargs: {selected}"
            )
        except TypeError as e:
            last_error = e
            last_kwargs = kwargs
            continue
        except Exception as e:
            last_error = e
            last_kwargs = kwargs
            continue

    reason = _format_exception(last_error) if last_error else "unknown RapidOCR initialization error"
    return None, f"RapidOCR ONNXRuntime could not initialize with tuning={tuning}, last kwargs={last_kwargs}: {reason}"


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
        image_times: list[float] = []
        recognized_lines = 0
        active_images = 0
        for index, filename in enumerate(filenames, 1):
            image_path = os.path.join(input_dir, filename)
            mapping = stitch_map.get(filename)
            if not mapping:
                continue

            img_start = time.perf_counter()
            raw_lines = _run_rapidocr_on_grid(engine, image_path)
            img_elapsed = time.perf_counter() - img_start
            image_times.append(img_elapsed)
            active_images += 1
            recognized_lines += len(raw_lines)

            for box, text, confidence in raw_lines:
                for adjusted_poly, meta in utils.unstitch_polygon(box, mapping):
                    key = (int(meta["frame_idx"]), int(meta["zone_idx"]))
                    outputs.setdefault(key, []).append([adjusted_poly, (text, confidence)])

            print(f"\rStep 2/3: Performing ONNX DirectML OCR on image {index} of {total}", end="", flush=True)
        print()
        if image_times:
            avg_t = sum(image_times) / len(image_times)
            print(
                f"[Perf] ONNX image timing: images={active_images}; avg={avg_t:.2f}s; "
                f"fastest={min(image_times):.2f}s; slowest={max(image_times):.2f}s; recognized lines={recognized_lines}",
                flush=True,
            )
        return outputs
    except Exception as e:
        print(
            "\nONNX DirectML OCR failed during processing. Falling back to EasyOCR DirectML Hybrid.\n"
            f"ONNX error:\n{_format_exception(e)}",
            flush=True,
        )
        from .easyocr_directml import run_easyocr_on_stitched_images
        return run_easyocr_on_stitched_images(input_dir, stitch_map, lang, use_gpu, directml_recognition_mode)
