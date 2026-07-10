from __future__ import annotations

import sys
import time

from . import utils
from .video import Video


def save_subtitles_to_file(
        video_path: str, file_path: str = 'subtitle.srt', ocr_engine: str = 'google_lens', lang: str = 'en',
        time_start: str = '0:00', time_end: str = '', conf_threshold: int = 75, sim_threshold: int = 80, max_merge_gap_sec: float = 0.1,
        use_fullframe: bool = False, use_gpu: bool = False, use_angle_cls: bool = False, use_server_model: bool = False,
        brightness_threshold: int | None = None, ssim_threshold: int = 92, subtitle_position: str = "center", frames_to_skip: int = 1,
        crop_zones: list[dict[str, int]] | None = None, ocr_image_max_width: int = 720, post_processing: bool = False, min_subtitle_duration_sec: float = 0.2,
        normalize_to_simplified_chinese: bool = True, subtitle_alignments: list[str | None] | None = None,
        directml_grid_max_width: int = 2400, directml_grid_max_height: int = 2400,
        directml_performance_preset: str = "balanced", directml_recognition_mode: str = "stable",
        directml_frame_scan_mode: str = "cpu_ssim") -> None:

    total_start = time.perf_counter()

    if crop_zones is None:
        crop_zones = []

    if subtitle_alignments is None:
        subtitle_alignments = [None, None]
    elif len(subtitle_alignments) == 1:
        subtitle_alignments.append(None)

    # PaddleOCR and Chrome Lens are standalone helper executables.
    # EasyOCR DirectML is a pure-Python backend, so it must not require the
    # PaddleOCR CUDA/CPU bundle or the NVIDIA hardware check.
    paddleocr_path = ""
    google_lens_path = ""
    det_model_dir = ""
    rec_model_dir = ""
    cls_model_dir = ""

    if ocr_engine in ("paddleocr", "google_lens"):
        paddleocr_path = utils.find_executable("paddleocr")
        try:
            utils.perform_hardware_check(paddleocr_path, use_gpu)
        except SystemExit as e:
            print(e, flush=True)
            sys.exit(1)

        if ocr_engine == 'paddleocr':
            det_model_dir, rec_model_dir, cls_model_dir = utils.resolve_model_dirs(lang, use_server_model)
        else:
            # For the Text-Detection-Only Pass just the default detection model is needed
            det_model_dir, rec_model_dir, cls_model_dir = utils.resolve_model_dirs('en', use_server_model)

    if ocr_engine == "google_lens":
        google_lens_path = utils.find_executable("chrome-lens")

    v = Video(video_path, paddleocr_path, det_model_dir, rec_model_dir, cls_model_dir, google_lens_path)
    try:
        ocr_start = time.perf_counter()
        v.run_ocr(
            use_gpu, ocr_engine, lang, use_angle_cls, time_start, time_end, conf_threshold,
            use_fullframe, brightness_threshold, ssim_threshold, subtitle_position,
            frames_to_skip, crop_zones, ocr_image_max_width, normalize_to_simplified_chinese,
            directml_grid_max_width, directml_grid_max_height,
            directml_performance_preset, directml_recognition_mode, directml_frame_scan_mode
        )
        ocr_end = time.perf_counter()
    except Exception as e:
        print(f"Error: {e}", flush=True)
        sys.exit(1)
    merge_start = time.perf_counter()
    subtitles = v.get_subtitles(sim_threshold, max_merge_gap_sec, lang, post_processing, min_subtitle_duration_sec, subtitle_alignments)

    with open(file_path, 'w+', encoding='utf-8') as f:
        f.write(subtitles)
    total_end = time.perf_counter()

    merge_write_sec = total_end - merge_start
    ocr_runtime_sec = ocr_end - ocr_start
    total_runtime_sec = total_end - total_start
    print(f"[Perf] Step 3 subtitle merge/write: {merge_write_sec:.2f}s", flush=True)
    print(f"[Perf] End-to-end runtime: {total_runtime_sec:.2f}s", flush=True)
    if getattr(v, 'duration_ms', 0) and total_runtime_sec > 0:
        video_sec = float(v.duration_ms) / 1000.0
        speed_x = video_sec / total_runtime_sec
        print(
            f"[Bench] Video duration: {video_sec:.2f}s; OCR runtime: {ocr_runtime_sec:.2f}s; "
            f"total runtime: {total_runtime_sec:.2f}s; speed: {speed_x:.2f}x real-time; "
            f"engine: {ocr_engine}; frame scan: {directml_frame_scan_mode}; recognition: {directml_recognition_mode}",
            flush=True,
        )
