#!/usr/bin/env python3
"""Benchmark AMD DirectML OCR presets on a short sample and rank them.

Example:
  python tools/benchmark_amd_presets.py "C:\\Videos\\EP86S.mp4" \
      --crop 0,793,1920,287 --seconds 180 --device-index 1

This tool intentionally runs the existing VideOCR CLI multiple times instead of
adding another hidden code path. The generated SRT files are temporary benchmark
outputs. Use the fastest preset from the printed ranking in the GUI. If this tool is launched with system Python, it automatically prefers the repo .venv Python for the CLI subprocesses when available.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


BENCH_RE = re.compile(
    r"\[Bench\]\s+Video duration:\s+(?P<video>[0-9.]+)s;\s+"
    r"OCR runtime:\s+(?P<ocr>[0-9.]+)s;\s+total runtime:\s+(?P<total>[0-9.]+)s;\s+"
    r"speed:\s+(?P<speed>[0-9.]+)x"
)
STEP1_RE = re.compile(r"\[Perf\]\s+Step 1 .*?:\s+(?P<step1>[0-9.]+)s")
STEP2_RE = re.compile(r"\[Perf\]\s+Step 2 .*?:\s+(?P<step2>[0-9.]+)s")
GRID_RE = re.compile(r"\[Perf\]\s+Filtered OCR frames:\s+(?P<frames>\d+);\s+stitched grids:\s+(?P<grids>\d+);")


@dataclass
class Preset:
    name: str
    ocr_engine: str
    onnx_tuning: str = "balanced"
    recognition: str = "auto"
    frame_scan: str = "ffmpeg_d3d11va"
    directml_preset: str = "max"
    grid_width: int = 2048
    grid_height: int = 2048


@dataclass
class Result:
    name: str
    ok: bool
    speed_x: float = 0.0
    total_sec: float = 0.0
    ocr_sec: float = 0.0
    step1_sec: float = 0.0
    step2_sec: float = 0.0
    grids: int = 0
    frames: int = 0
    output_srt: str = ""
    error: str = ""


def format_time(seconds: int) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def parse_crop(value: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in value.replace(":", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Crop must be x,y,width,height, for example 0,793,1920,287")
    try:
        x, y, w, h = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Crop values must be integers") from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("Crop width and height must be positive")
    return x, y, w, h




def resolve_cli_python(repo_root: Path) -> Path:
    """Use the active interpreter, but prefer the repo .venv when launched from system Python.

    This avoids accidentally running the benchmark subprocesses with Python 3.13
    when the DirectML dependencies were installed into .venv / Python 3.12.
    """
    current = Path(sys.executable)
    if getattr(sys, "base_prefix", sys.prefix) != sys.prefix:
        return current

    candidates = [
        repo_root / ".venv" / "Scripts" / "python.exe",
        repo_root / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return current


def default_presets() -> list[Preset]:
    return [
        Preset("ONNX Balanced + AMD Max Auto", "onnx_directml", "balanced", "auto", "ffmpeg_d3d11va", "max", 2048, 2048),
        Preset("ONNX Balanced + Stable Hybrid", "onnx_directml", "balanced", "stable", "ffmpeg_d3d11va", "max", 2048, 2048),
        Preset("ONNX Low VRAM + AMD Max Auto", "onnx_directml", "low_vram", "auto", "ffmpeg_d3d11va", "max", 1600, 1600),
        Preset("ONNX Max Throughput + AMD Max Auto", "onnx_directml", "max", "auto", "ffmpeg_d3d11va", "max", 3072, 3072),
        Preset("EasyOCR DirectML Hybrid", "easyocr_directml", "balanced", "stable", "ffmpeg_d3d11va", "max", 4096, 4096),
    ]


def build_common_args(args: argparse.Namespace, crop: tuple[int, int, int, int], output: Path, preset: Preset) -> list[str]:
    x, y, w, h = crop
    cmd = [
        str(args.python_executable),
        str(args.cli_path),
        "--video_path", str(args.video_path),
        "--output", str(output),
        "--ocr_engine", preset.ocr_engine,
        "--lang", args.lang,
        "--time_start", args.time_start,
        "--time_end", format_time(args.seconds),
        "--conf_threshold", str(args.conf_threshold),
        "--sim_threshold", str(args.sim_threshold),
        "--max_merge_gap", str(args.max_merge_gap),
        "--use_fullframe", "false",
        "--use_gpu", "true",
        "--use_angle_cls", "false",
        "--use_server_model", "false",
        "--ssim_threshold", str(args.ssim_threshold),
        "--subtitle_position", args.subtitle_position,
        "--frames_to_skip", str(args.frames_to_skip),
        "--normalize_to_simplified_chinese", str(args.normalize_to_simplified_chinese).lower(),
        "--post_processing", "false",
        "--min_subtitle_duration", str(args.min_subtitle_duration),
        "--ocr_image_max_width", str(args.ocr_image_max_width),
        "--directml_grid_max_width", str(preset.grid_width),
        "--directml_grid_max_height", str(preset.grid_height),
        "--directml_performance_preset", preset.directml_preset,
        "--directml_recognition_mode", preset.recognition,
        "--directml_frame_scan_mode", preset.frame_scan,
        "--onnx_directml_tuning", preset.onnx_tuning,
        "--benchmark_compare_engine", "false",
        "--benchmark_compare_sample_grids", "3",
        "--crop_x", str(x),
        "--crop_y", str(y),
        "--crop_width", str(w),
        "--crop_height", str(h),
        "--allow_system_sleep", "true",
    ]
    if args.device_index is not None:
        cmd.extend(["--directml_device_index", str(args.device_index)])
    return cmd


def parse_result(name: str, output_srt: Path, stdout: str, stderr: str, returncode: int) -> Result:
    if returncode != 0:
        err = (stderr.strip() or stdout.strip()).splitlines()[-12:]
        return Result(name=name, ok=False, output_srt=str(output_srt), error="\n".join(err))

    bench = BENCH_RE.search(stdout)
    if not bench:
        return Result(name=name, ok=False, output_srt=str(output_srt), error="No [Bench] line found in CLI output.")

    step1 = STEP1_RE.search(stdout)
    step2 = STEP2_RE.search(stdout)
    grid = GRID_RE.search(stdout)
    return Result(
        name=name,
        ok=True,
        speed_x=float(bench.group("speed")),
        total_sec=float(bench.group("total")),
        ocr_sec=float(bench.group("ocr")),
        step1_sec=float(step1.group("step1")) if step1 else 0.0,
        step2_sec=float(step2.group("step2")) if step2 else 0.0,
        grids=int(grid.group("grids")) if grid else 0,
        frames=int(grid.group("frames")) if grid else 0,
        output_srt=str(output_srt),
    )


def run_preset(args: argparse.Namespace, preset: Preset, crop: tuple[int, int, int, int], index: int) -> Result:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", preset.name).strip("_")
    output = args.output_dir / f"benchmark_{index:02d}_{safe_name}.srt"
    cmd = build_common_args(args, crop, output, preset)

    print(f"\n=== Benchmark {index}: {preset.name} ===", flush=True)
    print(" ".join(f'\"{p}\"' if " " in str(p) else str(p) for p in cmd), flush=True)
    started = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, cwd=str(args.repo_root))
    elapsed = time.perf_counter() - started
    result = parse_result(preset.name, output, proc.stdout, proc.stderr, proc.returncode)
    if result.ok:
        print(f"Result: {result.speed_x:.2f}x real-time, total {result.total_sec:.2f}s, Step2 {result.step2_sec:.2f}s, grids {result.grids}", flush=True)
    else:
        print(f"FAILED after {elapsed:.2f}s: {result.error}", flush=True)
    if args.keep_logs:
        log_path = args.output_dir / f"benchmark_{index:02d}_{safe_name}.log"
        log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8", errors="replace")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Benchmark VideOCR AMD DirectML presets on a short sample and rank them.")
    parser.add_argument("video_path", type=Path, help="Video file to test")
    parser.add_argument("--crop", type=parse_crop, required=True, help="Crop as x,y,width,height, example: 0,793,1920,287")
    parser.add_argument("--seconds", type=int, default=180, help="Sample length to test from the start, default: 180 seconds")
    parser.add_argument("--time-start", default="0:00", help="Sample start time, default: 0:00")
    parser.add_argument("--device-index", type=int, default=1, help="DirectML adapter index, default: 1 for RX 7900 XTX on Ryzen+iGPU desktops")
    parser.add_argument("--lang", default="en", help="OCR language, default: en")
    parser.add_argument("--subtitle-position", default="center", choices=["left", "center", "right", "any"], help="Subtitle position hint")
    parser.add_argument("--frames-to-skip", type=int, default=1)
    parser.add_argument("--ocr-image-max-width", type=int, default=720)
    parser.add_argument("--conf-threshold", type=int, default=75)
    parser.add_argument("--sim-threshold", type=int, default=80)
    parser.add_argument("--ssim-threshold", type=int, default=92)
    parser.add_argument("--max-merge-gap", type=float, default=0.1)
    parser.add_argument("--min-subtitle-duration", type=float, default=0.2)
    parser.add_argument("--normalize-to-simplified-chinese", action="store_true", default=False)
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for benchmark SRT/log outputs")
    parser.add_argument("--keep-logs", action="store_true", help="Write full stdout/stderr logs for each preset")
    parser.add_argument("--skip-easyocr", action="store_true", help="Only test ONNX presets")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON result path")
    args = parser.parse_args(list(argv) if argv is not None else None)

    args.repo_root = repo_root
    args.cli_path = repo_root / "CLI" / "videocr_cli.py"
    args.python_executable = resolve_cli_python(repo_root)
    args.video_path = args.video_path.resolve()
    if not args.video_path.exists():
        parser.error(f"Video file not found: {args.video_path}")
    if not args.cli_path.exists():
        parser.error(f"CLI not found: {args.cli_path}")
    if args.output_dir is None:
        args.output_dir = Path(tempfile.mkdtemp(prefix="videocr_amd_bench_"))
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    presets = default_presets()
    if args.skip_easyocr:
        presets = [p for p in presets if p.ocr_engine == "onnx_directml"]

    print(f"VideOCR AMD preset benchmark")
    print(f"Video: {args.video_path}")
    print(f"Crop: {args.crop}")
    print(f"Sample: {args.time_start} -> {format_time(args.seconds)}")
    print(f"Output dir: {args.output_dir}")
    print(f"CLI Python: {args.python_executable}")

    results = [run_preset(args, preset, args.crop, i + 1) for i, preset in enumerate(presets)]
    ok_results = sorted([r for r in results if r.ok], key=lambda r: r.total_sec)

    print("\n=== Ranking ===")
    if not ok_results:
        print("No preset completed successfully.")
    else:
        for rank, result in enumerate(ok_results, 1):
            print(
                f"#{rank}: {result.name} | {result.total_sec:.2f}s | {result.speed_x:.2f}x | "
                f"Step1 {result.step1_sec:.2f}s | Step2 {result.step2_sec:.2f}s | grids {result.grids}"
            )
        best = ok_results[0]
        print("\nRecommended settings:")
        print(f"  Preset: {best.name}")
        print("  OCR Engine: ONNX Runtime DirectML (AMD GPU Experimental)" if "ONNX" in best.name else "  OCR Engine: EasyOCR DirectML (AMD GPU)")
        print("  AMD Frame Scan Mode: AMD FFmpeg D3D11VA Decode + DirectML SSIM")
        print("  DirectML GPU: GPU 1: AMD Radeon RX 7900 XTX")

    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
        print(f"\nWrote JSON results: {args.json}")
    return 0 if ok_results else 1


if __name__ == "__main__":
    raise SystemExit(main())
