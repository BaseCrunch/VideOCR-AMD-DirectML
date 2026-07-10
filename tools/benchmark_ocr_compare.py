from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


BENCH_RE = re.compile(r"\[Bench\]\s*(.*)")


def parse_crop(crop: str) -> dict[str, str]:
    # Accept x:y:w:h because it is easy to paste from the GUI crop box.
    parts = crop.replace(",", ":").split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Crop must be x:y:w:h, for example 0:793:1920:287")
    x, y, w, h = [p.strip() for p in parts]
    return {"crop_x": x, "crop_y": y, "crop_width": w, "crop_height": h}


def run_engine(args: argparse.Namespace, engine: str, out_dir: Path) -> tuple[int, str, str]:
    root = Path(__file__).resolve().parents[1]
    cli = root / "CLI" / "videocr_cli.py"
    output = out_dir / f"benchmark_{engine}.srt"
    crop = parse_crop(args.crop) if args.crop else {}

    cmd = [
        sys.executable,
        str(cli),
        "--video_path", args.video,
        "--output", str(output),
        "--ocr_engine", engine,
        "--lang", args.lang,
        "--use_gpu", "true",
        "--subtitle_position", args.subtitle_position,
        "--frames_to_skip", str(args.frames_to_skip),
        "--ssim_threshold", str(args.ssim_threshold),
        "--ocr_image_max_width", str(args.ocr_image_max_width),
        "--directml_device_index", str(args.directml_device_index),
        "--directml_performance_preset", args.directml_performance_preset,
        "--directml_recognition_mode", "stable",
        "--directml_frame_scan_mode", args.directml_frame_scan_mode,
        "--onnx_directml_tuning", args.onnx_directml_tuning,
        "--benchmark_compare_engine", "false",
    ]

    if args.time_start:
        cmd += ["--time_start", args.time_start]
    if args.time_end:
        cmd += ["--time_end", args.time_end]
    for key, value in crop.items():
        cmd += [f"--{key}", value]

    print(f"\n=== Running {engine} ===", flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    bench_line = ""
    for line in combined.splitlines():
        if line.startswith("[Perf]") or line.startswith("[Bench]") or line.startswith("[BenchCompare]"):
            print(line, flush=True)
        m = BENCH_RE.search(line)
        if m:
            bench_line = line.strip()
    return proc.returncode, bench_line, combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare EasyOCR DirectML Hybrid vs ONNX Runtime DirectML on the same video/settings.")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--crop", default="", help="Crop as x:y:w:h, for example 0:793:1920:287")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--subtitle-position", default="center")
    parser.add_argument("--frames-to-skip", type=int, default=1)
    parser.add_argument("--ssim-threshold", type=int, default=92)
    parser.add_argument("--ocr-image-max-width", type=int, default=720)
    parser.add_argument("--directml-device-index", type=int, default=1)
    parser.add_argument("--directml-performance-preset", default="max", choices=["compatibility", "balanced", "max", "manual"])
    parser.add_argument("--directml-frame-scan-mode", default="ffmpeg_d3d11va", choices=["cpu_ssim", "directml_ssim", "ffmpeg_d3d11va"])
    parser.add_argument("--onnx-directml-tuning", default="balanced", choices=["low_vram", "balanced", "max", "manual"])
    parser.add_argument("--time-start", default="")
    parser.add_argument("--time-end", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="videocr_ocr_compare_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Benchmark output dir: {out_dir}")

    results = []
    for engine in ("easyocr_directml", "onnx_directml"):
        code, bench, full = run_engine(args, engine, out_dir)
        results.append((engine, code, bench))
        if code != 0:
            log_path = out_dir / f"benchmark_{engine}.log"
            log_path.write_text(full, encoding="utf-8", errors="replace")
            print(f"{engine} failed with exit code {code}. Full log: {log_path}")

    print("\n=== Summary ===")
    for engine, code, bench in results:
        status = "OK" if code == 0 else f"FAILED({code})"
        print(f"{engine}: {status}; {bench or 'no [Bench] line captured'}")
    return 0 if all(code == 0 for _, code, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
