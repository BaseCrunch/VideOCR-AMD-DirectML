from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    return p.returncode, p.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick FFmpeg D3D11VA decode/crop throughput test for VideOCR AMD mode.")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--crop", default="1920:287:0:793", help="FFmpeg crop as w:h:x:y")
    parser.add_argument("--scale", default="720:-2", help="FFmpeg scale as w:h")
    parser.add_argument("--frames-to-skip", type=int, default=1, help="Match VideOCR frames_to_skip. 1 means select every 2nd frame.")
    parser.add_argument("--seconds", type=float, default=60.0, help="Benchmark only the first N seconds")
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ERROR: ffmpeg was not found on PATH.")
        return 1

    video = str(Path(args.video))
    if not os.path.exists(video):
        print(f"ERROR: video not found: {video}")
        return 1

    modulo = max(1, int(args.frames_to_skip) + 1)
    vf = f"select=not(mod(n\\,{modulo})),crop={args.crop},scale={args.scale}:flags=area,format=rgb24"

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-hwaccel", "d3d11va",
        "-t", str(args.seconds),
        "-i", video,
        "-vf", vf,
        "-vsync", "0",
        "-an", "-sn",
        "-f", "null",
        "-",
    ]

    print("Running FFmpeg D3D11VA decode/crop benchmark...")
    print(" ".join(cmd))
    start = time.perf_counter()
    code, output = run(cmd)
    elapsed = time.perf_counter() - start
    print(output[-4000:])
    print(f"Elapsed: {elapsed:.2f}s for requested {args.seconds:.2f}s input window")
    if elapsed > 0:
        print(f"Approx throughput: {args.seconds / elapsed:.2f}x realtime")
    if code != 0:
        print(f"ffmpeg exited with code {code}")
        return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
