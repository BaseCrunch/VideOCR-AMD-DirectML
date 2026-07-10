from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

# Allow running this file directly from the source tree.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULES = [
    "numpy",
    "cv2",
    "PIL",
    "torch",
    "torchvision",
    "torch_directml",
    "easyocr",
    "av",
]


def check_module(name: str) -> bool:
    print(f"\n=== {name} ===")
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        file_path = getattr(mod, "__file__", "built-in/namespace")
        print(f"OK: {name}")
        print(f"Version: {version}")
        print(f"File: {file_path}")
        return True
    except Exception:
        print(f"FAILED: {name}")
        traceback.print_exc()
        return False


def main() -> int:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    all_ok = True
    for module in MODULES:
        all_ok = check_module(module) and all_ok

    if not all_ok:
        print("\nOne or more imports failed. Fix those before running VideOCR.")
        return 1

    try:
        import torch
        import torch_directml
        device = torch_directml.device()
        x = torch.tensor([1.0]).to(device)
        print("\nDirectML tensor test:", float((x + 2).cpu().item()))
        print("DirectML device:", device)
    except Exception:
        print("\nDirectML tensor test failed:")
        traceback.print_exc()
        return 2

    try:
        from CLI.videocr.easyocr_directml import create_easyocr_reader
        print("\nCreating patched EasyOCR DirectML reader...")
        reader = create_easyocr_reader("en", True)
        print("Reader OK")
        print("Reader device:", getattr(reader, "device", "unknown"))
    except Exception:
        print("\nPatched EasyOCR DirectML reader failed:")
        traceback.print_exc()
        return 3

    print("\nAll DirectML diagnostics passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
