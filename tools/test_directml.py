from __future__ import annotations

import os
import sys


def main() -> int:
    if sys.platform != "win32":
        print("DirectML is Windows-only. Run this on your Windows PC.")
        return 1

    try:
        import torch  # type: ignore
        import torch_directml  # type: ignore
    except Exception as e:
        print("Missing dependency:", e)
        print("Install with: python -m pip install easyocr torch-directml")
        return 1

    requested = os.environ.get("VIDEOCR_DIRECTML_DEVICE_INDEX", "").strip()
    candidates: list[int | None] = []
    if requested:
        try:
            candidates.append(int(requested))
        except ValueError:
            print("VIDEOCR_DIRECTML_DEVICE_INDEX must be a number, for example 1.")
            return 1

    # Prefer adapter 1 on Ryzen+iGPU systems, because adapter 0 is often the
    # integrated AMD Radeon Graphics and adapter 1 is the RX dGPU.
    candidates.extend([1, 0, None])

    seen: list[int | None] = []
    for idx in candidates:
        if idx in seen:
            continue
        seen.append(idx)
        try:
            if idx is None:
                device = torch_directml.device()
                label = "default"
            else:
                device = torch_directml.device(idx)
                label = str(idx)

            a = torch.tensor([1.0]).to(device)
            b = torch.tensor([2.0]).to(device)
            result = (a + b).cpu().item()
        except Exception as e:
            print(f"DirectML adapter {idx if idx is not None else 'default'} failed: {e}")
            continue

        print("DirectML test passed.")
        print(f"Selected adapter index: {label}")
        print(f"Device: {device}")
        print(f"Result: {result}")
        return 0 if result == 3.0 else 1

    print("DirectML test failed on all tested adapters.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
