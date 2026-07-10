from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from CLI.videocr.easyocr_directml import (
        _preferred_high_performance_adapter_index,
        _safe_directml_device_count,
        _windows_video_adapters,
    )

    print("Windows video adapters / likely DirectML order:")
    adapters = _windows_video_adapters()
    if adapters:
        for idx, adapter in enumerate(adapters):
            print(f"  GPU {idx}: {adapter.get('name')} / AdapterRAM={adapter.get('ram')}")
    else:
        print("  No Windows video adapters detected through CIM/WMIC.")

    try:
        import torch_directml  # type: ignore
        count = _safe_directml_device_count(torch_directml)
    except Exception as e:
        print("\ntorch-directml import failed:", e)
        return 1

    print("\ntorch-directml adapter count:", count if count is not None else "unknown")
    print("Preferred high-performance adapter index:", _preferred_high_performance_adapter_index())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
