from __future__ import annotations

from typing import Any

import numpy as np


class DirectMLFrameSimilarityFilter:
    """GPU-assisted frame similarity filter for AMD/DirectML runs.

    This does not decode video on the GPU. PyAV still decodes frames and crops/scales
    the subtitle region, but the expensive per-frame similarity comparison can be
    moved to the selected DirectML adapter. It uses a global SSIM approximation in
    PyTorch/DirectML so it can keep the same threshold semantics as VideOCR's normal
    CPU SSIM filter.
    """

    def __init__(self, threshold_ratio: float, device_index: int | None = None) -> None:
        try:
            import torch  # type: ignore
            import torch_directml  # type: ignore
        except Exception as e:  # pragma: no cover - depends on Windows DirectML install
            raise RuntimeError(
                "DirectML frame scan mode requires torch-directml. Install the DirectML extra first."
            ) from e

        self.torch = torch
        self.threshold_ratio = float(threshold_ratio)
        self.device_index = device_index

        if device_index is None:
            self.device = torch_directml.device()
            self.device_label = "default"
        else:
            self.device = torch_directml.device(int(device_index))
            self.device_label = str(device_index)

        # Tiny sanity check so failures happen before the long video scan starts.
        x = torch.tensor([1.0]).to(self.device)
        _ = (x + 1).cpu().item()

        self.comparisons = 0
        self.filtered = 0

    def _to_gray_tensor(self, sample: np.ndarray[Any, Any]) -> Any:
        torch = self.torch
        arr = np.ascontiguousarray(sample)
        tensor = torch.from_numpy(arr).to(dtype=torch.float32)
        if tensor.ndim == 3 and tensor.shape[-1] >= 3:
            # RGB luminance approximation. Move after tensor creation so the main
            # per-pixel math can run on the DirectML adapter.
            tensor = tensor.to(self.device)
            tensor = (tensor[..., 0] * 0.299 + tensor[..., 1] * 0.587 + tensor[..., 2] * 0.114) / 255.0
        else:
            tensor = tensor.to(self.device) / 255.0
        return tensor

    def ssim(self, previous: np.ndarray[Any, Any], current: np.ndarray[Any, Any]) -> float:
        """Return a GPU-computed global SSIM-style score in [roughly 0, 1]."""
        torch = self.torch
        x = self._to_gray_tensor(previous)
        y = self._to_gray_tensor(current)

        # Shape mismatches should not normally happen, but if they do, treat as
        # different so the frame is kept instead of accidentally discarded.
        if tuple(x.shape) != tuple(y.shape):
            return 0.0

        mu_x = torch.mean(x)
        mu_y = torch.mean(y)
        dx = x - mu_x
        dy = y - mu_y
        var_x = torch.mean(dx * dx)
        var_y = torch.mean(dy * dy)
        cov_xy = torch.mean(dx * dy)

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        score = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2))
        self.comparisons += 1
        return float(score.detach().cpu().item())

    def is_similar(self, previous: np.ndarray[Any, Any], current: np.ndarray[Any, Any]) -> bool:
        similar = self.ssim(previous, current) > self.threshold_ratio
        if similar:
            self.filtered += 1
        return similar
