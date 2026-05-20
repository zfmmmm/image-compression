from __future__ import annotations

from pathlib import Path
import math

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from utils.image_io import load_image_rgb as _load_image_rgb


def load_image_rgb(path: Path) -> np.ndarray:
    return _load_image_rgb(path)


def _validate_pair(original: np.ndarray, reconstructed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Image shapes differ: original={original.shape}, reconstructed={reconstructed.shape}"
        )
    if original.ndim != 3 or original.shape[2] != 3:
        raise ValueError(f"Expected RGB images with shape HxWx3, got {original.shape}")
    return original.astype(np.float64), reconstructed.astype(np.float64)


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    original_f, reconstructed_f = _validate_pair(original, reconstructed)
    mse = float(np.mean((original_f - reconstructed_f) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def compute_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Image shapes differ: original={original.shape}, reconstructed={reconstructed.shape}"
        )
    min_dim = min(original.shape[:2])
    kwargs: dict[str, object] = {
        "channel_axis": -1,
        "data_range": 255,
    }
    if min_dim < 7:
        win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
        if win_size < 3:
            return 1.0 if np.array_equal(original, reconstructed) else 0.0
        kwargs["win_size"] = win_size
    value = structural_similarity(original, reconstructed, **kwargs)
    return float(np.clip(value, 0.0, 1.0))


def compute_ms_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float | None:
    """MS-SSIM is optional in this traditional-codec baseline."""

    return None


def compute_edge_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Image shapes differ: original={original.shape}, reconstructed={reconstructed.shape}"
        )
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=80, threshold2=160)
    mask = edges > 0
    if not np.any(mask):
        return compute_psnr(original, reconstructed)

    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    edge_diff = diff[mask, :]
    mse = float(np.mean(edge_diff**2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def original_theoretical_size_bytes(original: np.ndarray) -> int:
    if original.ndim != 3:
        raise ValueError(f"Expected RGB image, got shape {original.shape}")
    return int(original.shape[0] * original.shape[1] * original.shape[2])


def compute_compression_ratio(original: np.ndarray, bitstream_size: int) -> float:
    if bitstream_size <= 0:
        return 0.0
    return original_theoretical_size_bytes(original) / float(bitstream_size)


def compute_bpp(original: np.ndarray, bitstream_size: int) -> float:
    if bitstream_size <= 0:
        return float("inf")
    height, width = original.shape[:2]
    return float(bitstream_size * 8) / float(height * width)
