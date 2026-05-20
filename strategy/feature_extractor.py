from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from utils.image_io import ensure_rgb_uint8


@dataclass(slots=True)
class ImageFeatures:
    width: int
    height: int
    channels: int
    entropy: float
    edge_density: float
    gradient_mean: float
    gradient_std: float
    colorfulness: float
    local_variance_mean: float
    estimated_noise: float


def _entropy(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    prob = hist / max(1.0, float(hist.sum()))
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def _colorfulness(image: np.ndarray) -> float:
    rgb = image.astype(np.float32)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    rg = red - green
    yb = 0.5 * (red + green) - blue
    std_root = math.sqrt(float(np.std(rg) ** 2 + np.std(yb) ** 2))
    mean_root = math.sqrt(float(np.mean(rg) ** 2 + np.mean(yb) ** 2))
    return float(std_root + 0.3 * mean_root)


def extract_features(image: np.ndarray) -> ImageFeatures:
    rgb = ensure_rgb_uint8(image)
    height, width, channels = rgb.shape
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, threshold1=80, threshold2=160)
    edge_density = float(np.mean(edges > 0))

    gray_f = gray.astype(np.float32)
    grad_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)

    mean = cv2.blur(gray_f, ksize=(7, 7))
    mean_sq = cv2.blur(gray_f * gray_f, ksize=(7, 7))
    local_variance = np.maximum(mean_sq - mean * mean, 0.0)

    laplacian = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)

    return ImageFeatures(
        width=int(width),
        height=int(height),
        channels=int(channels),
        entropy=_entropy(gray),
        edge_density=edge_density,
        gradient_mean=float(np.mean(magnitude)),
        gradient_std=float(np.std(magnitude)),
        colorfulness=_colorfulness(rgb),
        local_variance_mean=float(np.mean(local_variance)),
        estimated_noise=float(np.mean(np.abs(laplacian))),
    )
