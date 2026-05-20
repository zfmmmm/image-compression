from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)

SUPPORTED_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected grayscale/RGB/RGBA image, got shape {image.shape}")

    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)

    image_f = image.astype(np.float32)
    min_value = float(np.nanmin(image_f))
    max_value = float(np.nanmax(image_f))
    if max_value <= min_value:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (image_f - min_value) * (255.0 / (max_value - min_value))
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def load_image_rgb(path: Path) -> np.ndarray:
    """Load an image as RGB uint8, dropping alpha if present."""

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def save_image_rgb(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ensure_rgb_uint8(image), mode="RGB").save(path)


def convert_to_png(input_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as image:
        image.convert("RGB").save(output_path, format="PNG")
    return output_path


def iter_image_paths(input_dir: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    paths = [
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    return sorted(paths)


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)
