from __future__ import annotations

from pathlib import Path

from PIL import Image

from .base import BaseCodec


class JPEGCodec(BaseCodec):
    name = "jpeg"
    bitstream_extension = ".jpg"

    def is_available(self) -> bool:
        return True

    def encode(self, input_path: Path, bitstream_path: Path, param: dict) -> None:
        bitstream_path.parent.mkdir(parents=True, exist_ok=True)
        quality = int(param.get("quality", 80))
        subsampling = int(param.get("subsampling", 0))
        progressive = bool(param.get("progressive", False))
        with Image.open(input_path) as image:
            image.convert("RGB").save(
                bitstream_path,
                format="JPEG",
                quality=quality,
                subsampling=subsampling,
                optimize=True,
                progressive=progressive,
            )

    def decode(self, bitstream_path: Path, recon_path: Path) -> None:
        recon_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(bitstream_path) as image:
            image.convert("RGB").save(recon_path, format="PNG")

    def candidate_params(self) -> list[dict]:
        qualities = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10]
        return [{"quality": quality, "subsampling": 0, "progressive": False} for quality in qualities]
