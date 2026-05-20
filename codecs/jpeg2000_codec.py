from __future__ import annotations

from pathlib import Path
import shutil

from utils.command import run_command
from utils.image_io import convert_to_png

from .base import BaseCodec


class JPEG2000Codec(BaseCodec):
    name = "jpeg2000"
    bitstream_extension = ".jp2"

    def is_available(self) -> bool:
        return shutil.which("opj_compress") is not None and shutil.which("opj_decompress") is not None

    def encode(self, input_path: Path, bitstream_path: Path, param: dict) -> None:
        temp_input = bitstream_path.parent / "input_rgb.png"
        convert_to_png(input_path, temp_input)
        rate = float(param.get("rate", 16))
        run_command(
            ["opj_compress", "-i", temp_input, "-o", bitstream_path, "-r", str(rate)],
            timeout=self.timeout,
        )

    def decode(self, bitstream_path: Path, recon_path: Path) -> None:
        run_command(
            ["opj_decompress", "-i", bitstream_path, "-o", recon_path],
            timeout=self.timeout,
        )

    def candidate_params(self) -> list[dict]:
        return [{"rate": rate} for rate in [8, 10, 12, 14, 16, 18, 20, 24, 28, 32]]
