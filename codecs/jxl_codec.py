from __future__ import annotations

from pathlib import Path
import shutil

from utils.command import run_command
from utils.image_io import convert_to_png

from .base import BaseCodec


class JXLCodec(BaseCodec):
    name = "jxl"
    bitstream_extension = ".jxl"

    def is_available(self) -> bool:
        return shutil.which("cjxl") is not None and shutil.which("djxl") is not None

    def encode(self, input_path: Path, bitstream_path: Path, param: dict) -> None:
        temp_input = bitstream_path.parent / "input_rgb.png"
        convert_to_png(input_path, temp_input)
        distance = float(param.get("distance", 1.5))
        effort = int(param.get("effort", 7))
        run_command(
            [
                "cjxl",
                temp_input,
                bitstream_path,
                "--distance",
                str(distance),
                "--effort",
                str(effort),
            ],
            timeout=self.timeout,
        )

    def decode(self, bitstream_path: Path, recon_path: Path) -> None:
        run_command(["djxl", bitstream_path, recon_path], timeout=self.timeout)

    def candidate_params(self) -> list[dict]:
        distances = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0]
        return [{"distance": distance, "effort": 7} for distance in distances]
