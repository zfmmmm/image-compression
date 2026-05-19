from __future__ import annotations

from pathlib import Path
import shutil

from utils.command import run_command
from utils.image_io import convert_to_png

from .base import BaseCodec


class BPGCodec(BaseCodec):
    name = "bpg"
    bitstream_extension = ".bpg"

    def is_available(self) -> bool:
        return shutil.which("bpgenc") is not None and shutil.which("bpgdec") is not None

    def encode(self, input_path: Path, bitstream_path: Path, param: dict) -> None:
        temp_input = bitstream_path.parent / "input_rgb.png"
        convert_to_png(input_path, temp_input)
        quality = int(param.get("quality", 32))
        # BPG q is inverse to JPEG quality: smaller q means higher quality/larger files.
        run_command(["bpgenc", "-q", str(quality), "-o", bitstream_path, temp_input], timeout=self.timeout)

    def decode(self, bitstream_path: Path, recon_path: Path) -> None:
        run_command(["bpgdec", "-o", recon_path, bitstream_path], timeout=self.timeout)

    def candidate_params(self) -> list[dict]:
        return [{"quality": quality} for quality in [20, 24, 28, 32, 36, 40, 44, 48]]
