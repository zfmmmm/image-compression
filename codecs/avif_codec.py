from __future__ import annotations

from pathlib import Path
import logging
import shutil

from utils.command import CommandError, run_command
from utils.image_io import convert_to_png

from .base import BaseCodec


LOGGER = logging.getLogger(__name__)


class AVIFCodec(BaseCodec):
    name = "avif"
    bitstream_extension = ".avif"

    def is_available(self) -> bool:
        return shutil.which("avifenc") is not None and shutil.which("avifdec") is not None

    def encode(self, input_path: Path, bitstream_path: Path, param: dict) -> None:
        temp_input = bitstream_path.parent / "input_rgb.png"
        convert_to_png(input_path, temp_input)
        quality = int(param.get("quality", 60))
        speed = int(param.get("speed", 4))
        use_yuv444 = bool(param.get("yuv444", True))

        # libavif versions differ: newer tools accept --quality, while 1.0.x
        # uses --qcolor/-q. Try the requested yuv444 path first, then fall back.
        yuv_options = [["--yuv", "444"], []] if use_yuv444 else [[]]
        quality_options = [["--quality", str(quality)], ["--qcolor", str(quality)]]
        errors: list[CommandError] = []

        for yuv_option in yuv_options:
            for quality_option in quality_options:
                command = [
                    "avifenc",
                    "--speed",
                    str(speed),
                    *yuv_option,
                    *quality_option,
                    str(temp_input),
                    str(bitstream_path),
                ]
                try:
                    run_command(command, timeout=self.timeout)
                    if use_yuv444 and not yuv_option:
                        LOGGER.warning("avifenc did not accept --yuv 444; encoded with default chroma format")
                    return
                except CommandError as exc:
                    errors.append(exc)
                    LOGGER.debug("avifenc attempt failed: %s", exc)

        raise errors[-1]

    def decode(self, bitstream_path: Path, recon_path: Path) -> None:
        run_command(["avifdec", bitstream_path, recon_path], timeout=self.timeout)

    def candidate_params(self) -> list[dict]:
        qualities = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20]
        return [{"quality": quality, "speed": 4, "yuv444": True} for quality in qualities]
