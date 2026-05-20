from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import logging
import time

from metrics.image_metrics import (
    compute_bpp,
    compute_compression_ratio,
    compute_edge_psnr,
    compute_ms_ssim,
    compute_psnr,
    compute_ssim,
    load_image_rgb,
    original_theoretical_size_bytes,
)


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CodecResult:
    codec_name: str
    input_path: Path
    bitstream_path: Path
    recon_path: Path
    param: dict
    original_theoretical_size: int
    compressed_size: int
    compression_ratio: float
    bpp: float
    psnr: float
    ssim: float
    ms_ssim: float | None
    edge_psnr: float | None
    encode_time_ms: float
    decode_time_ms: float
    success: bool
    error_message: str | None
    passed: bool = False
    warning_message: str | None = None


class BaseCodec(ABC):
    name: str = "base"
    bitstream_extension: str = ".bin"

    def __init__(self, *, timeout: float = 600.0):
        self.timeout = timeout

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def encode(self, input_path: Path, bitstream_path: Path, param: dict) -> None:
        pass

    @abstractmethod
    def decode(self, bitstream_path: Path, recon_path: Path) -> None:
        pass

    @abstractmethod
    def candidate_params(self) -> list[dict]:
        pass

    def compress_and_eval(
        self,
        input_path: Path,
        work_dir: Path,
        param: dict,
    ) -> CodecResult:
        work_dir.mkdir(parents=True, exist_ok=True)
        bitstream_path = work_dir / f"compressed{self.bitstream_extension}"
        recon_path = work_dir / "recon.png"
        encode_time_ms = 0.0
        decode_time_ms = 0.0
        original_size = 0
        compressed_size = 0

        try:
            if not self.is_available():
                raise RuntimeError(f"Codec '{self.name}' is not available")

            original = load_image_rgb(input_path)
            original_size = original_theoretical_size_bytes(original)

            start = time.perf_counter()
            self.encode(input_path, bitstream_path, param)
            encode_time_ms = (time.perf_counter() - start) * 1000.0

            if not bitstream_path.exists():
                raise FileNotFoundError(f"Encoder did not create bitstream: {bitstream_path}")
            compressed_size = bitstream_path.stat().st_size
            if compressed_size <= 0:
                raise RuntimeError(f"Encoder created an empty bitstream: {bitstream_path}")

            start = time.perf_counter()
            self.decode(bitstream_path, recon_path)
            decode_time_ms = (time.perf_counter() - start) * 1000.0
            if not recon_path.exists():
                raise FileNotFoundError(f"Decoder did not create reconstruction: {recon_path}")

            reconstructed = load_image_rgb(recon_path)
            if reconstructed.shape != original.shape:
                raise ValueError(
                    f"Reconstruction shape {reconstructed.shape} differs from original {original.shape}"
                )

            return CodecResult(
                codec_name=self.name,
                input_path=input_path,
                bitstream_path=bitstream_path,
                recon_path=recon_path,
                param=dict(param),
                original_theoretical_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compute_compression_ratio(original, compressed_size),
                bpp=compute_bpp(original, compressed_size),
                psnr=compute_psnr(original, reconstructed),
                ssim=compute_ssim(original, reconstructed),
                ms_ssim=compute_ms_ssim(original, reconstructed),
                edge_psnr=compute_edge_psnr(original, reconstructed),
                encode_time_ms=encode_time_ms,
                decode_time_ms=decode_time_ms,
                success=True,
                error_message=None,
            )
        except Exception as exc:
            LOGGER.warning("%s failed on %s with param %s: %s", self.name, input_path, param, exc)
            if original_size == 0:
                try:
                    original_size = original_theoretical_size_bytes(load_image_rgb(input_path))
                except Exception:
                    original_size = 0
            if bitstream_path.exists():
                compressed_size = bitstream_path.stat().st_size
            return CodecResult(
                codec_name=self.name,
                input_path=input_path,
                bitstream_path=bitstream_path,
                recon_path=recon_path,
                param=dict(param),
                original_theoretical_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=0.0,
                bpp=float("inf"),
                psnr=float("-inf"),
                ssim=0.0,
                ms_ssim=None,
                edge_psnr=None,
                encode_time_ms=encode_time_ms,
                decode_time_ms=decode_time_ms,
                success=False,
                error_message=str(exc),
            )
