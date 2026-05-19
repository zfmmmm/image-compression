from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from strategy.adaptive_selector import AdaptiveSelector
from utils.codec_registry import build_codecs
from utils.logging_utils import setup_logging


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress one RGB image with adaptive traditional codecs.")
    parser.add_argument("--input", required=True, type=Path, help="Input image path")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--mode",
        default="ratio_first",
        choices=["ratio_first", "quality_first", "remote_sensing"],
        help="Adaptive selection mode",
    )
    parser.add_argument("--target-ratio", default=16.0, type=float, help="Target compression ratio")
    parser.add_argument("--min-psnr", default=35.0, type=float, help="Minimum PSNR in dB")
    parser.add_argument(
        "--codecs",
        default="jpeg,jxl,avif,jpeg2000",
        help="Comma-separated codec names: jpeg,jxl,avif,jpeg2000,bpg",
    )
    parser.add_argument(
        "--search",
        default="exhaustive",
        choices=["exhaustive", "coarse_to_fine"],
        help="Parameter search mode. coarse_to_fine is reserved and currently uses exhaustive order.",
    )
    parser.add_argument("--max-trials-per-codec", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=600.0, help="External command timeout in seconds")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    if not args.input.exists():
        LOGGER.error("Input image does not exist: %s", args.input)
        return 2

    codec_names = [name.strip() for name in args.codecs.split(",")]
    codecs = build_codecs(codec_names, timeout=args.timeout)
    if not codecs:
        LOGGER.error("No known codecs were requested")
        return 2

    selector = AdaptiveSelector(
        codecs,
        target_ratio=args.target_ratio,
        min_psnr=args.min_psnr,
        mode=args.mode,
        max_trials_per_codec=args.max_trials_per_codec,
        search_mode=args.search,
    )

    try:
        best, results, _features = selector.compress(args.input, args.output_dir)
    except Exception as exc:
        LOGGER.error("Compression failed: %s", exc)
        return 1

    LOGGER.info("Evaluated %d candidate results", len(results))
    LOGGER.info(
        "Best: codec=%s param=%s CR=%.4f bpp=%.4f PSNR=%.4f SSIM=%.4f passed=%s",
        best.codec_name,
        best.param,
        best.compression_ratio,
        best.bpp,
        best.psnr,
        best.ssim,
        best.passed,
    )
    if best.warning_message:
        LOGGER.warning(best.warning_message)
    LOGGER.info("Outputs written to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
