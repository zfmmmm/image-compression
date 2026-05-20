from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import tempfile

from strategy.adaptive_selector import AdaptiveSelector
from utils.codec_registry import build_codecs
from utils.logging_utils import setup_logging
from utils.output_artifacts import copy_best_bitstream
from utils.output_options import OutputOptions, resolve_output_options


LOGGER = logging.getLogger(__name__)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


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
    parser.add_argument(
        "--save-candidates",
        type=parse_bool,
        default=True,
        help="Whether to keep per-parameter candidate bitstreams/reconstructions/result.json files.",
    )
    parser.add_argument(
        "--save-all-results-csv",
        type=parse_bool,
        default=True,
        help="Whether to keep all_results.csv for the evaluated candidates.",
    )
    parser.add_argument(
        "--best-bitstreams-dir",
        type=Path,
        default=None,
        help="Optional flat directory for the selected best compressed bitstream.",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help=(
            "Only write the selected best compressed bitstream. "
            "If --best-bitstreams-dir is omitted, --output-dir is used as the flat bitstream directory."
        ),
    )
    return parser.parse_args()


def _run_compression(args: argparse.Namespace, codecs, work_dir: Path, output_options: OutputOptions) -> int:
    if output_options.best_bitstreams_dir is not None:
        output_options.best_bitstreams_dir.mkdir(parents=True, exist_ok=True)

    selector = AdaptiveSelector(
        codecs,
        target_ratio=args.target_ratio,
        min_psnr=args.min_psnr,
        mode=args.mode,
        max_trials_per_codec=args.max_trials_per_codec,
        search_mode=args.search,
        save_candidate_artifacts=output_options.save_candidate_artifacts,
        save_all_results_csv=output_options.save_all_results_csv,
    )

    try:
        best, results, _features = selector.compress(args.input, work_dir)
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
    if output_options.best_bitstreams_dir is not None and best.success:
        copied_path = copy_best_bitstream(best, output_options.best_bitstreams_dir, "compressed")
        LOGGER.info("Best bitstream copied to %s", copied_path)

    output_location = output_options.best_bitstreams_dir if args.best_only else args.output_dir
    LOGGER.info("Outputs written to %s", output_location)
    return 0


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

    output_options = resolve_output_options(
        output_dir=args.output_dir,
        best_bitstreams_dir=args.best_bitstreams_dir,
        best_only=args.best_only,
        save_candidates=args.save_candidates,
        save_all_results_csv=args.save_all_results_csv,
    )

    if args.best_only:
        with tempfile.TemporaryDirectory(prefix="image-compression-") as temp_dir:
            return _run_compression(args, codecs, Path(temp_dir), output_options)

    return _run_compression(args, codecs, args.output_dir, output_options)


if __name__ == "__main__":
    sys.exit(main())
