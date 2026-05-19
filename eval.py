from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from tqdm import tqdm

from utils.codec_imports import ensure_project_codecs_importable


ensure_project_codecs_importable()

from codecs.base import CodecResult
from metrics.image_metrics import load_image_rgb, original_theoretical_size_bytes
from strategy.adaptive_selector import AdaptiveSelector
from utils.codec_registry import build_codecs
from utils.image_io import iter_image_paths
from utils.logging_utils import setup_logging
from utils.report import (
    generate_dataset_summary,
    plot_cr_hist,
    plot_psnr_hist,
    plot_rd_scatter,
    save_all_results_csv,
)


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
    parser = argparse.ArgumentParser(description="Batch evaluation for traditional adaptive image compression.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--mode",
        default="ratio_first",
        choices=["ratio_first", "quality_first", "remote_sensing"],
    )
    parser.add_argument("--target-ratio", default=16.0, type=float)
    parser.add_argument("--min-psnr", default=35.0, type=float)
    parser.add_argument(
        "--codecs",
        default="jpeg,jxl,avif,jpeg2000",
        help="Comma-separated codec names: jpeg,jxl,avif,jpeg2000,bpg",
    )
    parser.add_argument("--recursive", type=parse_bool, default=False)
    parser.add_argument(
        "--search",
        default="exhaustive",
        choices=["exhaustive", "coarse_to_fine"],
    )
    parser.add_argument("--max-trials-per-codec", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _safe_image_id(input_dir: Path, image_path: Path) -> str:
    rel = image_path.relative_to(input_dir).with_suffix("")
    text = "__".join(rel.parts)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def _failure_result(input_path: Path, output_dir: Path, message: str) -> CodecResult:
    try:
        original_size = original_theoretical_size_bytes(load_image_rgb(input_path))
    except Exception:
        original_size = 0
    return CodecResult(
        codec_name="none",
        input_path=input_path,
        bitstream_path=output_dir / "best" / "compressed.bin",
        recon_path=output_dir / "best" / "recon.png",
        param={},
        original_theoretical_size=original_size,
        compressed_size=0,
        compression_ratio=0.0,
        bpp=float("inf"),
        psnr=float("-inf"),
        ssim=0.0,
        ms_ssim=None,
        edge_psnr=None,
        encode_time_ms=0.0,
        decode_time_ms=0.0,
        success=False,
        error_message=message,
        passed=False,
        warning_message=message,
    )


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    if not args.input_dir.exists():
        LOGGER.error("Input directory does not exist: %s", args.input_dir)
        return 2

    image_paths = iter_image_paths(args.input_dir, recursive=args.recursive)
    if not image_paths:
        LOGGER.error("No supported images found in %s", args.input_dir)
        return 2

    codec_names = [name.strip() for name in args.codecs.split(",")]
    codecs = build_codecs(codec_names, timeout=args.timeout)
    if not codecs:
        LOGGER.error("No known codecs were requested")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_image_best: list[CodecResult] = []
    all_candidates: list[CodecResult] = []

    for image_path in tqdm(image_paths, desc="Evaluating images"):
        image_work_dir = args.output_dir / "images" / _safe_image_id(args.input_dir, image_path)
        selector = AdaptiveSelector(
            codecs,
            target_ratio=args.target_ratio,
            min_psnr=args.min_psnr,
            mode=args.mode,
            max_trials_per_codec=args.max_trials_per_codec,
            search_mode=args.search,
        )
        try:
            best, candidates, _features = selector.compress(image_path, image_work_dir)
        except Exception as exc:
            LOGGER.error("Failed to evaluate %s: %s", image_path, exc)
            best = _failure_result(image_path, image_work_dir, str(exc))
            candidates = []
        per_image_best.append(best)
        all_candidates.extend(candidates)

    save_all_results_csv(per_image_best, args.output_dir / "per_image_results.csv")
    save_all_results_csv(all_candidates, args.output_dir / "all_candidate_results.csv")
    generate_dataset_summary(
        per_image_best,
        all_candidates,
        args.output_dir,
        dataset_path=args.input_dir,
        target_ratio=args.target_ratio,
        min_psnr=args.min_psnr,
        mode=args.mode,
    )
    plot_rd_scatter(all_candidates, args.output_dir / "rd_scatter.png")
    plot_psnr_hist(per_image_best, args.output_dir / "psnr_hist.png")
    plot_cr_hist(per_image_best, args.output_dir / "cr_hist.png")

    LOGGER.info("Evaluation complete. Reports written to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
