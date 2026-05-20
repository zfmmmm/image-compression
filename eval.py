from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
from typing import Sequence, Any

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

# 每个 worker 进程独立持有自己的 codec 实例，避免多进程 pickle / 共享状态问题
_WORKER_CODECS: Sequence[Any] | None = None


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

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Number of parallel worker processes. "
            "0 means use all CPU cores. "
            "1 means disable multiprocessing."
        ),
    )

    return parser.parse_args()


def _resolve_workers(requested_workers: int, num_images: int) -> int:
    if requested_workers <= 0:
        workers = os.cpu_count() or 1
    else:
        workers = requested_workers

    workers = max(1, workers)
    workers = min(workers, max(1, num_images))
    return workers


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


def _compress_one_image(
    *,
    image_path: Path,
    input_dir: Path,
    output_dir: Path,
    codecs: Sequence[Any],
    target_ratio: float,
    min_psnr: float,
    mode: str,
    max_trials_per_codec: int | None,
    search_mode: str,
) -> tuple[CodecResult, list[CodecResult]]:
    image_work_dir = output_dir / "images" / _safe_image_id(input_dir, image_path)

    selector = AdaptiveSelector(
        codecs,
        target_ratio=target_ratio,
        min_psnr=min_psnr,
        mode=mode,
        max_trials_per_codec=max_trials_per_codec,
        search_mode=search_mode,
    )

    try:
        best, candidates, _features = selector.compress(image_path, image_work_dir)
    except Exception as exc:
        LOGGER.exception("Failed to evaluate %s", image_path)
        best = _failure_result(image_path, image_work_dir, str(exc))
        candidates = []

    return best, candidates


def _init_worker(
    codec_names: list[str],
    timeout: float,
    log_level: str,
) -> None:
    """
    每个进程启动时调用一次。

    不要从主进程传 codec 对象进来，因为很多 codec wrapper 里可能有 subprocess、
    临时路径、状态缓存等，不一定能被 pickle。
    """
    global _WORKER_CODECS

    setup_logging(log_level)

    _WORKER_CODECS = build_codecs(codec_names, timeout=timeout)
    if not _WORKER_CODECS:
        raise RuntimeError(f"No known codecs were requested: {codec_names}")


def _worker_task(
    payload: tuple[
        Path,
        Path,
        Path,
        float,
        float,
        str,
        int | None,
        str,
    ],
) -> tuple[CodecResult, list[CodecResult]]:
    global _WORKER_CODECS

    if _WORKER_CODECS is None:
        raise RuntimeError("Worker codecs were not initialized")

    (
        image_path,
        input_dir,
        output_dir,
        target_ratio,
        min_psnr,
        mode,
        max_trials_per_codec,
        search_mode,
    ) = payload

    return _compress_one_image(
        image_path=image_path,
        input_dir=input_dir,
        output_dir=output_dir,
        codecs=_WORKER_CODECS,
        target_ratio=target_ratio,
        min_psnr=min_psnr,
        mode=mode,
        max_trials_per_codec=max_trials_per_codec,
        search_mode=search_mode,
    )


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    if not args.input_dir.exists():
        LOGGER.error("Input directory does not exist: %s", args.input_dir)
        return 2

    image_paths = list(iter_image_paths(args.input_dir, recursive=args.recursive))
    if not image_paths:
        LOGGER.error("No supported images found in %s", args.input_dir)
        return 2

    codec_names = [name.strip() for name in args.codecs.split(",") if name.strip()]
    if not codec_names:
        LOGGER.error("No codecs were requested")
        return 2

    workers = _resolve_workers(args.workers, len(image_paths))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_image_best_slots: list[CodecResult | None] = [None] * len(image_paths)
    candidate_slots: list[list[CodecResult]] = [[] for _ in image_paths]

    LOGGER.info("Found %d images", len(image_paths))
    LOGGER.info("Using %d worker process(es)", workers)

    if workers == 1:
        codecs = build_codecs(codec_names, timeout=args.timeout)
        if not codecs:
            LOGGER.error("No known codecs were requested")
            return 2

        for idx, image_path in enumerate(tqdm(image_paths, desc="Evaluating images")):
            best, candidates = _compress_one_image(
                image_path=image_path,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                codecs=codecs,
                target_ratio=args.target_ratio,
                min_psnr=args.min_psnr,
                mode=args.mode,
                max_trials_per_codec=args.max_trials_per_codec,
                search_mode=args.search,
            )

            per_image_best_slots[idx] = best
            candidate_slots[idx] = candidates

    else:
        # 主进程先构造一次，用于提前检查 codec 名称 / 环境。
        # 真正执行时，每个 worker 会在 _init_worker 中重新构造自己的 codec 实例。
        probe_codecs = build_codecs(codec_names, timeout=args.timeout)
        if not probe_codecs:
            LOGGER.error("No known codecs were requested")
            return 2
        del probe_codecs

        tasks = [
            (
                image_path,
                args.input_dir,
                args.output_dir,
                args.target_ratio,
                args.min_psnr,
                args.mode,
                args.max_trials_per_codec,
                args.search,
            )
            for image_path in image_paths
        ]

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(codec_names, args.timeout, args.log_level),
        ) as executor:
            future_to_index = {
                executor.submit(_worker_task, task): idx
                for idx, task in enumerate(tasks)
            }

            for future in tqdm(
                as_completed(future_to_index),
                total=len(future_to_index),
                desc="Evaluating images",
            ):
                idx = future_to_index[future]
                image_path = image_paths[idx]

                try:
                    best, candidates = future.result()
                except Exception as exc:
                    LOGGER.exception("Worker crashed while evaluating %s", image_path)
                    image_work_dir = args.output_dir / "images" / _safe_image_id(args.input_dir, image_path)
                    best = _failure_result(
                        image_path,
                        image_work_dir,
                        f"Worker crashed: {exc}",
                    )
                    candidates = []

                per_image_best_slots[idx] = best
                candidate_slots[idx] = candidates

    per_image_best = [result for result in per_image_best_slots if result is not None]
    all_candidates = [
        candidate
        for candidates in candidate_slots
        for candidate in candidates
    ]

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