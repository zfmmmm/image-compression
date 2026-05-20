from __future__ import annotations

from collections import Counter
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
import json
import math
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.codec_imports import ensure_project_codecs_importable


ensure_project_codecs_importable()

from codecs.base import CodecResult  # noqa: E402
from strategy.feature_extractor import ImageFeatures  # noqa: E402


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return None
        return value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    return value


def result_to_dict(result: CodecResult) -> dict[str, Any]:
    data = _json_safe(asdict(result))
    data["param_json"] = json.dumps(data.get("param", {}), ensure_ascii=False, sort_keys=True)
    return data


def save_result_json(result: CodecResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(asdict(result)), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_features_json(features: ImageFeatures, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(asdict(features)), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_all_results_csv(results: list[CodecResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result_to_dict(result) for result in results]
    if not rows:
        columns = [field.name for field in fields(CodecResult)] + ["param_json"]
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def _finite_values(results: list[CodecResult], attr: str) -> list[float]:
    values = []
    for result in results:
        value = getattr(result, attr)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _minimum(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.min(values))


def generate_dataset_summary(
    per_image_best_results: list[CodecResult],
    all_candidate_results: list[CodecResult],
    output_dir: Path,
    *,
    dataset_path: Path | None = None,
    target_ratio: float | None = None,
    min_psnr: float | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    successful_best = [result for result in per_image_best_results if result.success]
    passed_best = [result for result in per_image_best_results if result.passed]
    failed = [result for result in per_image_best_results if not result.passed]
    win_counts = Counter(result.codec_name for result in per_image_best_results if result.success)

    summary = {
        "dataset_path": str(dataset_path) if dataset_path else None,
        "num_images": len(per_image_best_results),
        "num_success_images": len(passed_best),
        "pass_rate": (len(passed_best) / len(per_image_best_results)) if per_image_best_results else 0.0,
        "avg_psnr": _mean(_finite_values(successful_best, "psnr")),
        "min_psnr": _minimum(_finite_values(successful_best, "psnr")),
        "avg_ssim": _mean(_finite_values(successful_best, "ssim")),
        "avg_edge_psnr": _mean(_finite_values(successful_best, "edge_psnr")),
        "avg_compression_ratio": _mean(_finite_values(successful_best, "compression_ratio")),
        "avg_bpp": _mean(_finite_values(successful_best, "bpp")),
        "avg_encode_time_ms": _mean(_finite_values(successful_best, "encode_time_ms")),
        "avg_decode_time_ms": _mean(_finite_values(successful_best, "decode_time_ms")),
        "codec_win_counts": dict(sorted(win_counts.items())),
        "failed_images": [
            {
                "input_path": str(result.input_path),
                "codec_name": result.codec_name,
                "compression_ratio": result.compression_ratio,
                "psnr": result.psnr,
                "warning_message": result.warning_message,
                "error_message": result.error_message,
            }
            for result in failed
        ],
        "target_ratio": target_ratio,
        "minimum_psnr": min_psnr,
        "mode": mode,
        "num_candidate_results": len(all_candidate_results),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_summary_markdown(summary, output_dir / "summary.md")
    return summary


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}{suffix}"
    return f"{value}{suffix}"


def _write_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    pass_rate = summary["pass_rate"] * 100.0
    lines = [
        "# Image Compression Evaluation Report",
        "",
        "## Test Setting",
        "",
        f"- Target compression ratio: {summary.get('target_ratio')}",
        f"- Minimum PSNR: {summary.get('minimum_psnr')} dB",
        f"- Mode: {summary.get('mode')}",
        f"- Number of images: {summary.get('num_images')}",
        "",
        "## Overall Result",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Pass rate | {pass_rate:.2f}% |",
        f"| Average PSNR | {_fmt(summary.get('avg_psnr'), ' dB')} |",
        f"| Minimum PSNR | {_fmt(summary.get('min_psnr'), ' dB')} |",
        f"| Average SSIM | {_fmt(summary.get('avg_ssim'))} |",
        f"| Average Edge-PSNR | {_fmt(summary.get('avg_edge_psnr'), ' dB')} |",
        f"| Average CR | {_fmt(summary.get('avg_compression_ratio'))} |",
        f"| Average bpp | {_fmt(summary.get('avg_bpp'))} |",
        f"| Average encoding time | {_fmt(summary.get('avg_encode_time_ms'), ' ms')} |",
        f"| Average decoding time | {_fmt(summary.get('avg_decode_time_ms'), ' ms')} |",
        "",
        "## Codec Win Counts",
        "",
        "| Codec | Count |",
        "|---|---:|",
    ]
    for codec_name, count in summary.get("codec_win_counts", {}).items():
        lines.append(f"| {codec_name} | {count} |")

    lines.extend(["", "## Failed Images", ""])
    failed_images = summary.get("failed_images", [])
    if not failed_images:
        lines.append("None.")
    else:
        lines.extend(["| Image | Codec | CR | PSNR | Message |", "|---|---|---:|---:|---|"])
        for item in failed_images:
            message = item.get("warning_message") or item.get("error_message") or ""
            lines.append(
                f"| {item.get('input_path')} | {item.get('codec_name')} | "
                f"{_fmt(item.get('compression_ratio'))} | {_fmt(item.get('psnr'))} | {message} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_rd_scatter(all_candidate_results: list[CodecResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    successful = [result for result in all_candidate_results if result.success and math.isfinite(result.psnr)]
    plt.figure(figsize=(8, 5))
    if successful:
        for codec_name in sorted({result.codec_name for result in successful}):
            rows = [result for result in successful if result.codec_name == codec_name]
            plt.scatter(
                [result.bpp for result in rows],
                [result.psnr for result in rows],
                label=codec_name,
                s=24,
                alpha=0.8,
            )
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No successful candidates", ha="center", va="center")
    plt.xlabel("bpp")
    plt.ylabel("PSNR (dB)")
    plt.title("Rate-Distortion Scatter")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_psnr_hist(best_results: list[CodecResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = _finite_values([result for result in best_results if result.success], "psnr")
    plt.figure(figsize=(7, 4))
    if values:
        plt.hist(values, bins=min(20, max(5, len(values))), color="#386cb0", alpha=0.85)
    else:
        plt.text(0.5, 0.5, "No successful images", ha="center", va="center")
    plt.xlabel("PSNR (dB)")
    plt.ylabel("Count")
    plt.title("Best Result PSNR")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_cr_hist(best_results: list[CodecResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = _finite_values([result for result in best_results if result.success], "compression_ratio")
    plt.figure(figsize=(7, 4))
    if values:
        plt.hist(values, bins=min(20, max(5, len(values))), color="#4daf4a", alpha=0.85)
    else:
        plt.text(0.5, 0.5, "No successful images", ha="center", va="center")
    plt.xlabel("Compression ratio")
    plt.ylabel("Count")
    plt.title("Best Result Compression Ratio")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
