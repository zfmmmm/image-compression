from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import logging
import math
import shutil

from utils.codec_imports import ensure_project_codecs_importable


ensure_project_codecs_importable()

from codecs.base import BaseCodec, CodecResult
from metrics.image_metrics import load_image_rgb
from strategy.feature_extractor import ImageFeatures, extract_features
from utils.report import save_all_results_csv, save_features_json, save_result_json


LOGGER = logging.getLogger(__name__)

VALID_MODES = {"ratio_first", "quality_first", "remote_sensing"}


def _param_slug(param: dict) -> str:
    if not param:
        return "default"
    parts = []
    for key in sorted(param):
        value = param[key]
        text = str(value).replace(".", "p").replace("/", "_")
        parts.append(f"{key}_{text}")
    return "_".join(parts)


class AdaptiveSelector:
    def __init__(
        self,
        codecs: list[BaseCodec],
        target_ratio: float = 16.0,
        min_psnr: float = 35.0,
        mode: str = "ratio_first",
        max_trials_per_codec: int | None = None,
        search_mode: str = "exhaustive",
        save_candidate_artifacts: bool = True,
        save_all_results_csv: bool = True,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Valid modes: {sorted(VALID_MODES)}")
        if search_mode not in {"exhaustive", "coarse_to_fine"}:
            raise ValueError("search_mode must be 'exhaustive' or 'coarse_to_fine'")
        self.codecs = codecs
        self.target_ratio = float(target_ratio)
        self.min_psnr = float(min_psnr)
        self.mode = mode
        self.max_trials_per_codec = max_trials_per_codec
        self.search_mode = search_mode
        self.save_candidate_artifacts = save_candidate_artifacts
        self.save_all_results_csv = save_all_results_csv

    def _params_for_codec(self, codec: BaseCodec) -> list[dict]:
        params = codec.candidate_params()
        if self.search_mode == "coarse_to_fine":
            LOGGER.info("coarse_to_fine is reserved; using exhaustive parameter order for %s", codec.name)
        if self.max_trials_per_codec is not None:
            return params[: self.max_trials_per_codec]
        return params

    def evaluate_all_codecs(self, input_path: Path, work_dir: Path) -> list[CodecResult]:
        candidates_dir = work_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        results: list[CodecResult] = []

        for codec in self.codecs:
            if not codec.is_available():
                LOGGER.warning("Codec '%s' is unavailable; skipping", codec.name)
                continue
            for param in self._params_for_codec(codec):
                candidate_dir = candidates_dir / f"{codec.name}_{_param_slug(param)}"
                result = codec.compress_and_eval(input_path, candidate_dir, param)
                if self.save_candidate_artifacts:
                    save_result_json(result, candidate_dir / "result.json")
                results.append(result)
        return results

    def _meets_acceptance(self, result: CodecResult) -> bool:
        return (
            result.success
            and result.compression_ratio >= self.target_ratio
            and result.psnr >= self.min_psnr
        )

    def _finalize_selection(
        self,
        result: CodecResult,
        *,
        primary_constraint_met: bool,
        primary_warning: str | None = None,
    ) -> CodecResult:
        result.passed = self._meets_acceptance(result)
        warnings: list[str] = []
        if primary_warning:
            warnings.append(primary_warning)
        if primary_constraint_met and result.compression_ratio < self.target_ratio:
            warnings.append(f"Selected result did not reach target CR >= {self.target_ratio:g}")
        if primary_constraint_met and result.psnr < self.min_psnr:
            warnings.append(f"Selected result did not reach minimum PSNR >= {self.min_psnr:g} dB")
        if warnings:
            result.warning_message = "; ".join(warnings)
        return result

    def _successful(self, results: list[CodecResult]) -> list[CodecResult]:
        return [result for result in results if result.success]

    def _select_ratio_first(self, results: list[CodecResult]) -> CodecResult:
        successful = self._successful(results)
        candidates = [result for result in successful if result.compression_ratio >= self.target_ratio]
        if candidates:
            best = max(candidates, key=lambda result: result.psnr)
            return self._finalize_selection(best, primary_constraint_met=True)
        if successful:
            best = min(successful, key=lambda result: (abs(result.compression_ratio - self.target_ratio), -result.psnr))
            return self._finalize_selection(
                best,
                primary_constraint_met=False,
                primary_warning=f"No successful candidate reached target CR >= {self.target_ratio:g}",
            )
        best = results[0]
        return self._finalize_selection(
            best,
            primary_constraint_met=False,
            primary_warning="No codec candidate completed successfully",
        )

    def _select_quality_first(self, results: list[CodecResult]) -> CodecResult:
        successful = self._successful(results)
        candidates = [result for result in successful if result.psnr >= self.min_psnr]
        if candidates:
            best = max(candidates, key=lambda result: result.compression_ratio)
            return self._finalize_selection(best, primary_constraint_met=True)
        if successful:
            best = max(successful, key=lambda result: result.psnr)
            return self._finalize_selection(
                best,
                primary_constraint_met=False,
                primary_warning=f"No successful candidate reached PSNR >= {self.min_psnr:g} dB",
            )
        best = results[0]
        return self._finalize_selection(
            best,
            primary_constraint_met=False,
            primary_warning="No codec candidate completed successfully",
        )

    def _score_remote_sensing(self, result: CodecResult, features: ImageFeatures) -> float:
        edge_psnr = result.edge_psnr if result.edge_psnr is not None else result.psnr
        psnr_norm = min(result.psnr, 50.0) / 50.0 if math.isfinite(result.psnr) else 1.0
        edge_norm = min(edge_psnr, 50.0) / 50.0 if math.isfinite(edge_psnr) else 1.0
        ssim_norm = min(max(result.ssim, 0.0), 1.0)
        cr_norm = min(result.compression_ratio / self.target_ratio, 2.0) / 2.0

        if features.edge_density > 0.15:
            weights = {
                "psnr": 0.50,
                "ssim": 0.20,
                "edge_psnr": 0.30,
                "compression_ratio": 0.00,
            }
        else:
            weights = {
                "psnr": 0.50,
                "ssim": 0.20,
                "edge_psnr": 0.20,
                "compression_ratio": 0.10,
            }

        total = sum(weights.values())
        return (
            weights["psnr"] * psnr_norm
            + weights["ssim"] * ssim_norm
            + weights["edge_psnr"] * edge_norm
            + weights["compression_ratio"] * cr_norm
        ) / total

    def _select_remote_sensing(self, results: list[CodecResult], features: ImageFeatures) -> CodecResult:
        successful = self._successful(results)
        candidates = [result for result in successful if result.compression_ratio >= self.target_ratio]
        if candidates:
            best = max(candidates, key=lambda result: self._score_remote_sensing(result, features))
            return self._finalize_selection(best, primary_constraint_met=True)
        return self._select_ratio_first(results)

    def select_best(self, results: list[CodecResult], features: ImageFeatures) -> CodecResult:
        if not results:
            raise ValueError("No codec results to select from")
        if self.mode == "ratio_first":
            return self._select_ratio_first(results)
        if self.mode == "quality_first":
            return self._select_quality_first(results)
        return self._select_remote_sensing(results, features)

    def _copy_best(self, best: CodecResult, output_dir: Path) -> CodecResult:
        best_dir = output_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        bitstream_dst = best_dir / f"compressed{best.bitstream_path.suffix}"
        recon_dst = best_dir / "recon.png"
        if best.bitstream_path.exists():
            shutil.copy2(best.bitstream_path, bitstream_dst)
        if best.recon_path.exists():
            shutil.copy2(best.recon_path, recon_dst)
        copied = replace(best, bitstream_path=bitstream_dst, recon_path=recon_dst)
        save_result_json(copied, best_dir / "best_result.json")
        return copied

    def compress(self, input_path: Path, output_dir: Path) -> tuple[CodecResult, list[CodecResult], ImageFeatures]:
        output_dir.mkdir(parents=True, exist_ok=True)
        image = load_image_rgb(input_path)
        features = extract_features(image)
        save_features_json(features, output_dir / "features.json")

        try:
            results = self.evaluate_all_codecs(input_path, output_dir)
            if self.save_all_results_csv:
                save_all_results_csv(results, output_dir / "all_results.csv")
            best = self.select_best(results, features)
            best_copy = self._copy_best(best, output_dir)
            return best_copy, results, features
        finally:
            if not self.save_candidate_artifacts:
                shutil.rmtree(output_dir / "candidates", ignore_errors=True)
