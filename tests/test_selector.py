from pathlib import Path

import pytest

from codecs.base import CodecResult
from strategy.adaptive_selector import AdaptiveSelector
from strategy.feature_extractor import ImageFeatures


def make_result(
    codec_name: str,
    cr: float,
    psnr: float,
    ssim: float = 0.9,
    edge_psnr: float = 35.0,
    success: bool = True,
) -> CodecResult:
    return CodecResult(
        codec_name=codec_name,
        input_path=Path("input.png"),
        bitstream_path=Path(f"{codec_name}.bin"),
        recon_path=Path(f"{codec_name}.png"),
        param={},
        original_theoretical_size=300,
        compressed_size=max(1, round(300 / cr)),
        compression_ratio=cr,
        bpp=24 / cr,
        psnr=psnr,
        ssim=ssim,
        ms_ssim=None,
        edge_psnr=edge_psnr,
        encode_time_ms=1.0,
        decode_time_ms=1.0,
        success=success,
        error_message=None,
        passed=False,
        warning_message=None,
    )


def make_features(edge_density: float = 0.05) -> ImageFeatures:
    return ImageFeatures(
        width=10,
        height=10,
        channels=3,
        entropy=5.0,
        edge_density=edge_density,
        gradient_mean=2.0,
        gradient_std=1.0,
        colorfulness=3.0,
        local_variance_mean=1.0,
        estimated_noise=0.5,
    )


def test_ratio_first_selects_highest_psnr_among_results_meeting_ratio():
    selector = AdaptiveSelector([], target_ratio=16.0, mode="ratio_first")
    results = [
        make_result("jpeg", 14.0, 42.0),
        make_result("jxl", 16.2, 37.0),
        make_result("avif", 18.0, 39.0),
    ]

    best = selector.select_best(results, make_features())

    assert best.codec_name == "avif"
    assert best.passed is True


def test_quality_first_selects_highest_cr_among_results_meeting_psnr():
    selector = AdaptiveSelector([], min_psnr=35.0, mode="quality_first")
    results = [
        make_result("jpeg", 20.0, 34.0),
        make_result("jxl", 16.0, 36.0),
        make_result("avif", 19.0, 35.5),
    ]

    best = selector.select_best(results, make_features())

    assert best.codec_name == "avif"
    assert best.passed is True


def test_remote_sensing_prefers_higher_edge_psnr_for_edge_dense_images():
    selector = AdaptiveSelector([], target_ratio=16.0, mode="remote_sensing")
    results = [
        make_result("higher_psnr", 16.5, 39.0, ssim=0.91, edge_psnr=34.0),
        make_result("higher_edge", 16.2, 37.5, ssim=0.91, edge_psnr=47.0),
    ]

    best = selector.select_best(results, make_features(edge_density=0.2))

    assert best.codec_name == "higher_edge"
    assert best.passed is True


def test_selector_rejects_unknown_mode():
    with pytest.raises(ValueError):
        AdaptiveSelector([], mode="unknown")
