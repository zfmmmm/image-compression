import math

import numpy as np

from metrics.image_metrics import (
    compute_bpp,
    compute_compression_ratio,
    compute_psnr,
    compute_ssim,
)


def test_psnr_is_infinite_for_identical_images():
    image = np.full((8, 8, 3), 128, dtype=np.uint8)

    assert math.isinf(compute_psnr(image, image.copy()))


def test_psnr_drops_when_noise_is_added():
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    noisy = image.copy()
    noisy[0:4, 0:4, :] = 138

    assert compute_psnr(image, noisy) < compute_psnr(image, image.copy())


def test_bpp_and_compression_ratio_use_theoretical_rgb_size():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    bitstream_size = 60

    assert compute_compression_ratio(image, bitstream_size) == 10.0
    assert compute_bpp(image, bitstream_size) == 2.4


def test_ssim_is_in_unit_interval():
    image = np.full((16, 16, 3), 64, dtype=np.uint8)
    reconstructed = image.copy()
    reconstructed[:, 8:, :] = 80

    value = compute_ssim(image, reconstructed)

    assert 0.0 <= value <= 1.0
