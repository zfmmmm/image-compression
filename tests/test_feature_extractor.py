import numpy as np

from strategy.feature_extractor import extract_features


def test_solid_color_has_low_entropy_and_almost_no_edges():
    image = np.full((64, 64, 3), 120, dtype=np.uint8)

    features = extract_features(image)

    assert features.entropy < 0.1
    assert features.edge_density < 0.01


def test_random_noise_has_high_entropy():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)

    features = extract_features(image)

    assert features.entropy > 7.0


def test_black_white_boundary_has_more_edges_than_solid_color():
    solid = np.full((64, 64, 3), 120, dtype=np.uint8)
    boundary = np.zeros((64, 64, 3), dtype=np.uint8)
    boundary[:, 32:, :] = 255

    solid_features = extract_features(solid)
    boundary_features = extract_features(boundary)

    assert boundary_features.edge_density > solid_features.edge_density
