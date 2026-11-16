"""Тести synthetic (M2)."""

import numpy as np

from lanes.synthetic import LANE_COLOR, ROAD_COLOR, SKY_COLOR, make_road


def test_lanes_by_hand():
    # Кадр 101×101: W − 1 = H − 1 = 100, тож частки одразу дають пікселі.
    # Ліва смуга (0.20, 1.0) → (0.46, 0.5): точки (20, 100) і (46, 50).
    # Права смуга (0.80, 1.0) → (0.54, 0.5): точки (80, 100) і (54, 50).
    _, lanes = make_road(101, 101)
    assert len(lanes) == 2
    np.testing.assert_allclose(lanes[0], [[20, 100], [46, 50]])
    np.testing.assert_allclose(lanes[1], [[80, 100], [54, 50]])


def test_scene_colors():
    image, lanes = make_road(320, 180)
    assert image.shape == (180, 320, 3)
    assert image.dtype == np.uint8
    # Верхній рядок — небо, низ по центру між смугами — асфальт.
    assert tuple(image[0, 160]) == SKY_COLOR
    assert tuple(image[179, 160]) == ROAD_COLOR
    # Середина кожної смуги біла й світліша за асфальт поруч.
    for lane in lanes:
        x, y = np.round(lane.mean(axis=0)).astype(int)
        assert tuple(image[y, x]) == LANE_COLOR
        assert image[y, x].sum() > image[y, 160].sum()


def test_brightness_scales_pixels():
    # Яскравість 0.5: біла смуга 255 → 127.5 → 128, асфальт 80 → 40.
    image, lanes = make_road(320, 180, brightness=0.5)
    x, y = np.round(lanes[0].mean(axis=0)).astype(int)
    assert tuple(image[y, x]) == (128, 128, 128)
    assert tuple(image[179, 160]) == (40, 40, 40)


def test_noise_is_deterministic_by_seed():
    first, _ = make_road(64, 36, noise_sigma=20.0, seed=7)
    second, _ = make_road(64, 36, noise_sigma=20.0, seed=7)
    other, _ = make_road(64, 36, noise_sigma=20.0, seed=8)
    clean, _ = make_road(64, 36)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    assert not np.array_equal(first, clean)


def test_without_noise_seed_does_not_matter():
    first, _ = make_road(64, 36, seed=1)
    second, _ = make_road(64, 36, seed=2)
    np.testing.assert_array_equal(first, second)


def test_single_pixel_frame():
    image, lanes = make_road(1, 1)
    assert image.shape == (1, 1, 3)
    assert len(lanes) == 2
