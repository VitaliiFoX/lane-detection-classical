"""Тести roi (M1)."""

import numpy as np

from lanes.roi import roi_mask, roi_polygon, roi_top

# Трапеція: низ від 0 до 100 % ширини, верх від 30 до 70 % ширини на 40 % висоти.
TRAPEZOID = [[0.0, 1.0], [0.3, 0.4], [0.7, 0.4], [1.0, 1.0]]
# Вужча трапеція, яка не торкається жодного кута фото.
NARROW = [[0.2, 1.0], [0.4, 0.4], [0.6, 0.4], [0.8, 1.0]]


def test_polygon_by_hand():
    # Фото 11×11: W − 1 = H − 1 = 10, тож частка 0.3 → 3, 0.4 → 4, 0.7 → 7, 1.0 → 10.
    polygon = roi_polygon((11, 11), TRAPEZOID)
    np.testing.assert_allclose(polygon, [(0, 10), (3, 4), (7, 4), (10, 10)])


def test_polygon_uses_width_for_x_and_height_for_y():
    # Фото 5 рядків × 21 стовпець: x = 0.5 · 20 = 10, y = 0.5 · 4 = 2.
    np.testing.assert_allclose(roi_polygon((5, 21), [[0.5, 0.5]]), [(10, 2)])


def test_mask_corners_outside_bottom_center_inside():
    # 10×10, вершини в пікселях: (1.8, 9), (3.6, 3.6), (5.4, 3.6), (7.2, 9).
    mask = roi_mask((10, 10), NARROW)
    assert mask.shape == (10, 10)
    assert mask.dtype == np.bool_
    # Усі чотири кутові пікселі лежать поза трапецією.
    assert not mask[0, 0] and not mask[0, 9] and not mask[9, 0] and not mask[9, 9]
    # Центр нижнього краю (x = 4 і x = 5 між 1.8 та 7.2) — усередині.
    assert mask[9, 4] and mask[9, 5]
    # Вище за верхню межу y = 3.6 (рядки 0…2) трапеції немає.
    assert not mask[:3].any()


def test_mask_narrows_towards_top():
    # У трапеції нижній рядок ширший за рядок біля верхньої межі.
    mask = roi_mask((10, 10), NARROW)
    assert mask[9].sum() > mask[5].sum() > 0


def test_config_polygon_top_corners_outside(config):
    mask = roi_mask((10, 10), config["roi"]["polygon"])
    assert not mask[0, 0] and not mask[0, 9]
    assert mask[9, 4] and mask[9, 5]


def test_top_by_hand():
    # Найменший y: 0.4 · (11 − 1) = 4.
    assert roi_top((11, 11), TRAPEZOID) == 4
    # 0.4 · (10 − 1) = 3.6 → округлюємо до рядка 4.
    assert roi_top((10, 10), NARROW) == 4


def test_single_pixel_image():
    # Фото 1×1: усі вершини зливаються в точку (0, 0), маска не падає.
    assert roi_mask((1, 1), TRAPEZOID).shape == (1, 1)
    assert roi_top((1, 1), TRAPEZOID) == 0
