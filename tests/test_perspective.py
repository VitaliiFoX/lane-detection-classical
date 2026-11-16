"""Тести perspective (M3)."""

import numpy as np
import pytest

from lanes.methods.perspective import homography, top_view_rectangle, transform_points, warp

IMAGE = np.arange(25, dtype=np.float64).reshape(5, 5)
# Трапеція на фото й прямокутник виду зверху: верх-ліво, верх-право, низ-право, низ-ліво.
SRC = np.array([[40, 50], [60, 50], [90, 100], [10, 100]], dtype=np.float64)
DST = np.array([[25, 0], [75, 0], [75, 100], [25, 100]], dtype=np.float64)
SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)


def test_homography_scale_by_hand():
    # Одиничний квадрат → квадрат зі стороною 2: u = 2x, v = 2y, перспективи
    # немає, тож матриця має бути діагональною: diag(2, 2, 1).
    matrix = homography(SQUARE, SQUARE * 2)
    np.testing.assert_allclose(matrix, np.diag([2.0, 2.0, 1.0]), atol=1e-12)


def test_homography_shift_by_hand():
    # Зсув на (3, 5): u = x + 3, v = y + 5 → третій стовпець (3, 5, 1).
    matrix = homography(SQUARE, SQUARE + [3, 5])
    np.testing.assert_allclose(matrix, [[1, 0, 3], [0, 1, 5], [0, 0, 1]], atol=1e-12)


def test_src_points_go_to_dst():
    matrix = homography(SRC, DST)
    np.testing.assert_allclose(transform_points(SRC, matrix), DST, atol=1e-9)
    # h33 зафіксовано рівним 1.
    assert matrix[2, 2] == 1.0


def test_there_and_back_is_identity():
    forward = homography(SRC, DST)
    backward = homography(DST, SRC)
    points = np.array([[50, 60], [30, 90], [70, 75]], dtype=np.float64)
    restored = transform_points(transform_points(points, forward), backward)
    np.testing.assert_allclose(restored, points, atol=1e-9)


def test_transform_points_divides_by_w():
    # matrix · (2, 4, 1) = (2, 4, 2) → після ділення на w = 2 точка (1, 2).
    matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=np.float64)
    np.testing.assert_allclose(transform_points(np.array([[2.0, 4.0]]), matrix), [[1, 2]])


def test_identity_warp_keeps_image():
    np.testing.assert_array_equal(warp(IMAGE, np.eye(3), (5, 5)), IMAGE)


def test_warp_shift_by_hand():
    # Матриця зсуває фото на 1 піксель праворуч: u = x + 1. Зворотне
    # відображення: піксель результату (u, v) бере фото в (u − 1, v).
    # Для u = 0 це x = −1, поза фото → 0.
    shift = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    result = warp(IMAGE, shift, (5, 5))
    np.testing.assert_array_equal(result[:, 0], np.zeros(5))
    np.testing.assert_array_equal(result[:, 1:], IMAGE[:, :4])


def test_warp_scale_uses_nearest_pixel():
    # Збільшення вдвічі: піксель результату (u, v) бере фото в (u / 2, v / 2),
    # округлене до найближчого. 2×2 → 4×4; для u = 3: 1.5 → 2, поза фото → 0.
    image = np.array([[10, 20], [30, 40]], dtype=np.float64)
    result = warp(image, np.diag([2.0, 2.0, 1.0]), (4, 4))
    assert result[0, 0] == 10
    assert result[0, 2] == 20
    assert result[2, 0] == 30
    assert result[2, 2] == 40
    assert result[0, 3] == 0


def test_warp_color_image_and_output_size():
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[2, 2] = (10, 20, 30)
    result = warp(image, np.eye(3), (7, 4))
    # out_size — (ширина, висота), масив — (висота, ширина, 3).
    assert result.shape == (4, 7, 3)
    assert result.dtype == np.uint8
    assert tuple(result[2, 2]) == (10, 20, 30)
    # Стовпці 5 і 6 лежать поза фото 5×5.
    assert not result[:, 5:].any()


def test_top_view_rectangle_by_hand():
    # Ширина 101, висота 51, відступ 0.25: зліва 0.25 · 100 = 25, справа 100 − 25 = 75.
    rectangle = top_view_rectangle((101, 51), 0.25)
    np.testing.assert_allclose(rectangle, [[25, 0], [75, 0], [75, 50], [25, 50]])


def test_degenerate_points_raise():
    # Усі чотири точки збіглися (фото 1×1) — перетворення не існує.
    with pytest.raises(np.linalg.LinAlgError):
        homography(np.zeros((4, 2)), DST)
