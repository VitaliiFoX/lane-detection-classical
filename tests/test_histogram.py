"""Тести histogram (F9)."""

import numpy as np

from lanes.methods.histogram import equalize_histogram


def test_equalization_by_hand():
    # 3×3: рівень 10 — 4 пікселі, 20 — 3, 30 — 2.
    gray = np.array([[10, 10, 10],
                     [10, 20, 20],
                     [20, 30, 30]], dtype=np.float64)
    # cdf: 10 → 4, 20 → 7, 30 → 9; cdf_min = 4, N − cdf_min = 5.
    # 10 → (4 − 4)/5 · 255 = 0; 20 → (7 − 4)/5 · 255 = 153; 30 → (9 − 4)/5 · 255 = 255.
    expected = np.array([[0, 0, 0],
                         [0, 153, 153],
                         [153, 255, 255]], dtype=np.float64)
    np.testing.assert_array_equal(equalize_histogram(gray), expected)


def test_low_contrast_is_stretched():
    gray = np.array([[100, 101], [102, 103]], dtype=np.float64)
    result = equalize_histogram(gray)
    assert result.min() == 0
    assert result.max() == 255


def test_uniform_image_unchanged():
    gray = np.full((3, 4), 77.0)
    np.testing.assert_array_equal(equalize_histogram(gray), gray)
