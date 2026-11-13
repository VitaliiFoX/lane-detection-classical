"""Тести threshold (F10): поріг Оцу."""

import numpy as np

from lanes.methods.threshold import otsu_binarize, otsu_threshold


def test_otsu_by_hand():
    # Три пікселі: 0, 10, 100 (кожен з імовірністю 1/3).
    # t = 0:  ω₀ = 1/3, μ₀ = 0;  ω₁ = 2/3, μ₁ = 55  → σ² = 1/3 · 2/3 · 55² ≈ 672.2
    # t = 10: ω₀ = 2/3, μ₀ = 5;  ω₁ = 1/3, μ₁ = 100 → σ² = 2/3 · 1/3 · 95² ≈ 2005.6
    # Для t від 10 до 99 класи ті самі, тож перший максимум — t = 10.
    assert otsu_threshold(np.array([0, 10, 100])) == 10


def test_bimodal_3x3_splits_between_modes():
    gray = np.array([[10, 12, 11],
                     [200, 205, 11],
                     [210, 10, 200]], dtype=np.float64)
    binary, thresholds = otsu_binarize(gray)
    assert 12 <= thresholds[0] < 200
    np.testing.assert_array_equal(binary, gray >= 200)


def test_iterations_narrow_the_mask():
    # Рівномірна шкала 0…255: кожна ітерація лишає приблизно світлішу половину.
    gray = np.arange(256, dtype=np.float64).reshape(16, 16)
    once, _ = otsu_binarize(gray, iterations=1)
    twice, thresholds = otsu_binarize(gray, iterations=2)
    assert len(thresholds) == 2
    assert thresholds[1] > thresholds[0]
    assert twice.sum() < once.sum()
    assert not (twice & ~once).any()


def test_mask_limits_pixels():
    gray = np.array([[0, 255], [0, 255]], dtype=np.float64)
    mask = np.array([[True, True], [False, False]])
    binary, _ = otsu_binarize(gray, mask=mask)
    # Нижній рядок поза маскою, тож білий лише верхній правий піксель.
    np.testing.assert_array_equal(binary, [[False, True], [False, False]])


def test_uniform_image_gives_empty_mask():
    binary, thresholds = otsu_binarize(np.full((3, 3), 50.0), iterations=2)
    assert thresholds == [50]
    assert not binary.any()
