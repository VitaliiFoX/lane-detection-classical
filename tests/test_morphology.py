"""Тести morphology (M4)."""

import numpy as np
import pytest

from lanes.methods.morphology import dilate, erode, opening, top_hat

# Одна світла точка 9 на темному полі 1.
DOT = np.array([[1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 9, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1]], dtype=np.float64)

# Рядки 0–1: вузька світла смуга (1 піксель) на фоні 50.
# Рядки 3–4: широка світла пляма (4 пікселі). Рядок 2 — лише фон.
STRIPE_AND_BLOB = np.array([[50, 50, 200, 50, 50],
                            [50, 50, 200, 50, 50],
                            [50, 50, 50, 50, 50],
                            [200, 200, 200, 200, 50],
                            [200, 200, 200, 200, 50]], dtype=np.float64)


def test_erode_removes_single_dot():
    # У вікні 3×3 навколо будь-якого пікселя є сусід зі значенням 1,
    # тож мінімум скрізь 1: точка зникає.
    np.testing.assert_array_equal(erode(DOT, 3, 3), np.ones((5, 5)))


def test_dilate_grows_dot_by_hand():
    # Максимум у вікні 3×3 дорівнює 9 для всіх пікселів, у чиє вікно
    # потрапляє центр (2, 2), тобто для квадрата 3×3 навколо нього.
    expected = np.ones((5, 5))
    expected[1:4, 1:4] = 9
    np.testing.assert_array_equal(dilate(DOT, 3, 3), expected)


def test_horizontal_element_works_along_rows_only():
    # Елемент 1×3: вікно — лівий сусід, сам піксель, правий сусід.
    # Точка розростається лише в рядку 2: стовпці 1, 2, 3.
    expected = np.ones((5, 5))
    expected[2, 1:4] = 9
    np.testing.assert_array_equal(dilate(DOT, 1, 3), expected)


def test_erode_row_by_hand():
    # Рядок [200, 200, 200, 200, 50], елемент 1×3, краї повторюються:
    # x = 0: min(200, 200, 200) = 200;  x = 1: 200;  x = 2: 200;
    # x = 3: min(200, 200, 50) = 50;    x = 4: min(200, 50, 50) = 50.
    eroded = erode(STRIPE_AND_BLOB, 1, 3)
    np.testing.assert_array_equal(eroded[3], [200, 200, 200, 50, 50])


def test_opening_restores_wide_blob_and_erases_stripe():
    opened = opening(STRIPE_AND_BLOB, 1, 3)
    # Смуга: ерозія [50, 50, 200, 50, 50] дає всі 50, дилатація лишає 50.
    np.testing.assert_array_equal(opened[0], [50, 50, 50, 50, 50])
    # Пляма: ерозія [200, 200, 200, 50, 50], дилатація:
    # x = 3: max(200, 50, 50) = 200, x = 4: max(50, 50, 50) = 50 → як оригінал.
    np.testing.assert_array_equal(opened[3], [200, 200, 200, 200, 50])


def test_top_hat_keeps_stripe_removes_blob():
    result = top_hat(STRIPE_AND_BLOB, 3)
    # Вузька смуга: 200 − 50 = 150 у стовпці 2, решта 0.
    np.testing.assert_array_equal(result[0], [0, 0, 150, 0, 0])
    np.testing.assert_array_equal(result[1], [0, 0, 150, 0, 0])
    # Широка пляма ширша за елемент — відкриття її відновило, різниця 0.
    assert not result[3:].any()
    assert not result[2].any()


def test_constant_image():
    flat = np.full((4, 6), 77.0)
    np.testing.assert_array_equal(erode(flat, 3, 3), flat)
    np.testing.assert_array_equal(dilate(flat, 3, 3), flat)
    assert not top_hat(flat, 5).any()


def test_element_wider_than_image():
    # Фото 1×1 і 2×2 з елементом 1×31: рамка з повтором країв рятує від падіння.
    assert top_hat(np.array([[5.0]]), 31).shape == (1, 1)
    assert not top_hat(np.full((2, 2), 9.0), 31).any()


def test_input_is_not_changed():
    before = STRIPE_AND_BLOB.copy()
    top_hat(STRIPE_AND_BLOB, 3)
    np.testing.assert_array_equal(STRIPE_AND_BLOB, before)


def test_even_element_is_rejected():
    with pytest.raises(ValueError):
        erode(DOT, 2, 3)
