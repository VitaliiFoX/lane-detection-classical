"""Тести filters.apply_kernel (F1)."""

import numpy as np
import pytest

from lanes.filters import apply_kernel

IMAGE_3X3 = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=np.float64)


def test_box_kernel_by_hand():
    result = apply_kernel(IMAGE_3X3, np.ones((3, 3)))
    # Центр (1, 1): сума всіх дев'яти пікселів 1 + 2 + … + 9 = 45.
    assert result[1, 1] == 45
    # Кут (0, 0): рамка повторює крайні пікселі, тож вікно
    # [[1, 1, 2], [1, 1, 2], [4, 4, 5]] → 1+1+2+1+1+2+4+4+5 = 21.
    assert result[0, 0] == 21


def test_kernel_is_not_flipped():
    # Ядро з одиницею праворуч від центру бере правого сусіда: result[y, x] = image[y, x + 1].
    right_neighbour = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    result = apply_kernel(IMAGE_3X3, right_neighbour)
    assert result[1, 0] == 5   # праворуч від 4 стоїть 5
    assert result[1, 2] == 6   # за краєм повторюється сам крайній піксель 6


def test_identity_kernel_keeps_image():
    identity = np.zeros((3, 3))
    identity[1, 1] = 1
    np.testing.assert_allclose(apply_kernel(IMAGE_3X3, identity), IMAGE_3X3)


def test_5x5_kernel_on_constant_image():
    # Однотонне 2 і ядро 5×5 з одиниць: у кожному вікні 25 пікселів по 2 → 50.
    result = apply_kernel(np.full((5, 5), 2.0), np.ones((5, 5)))
    np.testing.assert_allclose(result, np.full((5, 5), 50.0))


def test_constant_image_gives_zero_gradient():
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    np.testing.assert_allclose(apply_kernel(np.full((4, 6), 7.0), sobel_x), 0)


def test_single_pixel_image():
    assert apply_kernel(np.array([[3.0]]), np.ones((3, 3)))[0, 0] == 27


def test_input_is_not_modified():
    image = IMAGE_3X3.copy()
    apply_kernel(image, np.ones((3, 3)))
    np.testing.assert_array_equal(image, IMAGE_3X3)


def test_even_kernel_is_rejected():
    with pytest.raises(ValueError):
        apply_kernel(IMAGE_3X3, np.ones((2, 2)))
