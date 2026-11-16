"""Тести prewitt (M5)."""

import numpy as np

from lanes.methods.prewitt import prewitt_gradients, prewitt_magnitude

# 5×5: два темні стовпці (0) і три світлі (90) — вертикальний край між x = 1 і x = 2.
EDGE = np.array([[0, 0, 90, 90, 90]] * 5, dtype=np.float64)


def test_vertical_edge_by_hand():
    gx, gy = prewitt_gradients(EDGE)
    # Піксель (2, 2): ліві сусіди (x = 1) дорівнюють 0, праві (x = 3) — 90,
    # три рядки з вагою 1: gx = 3 · (90 − 0) = 270.
    assert gx[2, 2] == 270
    # Піксель (2, 1): ліворуч x = 0 → 0, праворуч x = 2 → 90 → теж 270.
    assert gx[2, 1] == 270
    # x = 3: ліворуч 90 і праворуч 90 → 0; x = 0: ліворуч повтор краю 0, праворуч 0 → 0.
    assert gx[2, 3] == 0
    assert gx[2, 0] == 0
    # Уздовж стовпців яскравість не змінюється → gy = 0 скрізь.
    assert not gy.any()


def test_diagonal_step_by_hand():
    # Світлий правий нижній кут. Для центру (1, 1):
    # gx = (0 − 0) + (10 − 0) + (10 − 0) = 20 (праві сусіди мінус ліві по рядках),
    # gy = (0 − 0) + (10 − 0) + (10 − 0) = 20 (нижні мінус верхні по стовпцях).
    image = np.array([[0, 0, 0], [0, 10, 10], [0, 10, 10]], dtype=np.float64)
    gx, gy = prewitt_gradients(image)
    assert gx[1, 1] == 20
    assert gy[1, 1] == 20
    # Модуль: sqrt(20² + 20²) = 20·√2.
    np.testing.assert_allclose(prewitt_magnitude(image)[1, 1], 20 * np.sqrt(2))


def test_magnitude_of_transposed_edge():
    assert prewitt_magnitude(EDGE)[2, 2] == 270
    # Транспонований край — горизонтальний: gy = 270, модуль той самий.
    assert prewitt_magnitude(EDGE.T)[2, 2] == 270


def test_constant_image_has_no_edges():
    assert not prewitt_magnitude(np.full((4, 4), 123.0)).any()


def test_single_pixel():
    assert prewitt_magnitude(np.array([[7.0]])).shape == (1, 1)
