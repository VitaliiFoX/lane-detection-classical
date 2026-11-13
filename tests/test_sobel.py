"""Тести sobel (F7)."""

import numpy as np

from lanes.methods.sobel import sobel_gradients, sobel_magnitude

# 5×5: два темні стовпці (0) і три світлі (90) — вертикальний край між x = 1 і x = 2.
EDGE = np.array([[0, 0, 90, 90, 90]] * 5, dtype=np.float64)


def test_vertical_edge_by_hand():
    gx, gy = sobel_gradients(EDGE)
    # Центр (2, 2): ліві сусіди 0, праві 90, рядки з вагами 1, 2, 1:
    # gx = 1·(90 − 0) + 2·(90 − 0) + 1·(90 − 0) = 360.
    assert gx[2, 2] == 360
    # Піксель (2, 1) теж біля краю: ліворуч 0, праворуч 90 → теж 360.
    assert gx[2, 1] == 360
    # Далеко від краю (x = 4) і на лівому краю (x = 0) все однакове → 0.
    assert gx[2, 4] == 0
    assert gx[2, 0] == 0
    # Уздовж стовпців яскравість не змінюється → gy = 0 скрізь.
    assert not gy.any()


def test_magnitude():
    # gx = 360, gy = 0 → модуль 360.
    assert sobel_magnitude(EDGE)[2, 2] == 360
    # Транспонований край — горизонтальний: тепер gy = 360, а модуль той самий.
    assert sobel_magnitude(EDGE.T)[2, 2] == 360


def test_constant_image_has_no_edges():
    assert not sobel_magnitude(np.full((4, 4), 123.0)).any()
