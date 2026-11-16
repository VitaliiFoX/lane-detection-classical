"""Тести sliding_windows (M6)."""

import numpy as np

from lanes.methods.sliding_windows import (column_histogram, detect_lanes_top_view,
                                           find_base_peaks, track_lane)

PARAMS = {"max_lanes": 4, "peak_min_distance": 10, "min_pixels": 3,
          "n_windows": 6, "window_margin": 4, "poly_degree": 2}


def _two_lines() -> np.ndarray:
    """Маска 60 рядків × 40 стовпців із двома вертикальними смугами x = 10 і x = 30."""
    edges = np.zeros((60, 40), dtype=np.bool_)
    edges[:, 10] = True
    edges[:, 30] = True
    return edges


def test_histogram_counts_lower_half_by_hand():
    # 4×5: нижня половина — рядки 2 і 3.
    edges = np.array([[1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 1],
                      [0, 1, 0, 1, 1]], dtype=np.bool_)
    # Стовпець 0: 0 + 0 = 0; 1: 1 + 1 = 2; 2: 0; 3: 0 + 1 = 1; 4: 1 + 1 = 2.
    # Рядки 0 і 1 (верхня половина) не рахуються.
    np.testing.assert_array_equal(column_histogram(edges), [0, 2, 0, 1, 2])


def test_peaks_by_hand():
    #            x:  0  1  2  3  4  5  6  7  8  9
    histogram = np.array([0, 5, 9, 4, 0, 0, 0, 7, 2, 0])
    # 1) максимум 9 у x = 2; обнуляємо x = 0…4 (ближче за 3);
    # 2) максимум 7 у x = 7; обнуляємо x = 5…9; 3) лишилися нулі — стоп.
    assert find_base_peaks(histogram, max_lanes=4, min_distance=3, min_height=3) == [2, 7]
    # Поріг висоти 8 пропускає тільки перший пік.
    assert find_base_peaks(histogram, max_lanes=4, min_distance=3, min_height=8) == [2]
    # Обмеження кількості: лише найвищий.
    assert find_base_peaks(histogram, max_lanes=1, min_distance=3, min_height=3) == [2]


def test_peaks_closer_than_min_distance_merge():
    # Сусідні стовпці 4 і 5 — два краї однієї смуги: з відстанню 3 це один пік.
    histogram = np.array([0, 0, 0, 0, 8, 6, 0, 0, 0, 0])
    assert find_base_peaks(histogram, max_lanes=4, min_distance=3, min_height=1) == [4]


def test_peaks_of_empty_histogram():
    assert not find_base_peaks(np.zeros(10, dtype=np.int64), 4, 3, 1)
    assert not find_base_peaks(np.array([], dtype=np.int64), 4, 3, 1)


def test_track_vertical_line():
    xs, ys, windows = track_lane(_two_lines(), x_base=10, n_windows=6, margin=4, min_pixels=3)
    # 6 вікон по 60 // 6 = 10 рядків покривають усю висоту: зібрано всі 60 пікселів смуги.
    assert xs.size == 60
    assert set(xs.tolist()) == {10}
    assert sorted(ys.tolist()) == list(range(60))
    # Перше вікно — найнижче: x від 10 − 4 до 10 + 4, y від 50 до 60.
    assert windows[0] == (6, 50, 14, 60)
    assert windows[-1] == (6, 0, 14, 10)


def test_track_follows_slanted_line():
    # Смуга зсувається на 1 стовпець праворуч кожні 5 рядків знизу вгору:
    # рядок y має x = 5 + (59 − y) // 5, тобто від x = 5 унизу до x = 16 угорі.
    edges = np.zeros((60, 40), dtype=np.bool_)
    rows = np.arange(60)
    edges[rows, 5 + (59 - rows) // 5] = True
    xs, _, windows = track_lane(edges, x_base=5, n_windows=6, margin=4, min_pixels=3)
    # Вікна зсуваються за смугою, тож жоден із 60 пікселів не загублено,
    # хоча верх смуги (x = 16) далеко за межами першого вікна (x < 9).
    assert xs.size == 60
    assert windows[-1][0] > windows[0][0]


def test_two_vertical_lines_give_two_lanes():
    lanes, debug = detect_lanes_top_view(_two_lines(), PARAMS)
    assert len(lanes) == 2
    # Кожна смуга — n_windows + 1 = 7 точок (x, y) від y = 0 до y = 59.
    assert lanes[0].shape == (7, 2)
    np.testing.assert_allclose(lanes[0][:, 1], np.linspace(0, 59, 7))
    # Парабола через пікселі з однаковим x вироджується в пряму x = const.
    np.testing.assert_allclose(lanes[0][:, 0], 10, atol=1e-6)
    np.testing.assert_allclose(lanes[1][:, 0], 30, atol=1e-6)
    assert debug.shape == (60, 40, 3)
    assert debug.dtype == np.uint8


def test_empty_mask_gives_no_lanes():
    lanes, debug = detect_lanes_top_view(np.zeros((60, 40), dtype=np.bool_), PARAMS)
    assert not lanes
    assert debug.shape == (60, 40, 3)
    assert not debug.any()


def test_tiny_mask_does_not_fail():
    # Маска 1×1 і маска, нижча за кількість вікон: висота вікна 0, смуг немає.
    for shape in ((1, 1), (3, 5)):
        lanes, debug = detect_lanes_top_view(np.ones(shape, dtype=np.bool_), PARAMS)
        assert not lanes
        assert debug.shape == shape + (3,)
