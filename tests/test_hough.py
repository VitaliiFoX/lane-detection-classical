"""Тести hough (F8)."""

import numpy as np

from lanes.methods.hough import accumulator, detect_lanes, find_peaks, peaks_to_lanes

PARAMS = {"rho_step": 1, "theta_step_deg": 1, "min_angle_deg": 20, "max_lanes": 4,
          "min_votes": 10, "suppress_rho": 20, "suppress_theta_deg": 5}


def test_vertical_line_by_hand():
    # Вертикальна лінія x = 2 у 5×5. Для θ = 0°: ρ = x·cos 0 + y·sin 0 = x = 2
    # для всіх п'яти пікселів → 5 голосів у комірці (ρ = 2, θ = 0°).
    edges = np.zeros((5, 5), dtype=bool)
    edges[:, 2] = True
    acc, rhos, thetas = accumulator(edges, rho_step=1, theta_step_deg=1, min_angle_deg=20)
    i_rho = int(np.nonzero(rhos == 2)[0][0])
    i_theta = int(np.nonzero(thetas == 0)[0][0])
    assert acc[i_rho, i_theta] == 5
    # Усього голосів = білих пікселів × кутів: кожен піксель голосує раз за кожен θ.
    assert acc.sum() == 5 * len(thetas)
    # Відрізок від нижнього рядка y = 4 до y_top = 1: x = (2 − y·0) / 1 = 2.
    lane = peaks_to_lanes([(i_rho, i_theta)], rhos, thetas, height=5, y_top=1)[0]
    np.testing.assert_allclose(lane, [[2, 4], [2, 1]])


def test_vertical_line_peak():
    # На малому 5×5 кілька сусідніх θ набирають ті самі 5 голосів через
    # округлення ρ, тож пік перевіряємо на довшій лінії: x = 20 у 50×50.
    edges = np.zeros((50, 50), dtype=bool)
    edges[:, 20] = True
    acc, rhos, thetas = accumulator(edges, rho_step=1, theta_step_deg=1, min_angle_deg=20)
    i_rho, i_theta = find_peaks(acc, max_peaks=1, min_votes=3,
                                suppress_rho_bins=2, suppress_theta_bins=2)[0]
    assert (rhos[i_rho], thetas[i_theta], acc[i_rho, i_theta]) == (20, 0, 50)


def test_diagonal_line():
    # Діагональ y = x: нормаль під 135°, ρ = (−x + y)/√2 = 0.
    edges = np.eye(50, dtype=bool)
    lanes, acc_image = detect_lanes(edges, y_top=10, params={**PARAMS, "max_lanes": 1})
    assert len(lanes) == 1
    # На діагоналі x = y: нижня точка (49, 49), верхня (10, 10).
    np.testing.assert_allclose(lanes[0], [[49, 49], [10, 10]], atol=1.0)
    assert acc_image.dtype == np.uint8
    assert acc_image.max() == 255


def test_horizontal_line_is_ignored():
    # Горизонтальна лінія має кут 0° до горизонталі < min_angle_deg = 20°.
    edges = np.zeros((50, 50), dtype=bool)
    edges[25, :] = True
    lanes, _ = detect_lanes(edges, y_top=10, params=PARAMS)
    assert not lanes


def test_two_lines_two_peaks():
    edges = np.zeros((60, 60), dtype=bool)
    edges[:, 10] = True
    edges[:, 45] = True
    # Поріг 30 голосів (половина висоти): слабкі «відлуння» лінії під сусідніми
    # кутами, що лишилися за межею придушення, набирають ~10 голосів і відсікаються.
    lanes, _ = detect_lanes(edges, y_top=0, params={**PARAMS, "min_votes": 30})
    bottoms = sorted(round(lane[0, 0]) for lane in lanes)
    assert bottoms == [10, 45]


def test_empty_mask():
    lanes, acc_image = detect_lanes(np.zeros((8, 8), dtype=bool), y_top=2, params=PARAMS)
    assert not lanes
    assert not acc_image.any()


def test_suppression_keeps_peaks_apart():
    acc = np.zeros((10, 10), dtype=np.int64)
    acc[5, 5], acc[5, 6], acc[1, 1] = 9, 8, 7
    # Сусід (5, 6) придушено разом із піком (5, 5); далекий (1, 1) лишається.
    assert find_peaks(acc, 3, 1, 1, 1) == [(5, 5), (1, 1)]
