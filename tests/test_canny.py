"""Тести canny (M7)."""

import numpy as np

from lanes.methods.canny import canny, gaussian_kernel, hysteresis, non_max_suppression


def test_gaussian_kernel_by_hand():
    # Підбираємо σ так, щоб exp(−1 / (2σ²)) = 0.5: σ² = 1 / (2·ln 2).
    # Тоді до нормування: центр exp(0) = 1, сусіди по стороні (відстань² = 1) — 0.5,
    # кутові (відстань² = 2) — 0.5² = 0.25. Сума: 1 + 4·0.5 + 4·0.25 = 4.
    # Після ділення на 4: центр 0.25, сторони 0.125, кути 0.0625.
    sigma = np.sqrt(1 / (2 * np.log(2)))
    expected = [[0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625]]
    np.testing.assert_allclose(gaussian_kernel(3, sigma), expected)


def test_gaussian_kernel_is_normalized_and_symmetric():
    kernel = gaussian_kernel(5, 1.4)
    assert kernel.shape == (5, 5)
    np.testing.assert_allclose(kernel.sum(), 1.0)
    np.testing.assert_allclose(kernel, kernel.T)
    np.testing.assert_allclose(kernel, kernel[::-1, ::-1])
    # Найбільша вага — у центрі.
    assert kernel[2, 2] == kernel.max()


def test_nms_horizontal_gradient_by_hand():
    # Градієнт спрямований уздовж x (кут 0°): порівнюємо з лівим і правим сусідом.
    # Рядок [0, 1, 3, 1, 0]: лише 3 не менший за обох сусідів (1 і 1).
    # Одиниці мають сусіда 3 → стають 0.
    magnitude = np.array([[0, 1, 3, 1, 0]] * 3, dtype=np.float64)
    result = non_max_suppression(magnitude, np.zeros((3, 5)))
    np.testing.assert_array_equal(result, [[0, 0, 3, 0, 0]] * 3)


def test_nms_vertical_gradient_by_hand():
    # Кут 90°: градієнт уздовж y, сусіди — верхній і нижній.
    magnitude = np.array([[0, 1, 3, 1, 0]] * 3, dtype=np.float64).T
    result = non_max_suppression(magnitude, np.full((5, 3), np.pi / 2))
    np.testing.assert_array_equal(result, np.array([[0, 0, 3, 0, 0]] * 3).T)


def test_nms_diagonal_gradient_by_hand():
    # Кут 45° (праворуч і вниз): сусіди — (y − 1, x − 1) і (y + 1, x + 1).
    # Діагональ 1, 5, 2: у центрі 5 ≥ 1 і 5 ≥ 2 → лишається;
    # кут (0, 0): сусід 5 більший → 0; кут (2, 2): сусід 5 більший → 0.
    magnitude = np.array([[1, 0, 0], [0, 5, 0], [0, 0, 2]], dtype=np.float64)
    result = non_max_suppression(magnitude, np.full((3, 3), np.pi / 4))
    np.testing.assert_array_equal(result, [[0, 0, 0], [0, 5, 0], [0, 0, 0]])
    # Кут 135° дивиться на іншу діагональ, де сусіди нульові: усе лишається.
    result = non_max_suppression(magnitude, np.full((3, 3), 3 * np.pi / 4))
    np.testing.assert_array_equal(result, magnitude)


def test_nms_opposite_direction_is_the_same():
    # Кут 180° — той самий напрям, що й 0°, лише в інший бік.
    magnitude = np.array([[0, 1, 3, 1, 0]], dtype=np.float64)
    result = non_max_suppression(magnitude, np.full((1, 5), np.pi))
    np.testing.assert_array_equal(result, [[0, 0, 3, 0, 0]])


def test_hysteresis_by_hand():
    # S — сильний (він же слабкий), w — лише слабкий:
    #   S w . . w        (0, 1) торкається сильного (0, 0) → контур;
    #   . . . . .        (0, 4) ізольований → відкидаємо;
    #   . . w . .        (2, 2) торкається (3, 1) по діагоналі, а той — (4, 0):
    #   . w . . .        ланцюжок підхоплюється за два кроки поширення.
    #   S . . . .
    strong = np.zeros((5, 5), dtype=np.bool_)
    strong[0, 0] = strong[4, 0] = True
    weak = strong.copy()
    weak[0, 1] = weak[0, 4] = weak[2, 2] = weak[3, 1] = True
    expected = weak.copy()
    expected[0, 4] = False
    np.testing.assert_array_equal(hysteresis(strong, weak), expected)


def test_hysteresis_without_strong_pixels():
    weak = np.ones((3, 3), dtype=np.bool_)
    assert not hysteresis(np.zeros((3, 3), dtype=np.bool_), weak).any()


def test_canny_finds_thin_vertical_edge():
    # 9×9: стовпці 0…3 темні, 4…8 світлі — край між x = 3 і x = 4.
    # Згладжування симетричне відносно краю, тож модуль градієнта найбільший
    # у двох стовпцях обабіч нього; потоншення лишає тільки їх.
    image = np.zeros((9, 9))
    image[:, 4:] = 200
    edges = canny(image, kernel_size=5, sigma=1.4, low=0.1, high=0.3)
    assert edges.dtype == np.bool_
    assert edges.shape == (9, 9)
    assert edges[:, 3:5].any(axis=1).all()
    assert not edges[:, :3].any()
    assert not edges[:, 5:].any()


def test_canny_constant_image_is_empty():
    assert not canny(np.full((6, 6), 50.0), 5, 1.4, 0.1, 0.3).any()
    assert not canny(np.zeros((6, 6)), 5, 1.4, 0.1, 0.3).any()


def test_canny_single_pixel():
    edges = canny(np.array([[9.0]]), 5, 1.4, 0.1, 0.3)
    assert edges.shape == (1, 1)
    assert not edges.any()


def test_canny_does_not_change_input():
    image = np.zeros((9, 9))
    image[:, 4:] = 200
    before = image.copy()
    canny(image, 5, 1.4, 0.1, 0.3)
    np.testing.assert_array_equal(image, before)
