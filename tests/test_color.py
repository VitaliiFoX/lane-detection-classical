"""Тести color (F6): сіре, HSV, маска білого й жовтого."""

import numpy as np
import pytest

from lanes.methods.color import rgb_to_gray, rgb_to_hsv, white_yellow_mask

# Один рядок пікселів: червоний, зелений, синій, жовтий, пурпуровий, білий, чорний, сірий.
PIXELS = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                    [255, 0, 255], [255, 255, 255], [0, 0, 0], [128, 128, 128]]],
                  dtype=np.uint8)


def test_gray_by_hand():
    gray = rgb_to_gray(PIXELS)
    # Червоний: 0.299·255 = 76.245; білий: (0.299 + 0.587 + 0.114)·255 = 255.
    assert gray[0, 0] == pytest.approx(76.245)
    assert gray[0, 5] == pytest.approx(255)
    assert gray[0, 6] == 0


def test_hsv_by_hand():
    h, s, v = rgb_to_hsv(PIXELS)
    # Відтінки: червоний 0°, зелений 120°, синій 240°;
    # жовтий: max = R = G, (G − B)/δ = 1 → 60°; пурпуровий: (G − B)/δ = −1 → −60° mod 360 = 300°.
    np.testing.assert_allclose(h[0, :5], [0, 120, 240, 60, 300])
    # Чисті кольори насичені (s = 1) і найяскравіші (v = 1).
    np.testing.assert_allclose(s[0, :5], 1)
    np.testing.assert_allclose(v[0, :5], 1)
    # Білий: s = 0, v = 1; чорний: s = 0, v = 0; сірий 128: s = 0, v = 128/255.
    assert (h[0, 5], s[0, 5], v[0, 5]) == (0, 0, 1)
    assert (h[0, 6], s[0, 6], v[0, 6]) == (0, 0, 0)
    assert s[0, 7] == 0
    assert v[0, 7] == pytest.approx(128 / 255)


def test_white_and_yellow_mask():
    mask = white_yellow_mask(PIXELS, white_s_max=0.2, white_v_min=0.75,
                             yellow_h_min=35, yellow_h_max=65,
                             yellow_s_min=0.35, yellow_v_min=0.5)
    # Лише жовтий (60°) і білий; сірий 128 занадто темний (v = 0.5 < 0.75).
    np.testing.assert_array_equal(mask[0], [False, False, False, True,
                                            False, True, False, False])


def test_uniform_image():
    image = np.full((2, 3, 3), 40, dtype=np.uint8)
    h, s, _ = rgb_to_hsv(image)
    assert not h.any()
    assert not s.any()
