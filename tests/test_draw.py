"""Тести draw (F4): маски смуг і фото з лініями."""

import numpy as np

from lanes.draw import lanes_mask, overlay


def test_horizontal_lane_has_given_thickness():
    lane = np.array([[0.0, 10.0], [19.0, 10.0]])
    mask = lanes_mask((20, 20), [lane], thickness=4)
    white_rows = np.nonzero((mask == 255).any(axis=1))[0]
    assert len(white_rows) == 4
    assert set(np.unique(mask)) == {0, 255}


def test_vertical_lane_position():
    lane = np.array([[5.0, 0.0], [5.0, 9.0]])
    mask = lanes_mask((10, 12), [lane], thickness=1)
    # Лінія товщиною 1 — рівно стовпець x = 5 на всю висоту.
    assert (mask[:, 5] == 255).all()
    assert mask.sum() == 10 * 255


def test_no_lanes_gives_black_mask():
    mask = lanes_mask((7, 9), [], thickness=16)
    assert mask.shape == (7, 9)
    assert mask.dtype == np.uint8
    assert not mask.any()


def test_overlay_draws_colors_and_keeps_input():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    predicted = [np.array([[2.0, 0.0], [2.0, 19.0]])]
    ground_truth = [np.array([[15.0, 0.0], [15.0, 19.0]])]
    result = overlay(image, predicted, ground_truth)
    assert tuple(result[10, 2]) == (255, 0, 0)
    assert tuple(result[10, 15]) == (0, 255, 0)
    assert not image.any()
