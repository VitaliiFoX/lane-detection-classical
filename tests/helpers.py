"""Спільні перевірки для тестів пайплайнів Фояка (p1, p2)."""

import numpy as np

from lanes.roi import roi_top


def assert_lanes_span_roi(lanes: list[np.ndarray], shape: tuple[int, int], config: dict) -> None:
    """Кожна смуга Хафа — відрізок від нижнього краю фото до верху ROI.

    На синтетичному кадрі дві смуги, тож знайдено має бути щонайменше дві.
    """
    height, width = shape
    y_top = roi_top((height, width), config["roi"]["polygon"])
    assert len(lanes) >= 2
    for lane in lanes:
        assert lane.shape == (2, 2)
        assert (lane[0, 1], lane[1, 1]) == (height - 1, y_top)
