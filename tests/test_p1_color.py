"""Тести пайплайна p1 «Колір» (F12). Крайні входи — tests/test_edge_cases.py."""

import numpy as np
import pytest

from lanes.checks import synthetic_offset, synthetic_scenarios
from lanes.pipelines.p1_color import run
from lanes.synthetic import make_road
from tests.helpers import assert_lanes_span_roi


def test_stages_and_lanes_contract(config):
    image, _ = make_road(*config["checks"]["synthetic_size"])
    result = run(image, config)
    stages = dict(result.stages)
    assert list(stages) == ["hsv_mask", "sobel_magnitude", "edges_roi", "hough_space"]

    assert stages["hsv_mask"].dtype == np.bool_
    # На синтетичному кадрі білі лише смуги: небо блакитне, асфальт сірий і темний.
    assert (image[stages["hsv_mask"]] == 255).all()
    assert stages["edges_roi"].any()
    # ROI лише прибирає контури — поза ним їх немає.
    assert not stages["edges_roi"][:image.shape[0] // 3].any()
    assert_lanes_span_roi(result.lanes, image.shape[:2], config)


@pytest.mark.parametrize("scenario", ["synthetic_straight", "synthetic_noise"])
def test_synthetic_scenarios(config, scenario):
    offset = synthetic_offset(run, config, synthetic_scenarios(config)[scenario])
    assert offset <= config["checks"]["offset_max_px"]


def test_dark_frame_is_a_known_limitation(config):
    # Обмеження методу, а не помилка: у темряві біла фарба має V = 0.4 · 1 = 0.4,
    # менше за white_v_min = 0.75, тож маска порожня і смуг немає.
    # Цей кейс C1 p1 не проходить — це результат для обговорення у звіті.
    offset = synthetic_offset(run, config, synthetic_scenarios(config)["synthetic_dark"])
    assert offset == float("inf")
