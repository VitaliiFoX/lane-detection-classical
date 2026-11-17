"""Тести пайплайна p3 «Вид зверху» (M8)."""

import numpy as np
import pytest

from lanes.checks import synthetic_offset, synthetic_scenarios
from lanes.pipelines.p3_top_view import run
from lanes.synthetic import make_road

SCENARIOS = ["synthetic_straight", "synthetic_dark", "synthetic_noise"]
STAGES = ["top_view", "prewitt_magnitude", "edges", "windows"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_synthetic_scenarios(config, scenario):
    # Поріг Прюітта — частка максимуму, тож темний кадр дає ті самі контури.
    offset = synthetic_offset(run, config, synthetic_scenarios(config)[scenario])
    assert offset <= config["checks"]["offset_max_px"]


def test_stages_and_lanes_contract(config):
    width, height = config["checks"]["synthetic_size"]
    image, _ = make_road(width, height)
    result = run(image, config)
    assert [name for name, _ in result.stages] == STAGES

    out_width, out_height = config["p3"]["warp_size"]
    stages = dict(result.stages)
    assert stages["top_view"].shape == (out_height, out_width)
    assert stages["edges"].dtype == np.bool_
    assert stages["windows"].shape == (out_height, out_width, 3)

    # Дві смуги в координатах фото, а не виду зверху: n_windows + 1 точок
    # кривої і ще одна — продовження до нижнього краю фото.
    assert len(result.lanes) == 2
    for lane in result.lanes:
        assert lane.shape == (config["p3"]["n_windows"] + 2, 2)
        assert lane[-1, 1] == height - 1
        assert lane[:, 0].min() >= 0 and lane[:, 0].max() <= width - 1
        assert lane[:, 1].min() >= 0 and lane[:, 1].max() <= height - 1


def test_photo_border_is_not_an_edge(config):
    # Асфальт без розмітки: єдиний перепад яскравості на виді зверху — межа
    # між фото й порожніми кутами. Вона не має стати контуром чи смугою.
    image = np.full((180, 320, 3), 80, dtype=np.uint8)
    result = run(image, config)
    assert not dict(result.stages)["edges"].any()
    assert not result.lanes
