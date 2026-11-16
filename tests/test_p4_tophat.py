"""Тести пайплайна p4 «Top-hat» (M9)."""

import numpy as np
import pytest

from lanes.checks import synthetic_offset, synthetic_scenarios
from lanes.pipelines.p4_tophat import run
from lanes.roi import roi_top
from lanes.synthetic import make_road

SCENARIOS = ["synthetic_straight", "synthetic_dark", "synthetic_noise"]
STAGES = ["tophat", "canny", "edges_roi", "hough_space"]


def test_stages_and_lanes_contract(config):
    image, _ = make_road(*config["checks"]["synthetic_size"])
    height, width = image.shape[:2]
    result = run(image, config)
    assert [name for name, _ in result.stages] == STAGES

    stages = dict(result.stages)
    assert stages["tophat"].shape == (height, width)
    assert stages["canny"].dtype == np.bool_
    # ROI лише прибирає контури, нових не додає.
    assert not (stages["edges_roi"] & ~stages["canny"]).any()
    assert stages["edges_roi"].any()

    # Дві прямі: кожна від нижнього краю фото до верхньої межі ROI.
    assert len(result.lanes) == 2
    y_top = roi_top((height, width), config["roi"]["polygon"])
    for lane in result.lanes:
        assert lane.shape == (2, 2)
        assert lane[0, 1] == height - 1
        assert lane[1, 1] == y_top


def test_wide_bright_object_is_removed_by_tophat(config):
    # Світлий прямокутник, ширший за структурний елемент, — не розмітка:
    # top-hat його стирає, тож контурів усередині нього немає й смуг теж.
    width, height = config["checks"]["synthetic_size"]
    image = np.full((height, width, 3), 80, dtype=np.uint8)
    image[height // 2:, 100:220] = 255
    result = run(image, config)
    assert not dict(result.stages)["tophat"].any()
    assert not result.lanes


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_synthetic_scenarios(config, scenario):
    # Пороги Кенні — частки максимуму градієнта, тож темний кадр дає ті самі контури.
    offset = synthetic_offset(run, config, synthetic_scenarios(config)[scenario])
    assert offset <= config["checks"]["offset_max_px"]
