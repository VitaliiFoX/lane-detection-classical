"""Тести пайплайна p2 «Оцу» (F13). Крайні входи — tests/test_edge_cases.py."""

import pytest

from lanes.checks import synthetic_offset, synthetic_scenarios
from lanes.pipelines.p2_otsu import run
from lanes.roi import roi_mask
from lanes.synthetic import make_road
from tests.helpers import assert_lanes_span_roi


def test_stages_and_lanes_contract(config):
    image, _ = make_road(*config["checks"]["synthetic_size"])
    result = run(image, config)
    stages = dict(result.stages)
    assert list(stages) == ["equalized", "otsu", "hough_space"]

    # Вирівнювання розтягує яскравість на весь діапазон.
    assert (stages["equalized"].min(), stages["equalized"].max()) == (0, 255)
    # Поза ROI білих пікселів немає.
    outside = ~roi_mask(image.shape[:2], config["roi"]["polygon"])
    assert stages["otsu"].any()
    assert not (stages["otsu"] & outside).any()
    assert_lanes_span_roi(result.lanes, image.shape[:2], config)


def test_second_otsu_iteration_keeps_only_lanes(config):
    # У ROI синтетичного кадру три рівні: асфальт, смуга неба під верхом
    # трапеції і білі смуги. Перша ітерація відділяє асфальт, друга — небо,
    # тож лишаються самі смуги: усі білі пікселі маски — білі пікселі фото.
    image, _ = make_road(*config["checks"]["synthetic_size"])
    otsu = dict(run(image, config).stages)["otsu"]
    assert (image[otsu] == 255).all()


@pytest.mark.parametrize("scenario", ["synthetic_straight", "synthetic_dark", "synthetic_noise"])
def test_synthetic_scenarios(config, scenario):
    # Темний кадр вирівнювання гістограми розтягує назад до 0…255, тож p2 його проходить.
    offset = synthetic_offset(run, config, synthetic_scenarios(config)[scenario])
    assert offset <= config["checks"]["offset_max_px"]
