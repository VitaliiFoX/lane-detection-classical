"""Тести обробки для вікна gui.py (саме вікно не відкриваємо)."""

import numpy as np
import pytest

from gui import STEPS, draw_roi, number, process
from lanes.synthetic import make_road


def test_process_shows_all_steps(config):
    image, lanes = make_road(*config["checks"]["synthetic_size"])
    steps, scores = process(image, lanes, "p1", config)
    titles = [title for title, _, _ in steps]
    # Фото, зона пошуку, 4 стадії p1, маска, результат — кожен крок з поясненням.
    assert titles == ["Фото", "Зона пошуку", "Біла й жовта фарба", "Межі плям",
                      "Межі в зоні пошуку", "Голосування за прямі", "Знайдене (маска)",
                      "Результат"]
    assert all(text for _, _, text in steps)
    assert 0.0 <= scores["iou"] <= 1.0
    assert scores["expected"] == len(lanes)


@pytest.mark.parametrize("name", ["p2", "p3", "p4"])
def test_every_stage_has_plain_name(config, name):
    # Назва стадії з коду (наприклад, «tophat») має бути перекладена простими словами.
    image, lanes = make_road(*config["checks"]["synthetic_size"])
    steps, _ = process(image, lanes, name, config)
    assert all(title in {plain for plain, _ in STEPS.values()} for title, _, _ in steps)


def test_baseline_finds_nothing(config):
    image, lanes = make_road(*config["checks"]["synthetic_size"])
    _, scores = process(image, lanes, "p0", config)
    assert scores["iou"] == 0.0 and scores["found"] == 0


def test_draw_roi_keeps_input():
    # Сіре фото 10×10: контур ROI змінює копію, а не вхідний масив.
    image = np.full((10, 10, 3), 100, dtype=np.uint8)
    framed = draw_roi(image, {"roi": {"polygon": [[0, 1], [0.5, 0], [1, 1]]}})
    assert (image == 100).all()
    assert (framed != 100).any()


def test_number_uses_comma():
    assert number(0.4567) == "0,46"
    assert number(58.7, 0) == "59"
