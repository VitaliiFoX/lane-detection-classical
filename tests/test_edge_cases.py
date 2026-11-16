"""Крайні випадки для пайплайнів (M12): крихітні й однотонні фото.

Пайплайн не має падати: повертає нуль смуг і всі свої стадії
(``lanes/pipelines/AGENTS.md``). Поки перевіряємо готові пайплайни;
p1 і p2 додамо до ``PIPELINES`` після задач F12 і F13.
"""

import numpy as np
import pytest

from lanes.pipelines import REGISTRY
from lanes.synthetic import make_road

PIPELINES = ["p0", "p3", "p4"]
# Назви стадій з таблиць docs/pipelines.md.
STAGES = {
    "p0": [],
    "p3": ["top_view", "prewitt_magnitude", "edges", "windows"],
    "p4": ["tophat", "canny", "edges_roi", "hough_space"],
}
# (висота, ширина): один піксель, 5×5 і невелике фото.
SHAPES = [(1, 1), (5, 5), (40, 60)]
# Чорне, сіре (однотонне) і біле фото.
VALUES = [0, 128, 255]


@pytest.mark.parametrize("name", PIPELINES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("value", VALUES)
def test_constant_image_gives_no_lanes(config, name, shape, value):
    image = np.full(shape + (3,), value, dtype=np.uint8)
    result = REGISTRY[name].run(image, config)
    # На однотонному фото немає жодного контуру, тож і смуг немає.
    assert not result.lanes
    assert [stage for stage, _ in result.stages] == STAGES[name]


@pytest.mark.parametrize("name", PIPELINES)
def test_tiny_noisy_image_does_not_fail(config, name):
    # 5×5 з випадковими пікселями: результат будь-який, головне — без винятку.
    rng = np.random.default_rng(config["seed"])
    image = rng.integers(0, 256, size=(5, 5, 3), dtype=np.uint8)
    result = REGISTRY[name].run(image, config)
    assert [stage for stage, _ in result.stages] == STAGES[name]


@pytest.mark.parametrize("name", PIPELINES)
def test_input_is_not_changed(config, name):
    image, _ = make_road(64, 36)
    before = image.copy()
    REGISTRY[name].run(image, config)
    np.testing.assert_array_equal(image, before)
