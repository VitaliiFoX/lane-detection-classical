"""Каркас проєкту: конфігурація читається, пайплайни зареєстровані."""

import numpy as np

from lanes.pipelines import REGISTRY


def test_config_has_all_sections(config):
    for key in ("data", "roi", "masks", "metrics", "hough", "p1", "p2", "p3", "p4", "checks"):
        assert key in config


def test_registry_lists_all_pipelines():
    assert list(REGISTRY) == ["p0", "p1", "p2", "p3", "p4"]


def test_baseline_finds_nothing(config):
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    result = REGISTRY["p0"].run(image, config)
    assert not result.lanes
    assert not result.stages
