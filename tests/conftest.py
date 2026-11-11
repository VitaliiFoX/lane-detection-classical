"""Спільні фікстури тестів."""

import pytest

from lanes.config import load_config


@pytest.fixture(name="config")
def config_fixture() -> dict:
    """Конфігурація проєкту з ``config.json``."""
    return load_config()
