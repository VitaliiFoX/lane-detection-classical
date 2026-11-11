"""Збирає вибірку зі 100 фото CULane. Задача F3, власник: Фояк Віталій.

Запуск з кореня репозиторію: ``python -m tools.make_sample``. Що саме робить —
``docs/data.md``, розділ «Наша вибірка».
"""

from lanes.config import load_config


def make_sample(config: dict) -> int:
    """Відбирає ``per_group`` кадрів кожної групи з ``seed`` і копіює їх у ``sample_dir``.

    Беремо лише кадри, у яких ``.lines.txt`` містить хоча б одну смугу.
    Пише ``data/sample.csv`` (``group,image,source``) і повертає кількість
    відібраних фото. Якщо CULane не знайдено — пояснює, що завантажити.
    """
    raise NotImplementedError("F3")


if __name__ == "__main__":
    make_sample(load_config())
