"""Збирає вибірку зі 100 фото CULane. Задача F3, власник: Фояк Віталій.

Запуск з кореня репозиторію: ``python -m tools.make_sample``. Що саме робить —
``docs/data.md``, розділ «Наша вибірка».
"""

import csv
import shutil

import numpy as np

from lanes.config import load_config, project_path
from lanes.io_utils import read_lanes
from lanes.runner import lines_path


def make_sample(config: dict) -> int:
    """Відбирає ``per_group`` кадрів кожної групи з ``seed`` і копіює їх у ``sample_dir``.

    Беремо лише кадри з архівів ``data.archives``, у яких ``.lines.txt``
    містить хоча б одну смугу. Пише ``data/sample.csv`` (``group,image,source``)
    і повертає кількість відібраних фото. Якщо CULane не знайдено — пояснює,
    що завантажити.
    """
    data = config["data"]
    culane = project_path(data["culane_root"])
    sample_dir = project_path(data["sample_dir"])
    lists = culane / "list" / "test_split"
    if not lists.is_dir():
        raise FileNotFoundError(f"Немає {lists}: завантажте CULane за docs/data.md")

    # Один генератор на всі групи з seed з конфігурації: той самий CULane
    # завжди дає ту саму вибірку.
    rng = np.random.default_rng(config["seed"])
    rows = []
    for group, list_name in data["groups"].items():
        # У списках CULane шляхи починаються з «/», відносно кореня набору.
        listed = [path.lstrip("/") for path in
                  (lists / list_name).read_text(encoding="utf-8").split()]
        # Перша тека шляху — архів CULane, з якого кадр (docs/data.md, чому не всі).
        candidates = [path for path in listed
                      if path.split("/")[0] in data["archives"]
                      and (culane / path).is_file()
                      and read_lanes(lines_path(culane / path))]
        if len(candidates) < data["per_group"]:
            raise ValueError(f"Група {group}: знайдено лише {len(candidates)} кадрів зі смугами, "
                             f"потрібно {data['per_group']}")

        chosen = rng.choice(len(candidates), size=data["per_group"], replace=False)
        for index in sorted(chosen):
            source = candidates[index]
            # Назва файлу — шлях у CULane, де «/» замінено на «__»: так кадри з
            # різних кліпів з однаковою назвою 00000.jpg не перезапишуть одне одного.
            name = source.replace("/", "__")
            target = sample_dir / group / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(culane / source, target)
            shutil.copyfile(lines_path(culane / source), lines_path(target))
            rows.append({"group": group, "image": f"{group}/{name}", "source": source})

    csv_path = project_path(data["sample_csv"])
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["group", "image", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Відібрано {len(rows)} фото → {sample_dir}, список → {csv_path}")
    return len(rows)


if __name__ == "__main__":
    try:
        make_sample(load_config())
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
