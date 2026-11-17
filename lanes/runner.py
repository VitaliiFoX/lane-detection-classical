"""Обробка вибірки: стадії, маски, метрики. Задача F11, власник: Фояк Віталій.

Потік даних — ``ARCHITECTURE.md``, розділ 4; формат ``metrics.csv`` —
``docs/metrics.md``.
"""

import csv
import time
from pathlib import Path

import numpy as np

from lanes.config import project_path
from lanes.draw import lanes_mask, overlay
from lanes.io_utils import ImageLoadError, load_rgb, read_lanes, save_png
from lanes.metrics import iou, mse, psnr, ssim
from lanes.pipelines import REGISTRY

METRICS_FIELDS = ["pipeline", "group", "image", "mse", "psnr", "ssim", "iou",
                  "lanes_found", "lanes_gt", "time_ms"]

# Помилки, які пайплайн може кинути на незвичному фото. Ловимо саме їх, а не
# будь-що: прогін ста фото не має зупинятися через одне, але помилку в самому
# раннері (наприклад, немає теки) ховати не можна. NotImplementedError —
# підклас RuntimeError, тож ще не написаний пайплайн теж не зупиняє прогін.
PIPELINE_ERRORS = (ArithmeticError, LookupError, ValueError, TypeError, RuntimeError)


def read_sample(path: Path) -> list[dict]:
    """Читає ``data/sample.csv`` (``group,image,source``)."""
    if not path.is_file():
        raise FileNotFoundError(f"Немає {path}: спершу зберіть вибірку (docs/data.md)")
    with open(path, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def lines_path(image_path: Path) -> Path:
    """Розмітка CULane лежить поруч із фото: ``кадр.jpg`` → ``кадр.lines.txt``."""
    return image_path.with_suffix(".lines.txt")


def _read_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRICS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _empty_row(name: str, sample: dict, lanes_gt: int | str = "") -> dict:
    """Рядок без метрик: фото не прочиталося або пайплайн упав."""
    row = dict.fromkeys(METRICS_FIELDS, "")
    row.update(pipeline=name, group=sample["group"], image=sample["image"], lanes_gt=lanes_gt)
    return row


def _run_one(name: str, image: np.ndarray, gt_lanes: list[np.ndarray], gt_mask: np.ndarray,
             sample: dict, config: dict, output: Path) -> dict:
    """Один пайплайн на одному фото: запуск, збереження, рядок метрик."""
    start = time.perf_counter()
    try:
        result = REGISTRY[name].run(image, config)
    except PIPELINE_ERRORS as error:
        print(f"    {name}: помилка {type(error).__name__}: {error}")
        return _empty_row(name, sample, len(gt_lanes))
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Маску знайдених смуг малюємо так само, як еталонну (ADR-0005).
    found_mask = lanes_mask(image.shape[:2], result.lanes, config["masks"]["thickness"])

    folder = output / name / sample["group"] / Path(sample["image"]).stem
    for index, (stage, array) in enumerate(result.stages, start=1):
        save_png(folder / f"{index:02d}_{stage}.png", array)
    save_png(folder / "mask.png", found_mask)
    save_png(folder / "overlay.png", overlay(image, result.lanes, gt_lanes))

    settings = config["metrics"]
    similarity = ssim(gt_mask, found_mask, settings["ssim_window"],
                      settings["ssim_k1"], settings["ssim_k2"])
    return {
        "pipeline": name,
        "group": sample["group"],
        "image": sample["image"],
        "mse": f"{mse(gt_mask, found_mask):.4f}",
        "psnr": f"{psnr(gt_mask, found_mask, settings['psnr_cap']):.4f}",
        "ssim": f"{similarity:.4f}",
        "iou": f"{iou(gt_mask, found_mask):.4f}",
        "lanes_found": len(result.lanes),
        "lanes_gt": len(gt_lanes),
        "time_ms": f"{elapsed_ms:.1f}",
    }


def run_pipelines(config: dict, names: list[str], limit: int | None = None) -> Path:
    """Запускає пайплайни ``names`` на фото з ``data/sample.csv``.

    ``limit`` — обробити лише перші N фото (пілот). Для кожного фото
    зберігає ``results/<пайплайн>/<група>/<фото>/NN_<стадія>.png``,
    ``mask.png`` і ``overlay.png``, рахує метрики проти розмітки й час.
    Пайплайн, що впав, і битий файл не зупиняють прогін: рядок з порожніми
    метриками і повідомлення в консоль. Рядки інших пайплайнів з
    попередніх запусків у ``metrics.csv`` лишаються. Повертає шлях до файлу.
    """
    sample_dir = project_path(config["data"]["sample_dir"])
    if not sample_dir.is_dir():
        # Без цієї перевірки кожне зі 100 фото дало б окреме «пропущено».
        raise FileNotFoundError(f"Немає фото в {sample_dir}: розпакуйте sample.zip "
                                "або зберіть вибірку (docs/data.md)")
    samples = read_sample(project_path(config["data"]["sample_csv"]))[:limit]
    output = project_path(config["output_dir"])
    metrics_path = output / "metrics.csv"

    kept = [row for row in _read_metrics(metrics_path) if row["pipeline"] not in names]
    rows = []
    for number, sample in enumerate(samples, start=1):
        image_path = sample_dir / sample["image"]
        print(f"[{number}/{len(samples)}] {sample['image']}")
        try:
            image = load_rgb(image_path)
        except ImageLoadError as error:
            print(f"    пропущено: {error}")
            rows.extend(_empty_row(name, sample) for name in names)
            continue
        gt_lanes = read_lanes(lines_path(image_path))
        gt_mask = lanes_mask(image.shape[:2], gt_lanes, config["masks"]["thickness"])
        for name in names:
            rows.append(_run_one(name, image, gt_lanes, gt_mask, sample, config, output))

    _write_metrics(metrics_path, kept + rows)
    print(f"Метрики: {metrics_path}")
    return metrics_path
