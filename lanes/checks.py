"""Еталонні сценарії для критерію C1. Задача M10, власник: Сивопляс Максим.

Сценарії й пороги — ``docs/metrics.md``. Результат — ``results/c1_check.csv``.
"""

import csv
from pathlib import Path
from typing import Callable

import numpy as np

from lanes.config import project_path
from lanes.pipelines import REGISTRY, PipelineResult
from lanes.runner import PIPELINE_ERRORS
from lanes.synthetic import make_road

CHECK_FIELDS = ["pipeline", "scenario", "value", "threshold", "passed"]
BASELINE = "p0"
NORMAL_GROUP = "normal"


def _x_at_levels(lane: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """``x`` смуги на рівнях ``ys`` лінійною інтерполяцією по її точках.

    Рівень поза діапазоном ``y`` смуги дає ``nan``: смуга його не перетинає.
    """
    # np.interp вимагає зростання аргументу, тому впорядковуємо точки за y.
    order = np.argsort(lane[:, 1])
    return np.interp(ys, lane[order, 1], lane[order, 0], left=np.nan, right=np.nan)


def mean_lane_offset(gt_lanes: list[np.ndarray], pred_lanes: list[np.ndarray],
                     ys: np.ndarray) -> float:
    """Середнє горизонтальне відхилення знайдених смуг від еталонних, px.

    Для кожної еталонної смуги шукаємо найближчу знайдену за середнім
    ``|Δx|`` на рівнях ``ys``; ``x`` на рівні ``y`` — лінійна інтерполяція
    по точках смуги. Знайдених менше, ніж еталонних, або смуга не перетинає
    якийсь рівень — ``inf``.
    """
    if not gt_lanes:
        return 0.0
    if len(pred_lanes) < len(gt_lanes):
        return float("inf")
    ys = np.asarray(ys, dtype=np.float64)
    pred_xs = [_x_at_levels(lane, ys) for lane in pred_lanes]

    offsets = []
    for gt_lane in gt_lanes:
        gt_x = _x_at_levels(gt_lane, ys)
        # Середнє |Δx| до кожної знайденої смуги; nan (смуга не перетинає
        # рівень) робить усе середнє nan — замінюємо його на inf.
        distances = [float(np.mean(np.abs(gt_x - pred_x))) for pred_x in pred_xs]
        distances = [distance if np.isfinite(distance) else float("inf")
                     for distance in distances]
        offsets.append(min(distances))
    return float(np.mean(offsets))


def synthetic_scenarios(config: dict) -> dict[str, dict]:
    """Назва синтетичного сценарію → аргументи ``make_road`` (окрім розміру кадру)."""
    settings = config["checks"]
    return {
        "synthetic_straight": {},
        "synthetic_dark": {"brightness": settings["dark_brightness"]},
        "synthetic_noise": {"noise_sigma": settings["noise_sigma"], "seed": config["seed"]},
    }


def synthetic_offset(run: Callable[[np.ndarray, dict], PipelineResult], config: dict,
                     road_args: dict) -> float:
    """Відхилення смуг пайплайна ``run`` від еталонних на синтетичному кадрі, px.

    Кадр розміру ``checks.synthetic_size`` будує ``make_road(**road_args)``;
    відхилення міряємо на рівнях ``checks.offset_levels`` (частки висоти).
    Цією ж функцією користуються тести пайплайнів.
    """
    settings = config["checks"]
    width, height = settings["synthetic_size"]
    image, gt_lanes = make_road(width, height, **road_args)
    ys = np.array(settings["offset_levels"]) * (height - 1)
    return mean_lane_offset(gt_lanes, run(image, config).lanes, ys)


def _synthetic_row(name: str, scenario: str, road_args: dict, config: dict) -> dict:
    """Один пайплайн в одному синтетичному сценарії: відхилення проти порогу."""
    try:
        offset = synthetic_offset(REGISTRY[name].run, config, road_args)
    except PIPELINE_ERRORS as error:
        # Пайплайн упав або ще не написаний — кейс не пройдено.
        print(f"    {name} / {scenario}: помилка {type(error).__name__}: {error}")
        offset = float("inf")
    limit = config["checks"]["offset_max_px"]
    return {"pipeline": name, "scenario": scenario, "value": f"{offset:.2f}",
            "threshold": f"{limit}", "passed": bool(offset <= limit)}


def _read_group_metrics(path: Path, group: str) -> dict[str, dict[str, tuple[float, float]]]:
    """З ``metrics.csv``: пайплайн → фото → ``(psnr, ssim)`` для фото групи ``group``.

    Рядки без метрик (фото не прочиталося або пайплайн упав) пропускаємо.
    """
    table: dict[str, dict[str, tuple[float, float]]] = {}
    if not path.is_file():
        return table
    with open(path, encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["group"] == group and row["psnr"] and row["ssim"]:
                table.setdefault(row["pipeline"], {})[row["image"]] = (
                    float(row["psnr"]), float(row["ssim"]))
    return table


def _normal_group_row(name: str, table: dict, config: dict) -> dict:
    """Сценарій ``normal_group``: середні PSNR і SSIM проти порогів і проти p0."""
    settings = config["checks"]
    row = {"pipeline": name, "scenario": "normal_group", "value": "немає даних",
           "threshold": f"psnr>={settings['psnr_min']}; ssim>={settings['ssim_min']}; обидва > p0",
           "passed": False}
    # Порівнюємо на тих самих фото: беремо лише ті, де є метрики і пайплайна, і p0.
    images = sorted(set(table.get(name, {})) & set(table.get(BASELINE, {})))
    if not images:
        return row
    psnr, ssim = np.mean([table[name][image] for image in images], axis=0)
    base_psnr, base_ssim = np.mean([table[BASELINE][image] for image in images], axis=0)
    row["value"] = f"psnr={psnr:.2f}; ssim={ssim:.4f}"
    row["threshold"] += f" (psnr={base_psnr:.2f}; ssim={base_ssim:.4f})"
    row["passed"] = bool(psnr >= settings["psnr_min"] and ssim >= settings["ssim_min"]
                         and psnr > base_psnr and ssim > base_ssim)
    return row


def run_checks(config: dict) -> list[dict]:
    """Проганяє p1–p4 по чотирьох сценаріях і пише ``c1_check.csv``.

    Три синтетичні сценарії будує ``synthetic.make_road`` розміру
    ``checks.synthetic_size``; ``normal_group`` бере середні з
    ``results/metrics.csv`` і порівнює ще й з ``p0``. Повертає рядки таблиці
    ``{pipeline, scenario, value, threshold, passed}`` і друкує підсумок
    «пройдено k з 16».
    """
    output = project_path(config["output_dir"])
    table = _read_group_metrics(output / "metrics.csv", NORMAL_GROUP)
    if not table:
        print("Немає метрик групи normal: спершу main.py run; normal_group не пройдено.")

    rows = []
    for name in REGISTRY:
        if name == BASELINE:
            continue
        for scenario, road_args in synthetic_scenarios(config).items():
            rows.append(_synthetic_row(name, scenario, road_args, config))
        rows.append(_normal_group_row(name, table, config))

    output.mkdir(parents=True, exist_ok=True)
    with open(output / "c1_check.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CHECK_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(row["passed"] for row in rows)
    print(f"Пройдено {passed} з {len(rows)}: {output / 'c1_check.csv'}")
    return rows
