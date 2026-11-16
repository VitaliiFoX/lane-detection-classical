"""Зведені таблиці й графіки для звіту. Задача M11, власник: Сивопляс Максим."""

import csv
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from lanes.config import project_path
from lanes.io_utils import ImageLoadError, load_rgb

METRICS = ["mse", "psnr", "ssim", "iou", "time_ms"]
SUMMARY_FIELDS = ["pipeline", "group", "images", *METRICS]
# Позначка рядка, усередненого по всіх групах.
ALL_GROUPS = "all"

# Графіки: файл → (стовпець summary.csv, підпис осі y).
CHARTS = {
    "iou_by_group.png": ("iou", "IoU"),
    "psnr_by_group.png": ("psnr", "PSNR, дБ"),
    "ssim_by_group.png": ("ssim", "SSIM"),
    "time_by_group.png": ("time_ms", "Час на фото, мс"),
}

# Оформлення графіків — лише візуалізація. Колір закріплено за пайплайном, а
# не за його місцем на графіку; базова лінія p0 — нейтральна сіра.
PIPELINE_COLORS = {"p0": "#8a8984", "p1": "#2a78d6", "p2": "#eb6834",
                   "p3": "#1baf7a", "p4": "#eda100"}
DEFAULT_COLOR = "#6250d6"
GRID_COLOR = "#d9d8d2"
BARS_WIDTH = 0.8     # частка місця групи, яку займають усі стовпчики разом
BAR_GAP = 0.9        # частка ширини стовпчика, яку зафарбовуємо (решта — проміжок)
CELL_INCHES = 2.6    # розмір однієї картинки в сітці прикладів
FIGURE_DPI = 150


def read_metrics(path: Path) -> list[dict]:
    """Читає ``metrics.csv``; рядки без метрик (збій на фото) пропускає."""
    if not path.is_file():
        raise FileNotFoundError(f"Немає {path}: спершу запустіть main.py run")
    with open(path, encoding="utf-8", newline="") as file:
        return [row for row in csv.DictReader(file) if all(row[key] for key in METRICS)]


def _unique(values: list[str]) -> list[str]:
    """Різні значення в порядку першої появи."""
    return list(dict.fromkeys(values))


def summarize(rows: list[dict]) -> list[dict]:
    """Середні метрики для кожної пари (пайплайн, група) і рядок «усі групи».

    Середнє арифметичне по фото; ``images`` — скільки фото усереднено.
    Числа лишаємо дійсними — форматує їх ``build_report`` під час запису.
    """
    summary = []
    for pipeline in sorted(_unique([row["pipeline"] for row in rows])):
        own = [row for row in rows if row["pipeline"] == pipeline]
        groups = _unique([row["group"] for row in own])
        for group in [*groups, ALL_GROUPS]:
            chosen = [row for row in own if group in (row["group"], ALL_GROUPS)]
            line = {"pipeline": pipeline, "group": group, "images": len(chosen)}
            for metric in METRICS:
                line[metric] = float(np.mean([float(row[metric]) for row in chosen]))
            summary.append(line)
    return summary


def _write_summary(path: Path, summary: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for line in summary:
            writer.writerow({key: f"{value:.4f}" if key in METRICS else value
                             for key, value in line.items()})


def _bar_chart(summary: list[dict], metric: str, label: str, path: Path) -> None:
    """Стовпчикова діаграма: групи фото по осі x, стовпчик на кожен пайплайн."""
    pipelines = _unique([line["pipeline"] for line in summary])
    groups = _unique([line["group"] for line in summary])
    values = {(line["pipeline"], line["group"]): line[metric] for line in summary}

    figure = Figure(figsize=(9, 4.5))
    axes = figure.add_subplot(1, 1, 1)
    # Кожна група займає на осі x відрізок довжини 1 з центром у своєму
    # номері; стовпчики пайплайнів ділять між собою частку BARS_WIDTH.
    bar_width = BARS_WIDTH / len(pipelines)
    positions = np.arange(len(groups))
    for index, pipeline in enumerate(pipelines):
        heights = [values.get((pipeline, group), 0.0) for group in groups]
        # Зсув стовпчика від центру групи: перший — найлівіший.
        shift = (index - (len(pipelines) - 1) / 2) * bar_width
        axes.bar(positions + shift, heights, bar_width * BAR_GAP, label=pipeline,
                 color=PIPELINE_COLORS.get(pipeline, DEFAULT_COLOR))

    axes.set_xticks(positions)
    axes.set_xticklabels(groups)
    axes.set_xlabel("Група фото")
    axes.set_ylabel(label)
    axes.set_title(f"{label} за групами фото")
    axes.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.legend(title="Пайплайн", frameon=False, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)


def _example_images(pipeline: str, image: str, group: str, sample_dir: Path,
                    output: Path) -> list[tuple[str, np.ndarray]]:
    """Картинки одного прикладу: фото, стадії пайплайна по порядку, результат.

    Стадії лежать там, куди їх зберіг раннер:
    ``results/<пайплайн>/<група>/<фото>/NN_<стадія>.png`` і ``overlay.png``.
    Файл, якого немає чи який не читається, пропускаємо.
    """
    folder = output / pipeline / group / Path(image).stem
    files = [("фото", sample_dir / image)]
    files += [(path.stem, path) for path in sorted(folder.glob("[0-9][0-9]_*.png"))]
    files.append(("результат", folder / "overlay.png"))
    pictures = []
    for title, path in files:
        try:
            pictures.append((title, load_rgb(path)))
        except ImageLoadError:
            continue
    return pictures


def _examples_grid(pipeline: str, rows: list[dict], sample_dir: Path, output: Path,
                   path: Path) -> bool:
    """Сітка «фото → стадії → результат»: рядок на групу, перше фото групи.

    Повертає ``False`` і нічого не зберігає, якщо картинок немає.
    """
    own = [row for row in rows if row["pipeline"] == pipeline]
    examples = []
    for group in _unique([row["group"] for row in own]):
        image = next(row["image"] for row in own if row["group"] == group)
        pictures = _example_images(pipeline, image, group, sample_dir, output)
        if pictures:
            examples.append((group, pictures))
    if not examples:
        return False

    n_columns = max(len(pictures) for _, pictures in examples)
    figure = Figure(figsize=(CELL_INCHES * n_columns, CELL_INCHES * 0.6 * len(examples) + 0.6))
    for row_index, (group, pictures) in enumerate(examples):
        for column, (title, picture) in enumerate(pictures):
            # Клітинки сітки нумеруються з 1 зліва направо, рядок за рядком.
            axes = figure.add_subplot(len(examples), n_columns,
                                      row_index * n_columns + column + 1)
            axes.imshow(picture)
            axes.set_title(f"{group}: {title}", fontsize=8)
            axes.axis("off")
    figure.suptitle(f"{pipeline}: фото → стадії → результат", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    return True


def build_report(config: dict) -> None:
    """Читає ``results/metrics.csv`` і будує матеріали для звіту.

    - ``results/summary.csv`` — середні MSE, PSNR, SSIM, IoU і час для кожної
      пари (пайплайн, група) і рядок «усі групи»;
    - ``results/figures/`` — стовпчикові діаграми IoU, PSNR, SSIM і часу за
      групами (Matplotlib) і сітка прикладів «фото → стадії → результат».
    """
    output = project_path(config["output_dir"])
    rows = read_metrics(output / "metrics.csv")
    summary = summarize(rows)
    _write_summary(output / "summary.csv", summary)
    print(f"Зведена таблиця: {output / 'summary.csv'}")
    if not summary:
        print("У metrics.csv немає жодного рядка з метриками — графіки не будуємо.")
        return

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for file_name, (metric, label) in CHARTS.items():
        _bar_chart(summary, metric, label, figures / file_name)

    sample_dir = project_path(config["data"]["sample_dir"])
    for pipeline in _unique([line["pipeline"] for line in summary]):
        _examples_grid(pipeline, rows, sample_dir, output, figures / f"examples_{pipeline}.png")
    print(f"Графіки: {figures}")
