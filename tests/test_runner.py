"""Тести runner (F11) на маленькій штучній вибірці у тимчасовій теці."""

import csv
import copy

import numpy as np

from lanes.io_utils import save_png
from lanes.pipelines import REGISTRY, PipelineInfo
from lanes.runner import run_pipelines


def _make_sample(root):
    """Два фото з однією вертикальною смугою і один битий файл."""
    sample_dir = root / "sample"
    rows = []
    for index in range(2):
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        image[:, 30] = 255
        name = f"normal/frame{index}.jpg"
        save_png(sample_dir / name, image)
        (sample_dir / f"normal/frame{index}.lines.txt").write_text("30 39 30 0\n", encoding="utf-8")
        rows.append({"group": "normal", "image": name, "source": "-"})
    (sample_dir / "night").mkdir()
    (sample_dir / "night/broken.jpg").write_bytes(b"not an image")
    rows.append({"group": "night", "image": "night/broken.jpg", "source": "-"})

    with open(root / "sample.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["group", "image", "source"])
        writer.writeheader()
        writer.writerows(rows)
    return sample_dir


def _config(base, root):
    config = copy.deepcopy(base)
    config["data"]["sample_dir"] = str(_make_sample(root))
    config["data"]["sample_csv"] = str(root / "sample.csv")
    config["output_dir"] = str(root / "results")
    return config


def _read(path):
    with open(path, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_baseline_run_writes_metrics_and_images(config, tmp_path):
    config = _config(config, tmp_path)
    metrics_path = run_pipelines(config, ["p0"])
    rows = _read(metrics_path)
    assert len(rows) == 3

    good = rows[0]
    # Порожня маска проти смуги: перетину немає → IoU = 0.
    assert (good["pipeline"], good["iou"], good["lanes_found"], good["lanes_gt"]) == \
        ("p0", "0.0000", "0", "1")
    assert float(good["psnr"]) > 0
    assert (tmp_path / "results/p0/normal/frame0/mask.png").is_file()
    assert (tmp_path / "results/p0/normal/frame0/overlay.png").is_file()

    broken = rows[2]
    assert broken["group"] == "night"
    assert broken["mse"] == ""


def test_failed_pipeline_does_not_stop_run(config, tmp_path, monkeypatch):
    def broken_run(_image, _config):
        raise ValueError("зламано для тесту")

    monkeypatch.setitem(REGISTRY, "p1", PipelineInfo("p1", "Зламаний", "тест", broken_run))
    config = _config(config, tmp_path)
    rows = _read(run_pipelines(config, ["p0", "p1"]))
    assert len(rows) == 6
    assert all(row["mse"] == "" for row in rows if row["pipeline"] == "p1")


def test_rerun_keeps_other_pipelines_and_limit(config, tmp_path):
    config = _config(config, tmp_path)
    run_pipelines(config, ["p0", "p1"])
    rows = _read(run_pipelines(config, ["p0"], limit=1))
    # p1 з першого запуску (3 рядки) лишився, p0 перезапущено лише на 1 фото.
    assert sum(row["pipeline"] == "p1" for row in rows) == 3
    assert sum(row["pipeline"] == "p0" for row in rows) == 1
