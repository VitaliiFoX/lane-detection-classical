"""Тести report (M11) на штучних метриках у тимчасовій теці."""

import copy
import csv

import numpy as np
import pytest

from lanes.io_utils import save_png
from lanes.report import build_report, read_metrics, summarize
from lanes.runner import METRICS_FIELDS

# pipeline, group, image, mse, psnr, ssim, iou, time_ms
ROWS = [
    ("p0", "normal", "normal/a.jpg", "100", "20", "0.50", "0.0", "1"),
    ("p0", "normal", "normal/b.jpg", "300", "10", "0.70", "0.0", "3"),
    ("p0", "night", "night/c.jpg", "500", "6", "0.30", "0.0", "2"),
    ("p3", "normal", "normal/a.jpg", "50", "30", "0.90", "0.6", "40"),
    ("p3", "normal", "normal/b.jpg", "", "", "", "", ""),
]


def _config(base, root):
    config = copy.deepcopy(base)
    config["output_dir"] = str(root / "results")
    config["data"]["sample_dir"] = str(root / "sample")
    return config


def _write_metrics(root, rows=None):
    path = root / "results" / "metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["pipeline", "group", "image", "mse", "psnr", "ssim", "iou",
                         "time_ms"])
        writer.writerows(ROWS if rows is None else rows)
    return path


def test_metrics_columns_match_runner():
    # Звіт читає ті самі стовпці, які пише раннер.
    assert {"pipeline", "group", "image", "mse", "psnr", "ssim", "iou",
            "time_ms"} <= set(METRICS_FIELDS)


def test_summary_by_hand(tmp_path):
    summary = summarize(read_metrics(_write_metrics(tmp_path)))
    table = {(line["pipeline"], line["group"]): line for line in summary}
    # p0 / normal: два фото. MSE (100 + 300) / 2 = 200, PSNR (20 + 10) / 2 = 15,
    # SSIM (0.5 + 0.7) / 2 = 0.6, час (1 + 3) / 2 = 2.
    line = table[("p0", "normal")]
    assert line["images"] == 2
    assert line["mse"] == pytest.approx(200)
    assert line["psnr"] == pytest.approx(15)
    assert line["ssim"] == pytest.approx(0.6)
    assert line["time_ms"] == pytest.approx(2)
    # p0 / усі групи: три фото. MSE (100 + 300 + 500) / 3 = 300, PSNR (20 + 10 + 6) / 3 = 12.
    line = table[("p0", "all")]
    assert line["images"] == 3
    assert line["mse"] == pytest.approx(300)
    assert line["psnr"] == pytest.approx(12)
    # p3: рядок без метрик (пайплайн упав на фото b) у середнє не входить.
    assert table[("p3", "normal")]["images"] == 1
    assert table[("p3", "normal")]["iou"] == pytest.approx(0.6)
    assert table[("p3", "all")]["iou"] == pytest.approx(0.6)
    # Групи night для p3 немає — немає й рядка.
    assert ("p3", "night") not in table


def test_build_report_writes_summary_and_charts(config, tmp_path):
    _write_metrics(tmp_path)
    build_report(_config(config, tmp_path))

    with open(tmp_path / "results" / "summary.csv", encoding="utf-8", newline="") as file:
        saved = list(csv.DictReader(file))
    # p0: normal, night, all; p3: normal, all.
    assert [(row["pipeline"], row["group"]) for row in saved] == [
        ("p0", "normal"), ("p0", "night"), ("p0", "all"), ("p3", "normal"), ("p3", "all")]
    assert saved[0]["mse"] == "200.0000"
    assert saved[0]["images"] == "2"

    figures = tmp_path / "results" / "figures"
    for name in ("iou_by_group.png", "psnr_by_group.png", "ssim_by_group.png",
                 "time_by_group.png"):
        assert (figures / name).stat().st_size > 0
    # Фото й стадій у теці немає — сітку прикладів не будуємо.
    assert not (figures / "examples_p3.png").exists()


def test_examples_grid_from_saved_stages(config, tmp_path):
    _write_metrics(tmp_path)
    photo = np.full((20, 30, 3), 90, dtype=np.uint8)
    save_png(tmp_path / "sample" / "normal" / "a.jpg", photo)
    folder = tmp_path / "results" / "p3" / "normal" / "a"
    save_png(folder / "01_top_view.png", np.zeros((30, 20)))
    save_png(folder / "02_edges.png", np.zeros((30, 20), dtype=np.bool_))
    save_png(folder / "overlay.png", photo)

    build_report(_config(config, tmp_path))
    assert (tmp_path / "results" / "figures" / "examples_p3.png").stat().st_size > 0


def test_metrics_without_values_give_empty_summary(config, tmp_path):
    _write_metrics(tmp_path, [("p3", "normal", "normal/b.jpg", "", "", "", "", "")])
    build_report(_config(config, tmp_path))
    with open(tmp_path / "results" / "summary.csv", encoding="utf-8", newline="") as file:
        assert not list(csv.DictReader(file))
    assert not (tmp_path / "results" / "figures").exists()


def test_missing_metrics_file(config, tmp_path):
    with pytest.raises(FileNotFoundError):
        build_report(_config(config, tmp_path))
