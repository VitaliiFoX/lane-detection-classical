"""Тести checks (M10)."""

import copy
import csv

import numpy as np

from lanes.checks import mean_lane_offset, run_checks
from lanes.pipelines import REGISTRY, PipelineInfo, PipelineResult
from lanes.runner import METRICS_FIELDS
from lanes.synthetic import make_road

# Дві вертикальні еталонні смуги x = 10 і x = 30 від y = 0 до y = 20.
GT = [np.array([[10.0, 0.0], [10.0, 20.0]]), np.array([[30.0, 0.0], [30.0, 20.0]])]
YS = np.array([5.0, 10.0, 15.0])


def test_offset_of_shifted_lanes_by_hand():
    # Знайдені смуги зсунуті: x = 12 (на 2 px) і x = 27 (на 3 px).
    # Найближча до x = 10 — x = 12 (|Δx| = 2, а не 17); до x = 30 — x = 27 (3, а не 18).
    # Середнє: (2 + 3) / 2 = 2.5.
    pred = [np.array([[27.0, 0.0], [27.0, 20.0]]), np.array([[12.0, 0.0], [12.0, 20.0]])]
    assert mean_lane_offset(GT, pred, YS) == 2.5


def test_offset_uses_linear_interpolation():
    # Знайдена смуга нахилена: від (10, 0) до (14, 20), тобто x = 10 + y / 5.
    # На рівнях y = 5, 10, 15: x = 11, 12, 13 → |Δx| = 1, 2, 3, середнє 2.
    gt = [GT[0]]
    pred = [np.array([[10.0, 0.0], [14.0, 20.0]])]
    np.testing.assert_allclose(mean_lane_offset(gt, pred, YS), 2.0)


def test_offset_does_not_depend_on_point_order():
    # Та сама смуга, точки записано знизу вгору.
    pred = [np.array([[14.0, 20.0], [10.0, 0.0]])]
    np.testing.assert_allclose(mean_lane_offset([GT[0]], pred, YS), 2.0)


def test_exact_match_is_zero():
    assert mean_lane_offset(GT, [lane.copy() for lane in GT], YS) == 0.0


def test_fewer_found_lanes_is_inf():
    assert mean_lane_offset(GT, [GT[0]], YS) == float("inf")
    assert mean_lane_offset(GT, [], YS) == float("inf")


def test_lane_not_reaching_level_is_inf():
    # Знайдена смуга закінчується на y = 12 і не перетинає рівень y = 15.
    short = [np.array([[10.0, 0.0], [10.0, 12.0]])]
    assert mean_lane_offset([GT[0]], short, YS) == float("inf")


def test_no_ground_truth_is_zero():
    assert mean_lane_offset([], [], YS) == 0.0


def _config(base, root):
    config = copy.deepcopy(base)
    config["output_dir"] = str(root / "results")
    return config


def _exact_run(image, config):
    """Пайплайн-заглушка: повертає еталонні смуги синтетичного кадру."""
    del config
    _, lanes = make_road(image.shape[1], image.shape[0])
    return PipelineResult(lanes=lanes)


def _empty_run(_image, _config):
    return PipelineResult()


def _broken_run(_image, _config):
    raise ValueError("зламано для тесту")


def _fake_registry(monkeypatch):
    """p1 знаходить смуги точно, p2 нічого не знаходить, p3 падає, p4 теж точний."""
    for name, run in (("p1", _exact_run), ("p2", _empty_run),
                      ("p3", _broken_run), ("p4", _exact_run)):
        monkeypatch.setitem(REGISTRY, name, PipelineInfo(name, name, "тест", run))


def _write_metrics(root, rows):
    path = root / "results" / "metrics.csv"
    path.parent.mkdir(parents=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRICS_FIELDS)
        writer.writeheader()
        for pipeline, group, image, psnr, ssim in rows:
            row = dict.fromkeys(METRICS_FIELDS, "")
            row.update(pipeline=pipeline, group=group, image=image, psnr=psnr, ssim=ssim)
            writer.writerow(row)


def test_run_checks_writes_16_cases(config, tmp_path, monkeypatch):
    _fake_registry(monkeypatch)
    rows = run_checks(_config(config, tmp_path))
    # 4 пайплайни (без p0) × 4 сценарії.
    assert len(rows) == 16
    assert {row["pipeline"] for row in rows} == {"p1", "p2", "p3", "p4"}
    assert {row["scenario"] for row in rows} == {
        "synthetic_straight", "synthetic_dark", "synthetic_noise", "normal_group"}

    with open(tmp_path / "results" / "c1_check.csv", encoding="utf-8", newline="") as file:
        saved = list(csv.DictReader(file))
    assert len(saved) == 16
    assert list(saved[0]) == ["pipeline", "scenario", "value", "threshold", "passed"]


def test_synthetic_scenarios_pass_and_fail(config, tmp_path, monkeypatch):
    _fake_registry(monkeypatch)
    rows = run_checks(_config(config, tmp_path))
    synthetic = {(row["pipeline"], row["scenario"]): row for row in rows
                 if row["scenario"] != "normal_group"}
    # Точні смуги: відхилення 0 ≤ 5 px у всіх трьох сценаріях.
    for scenario in ("synthetic_straight", "synthetic_dark", "synthetic_noise"):
        assert synthetic[("p1", scenario)]["value"] == "0.00"
        assert synthetic[("p1", scenario)]["passed"] is True
        # Жодної смуги і пайплайн, що впав, — відхилення inf, кейс не пройдено.
        assert synthetic[("p2", scenario)]["value"] == "inf"
        assert synthetic[("p2", scenario)]["passed"] is False
        assert synthetic[("p3", scenario)]["passed"] is False


def test_normal_group_without_metrics_fails(config, tmp_path, monkeypatch):
    _fake_registry(monkeypatch)
    rows = run_checks(_config(config, tmp_path))
    normal = [row for row in rows if row["scenario"] == "normal_group"]
    assert len(normal) == 4
    assert all(row["value"] == "немає даних" and row["passed"] is False for row in normal)


def test_normal_group_by_hand(config, tmp_path, monkeypatch):
    _fake_registry(monkeypatch)
    # Пороги з config.json: PSNR ≥ 15, SSIM ≥ 0.6, і обидва вищі, ніж у p0.
    # p0: PSNR (16 + 18) / 2 = 17, SSIM (0.70 + 0.80) / 2 = 0.75.
    # p1: PSNR (19 + 21) / 2 = 20 > 17, SSIM (0.80 + 0.90) / 2 = 0.85 > 0.75 → пройдено.
    # p2: PSNR (16 + 16) / 2 = 16 ≥ 15, але не вищий за p0 (17) → не пройдено.
    # p4: SSIM (0.50 + 0.60) / 2 = 0.55 < 0.6 → не пройдено.
    # Рядок групи night і рядок p1 без метрик у розрахунок не входять.
    _write_metrics(tmp_path, [
        ("p0", "normal", "a.jpg", "16", "0.70"), ("p0", "normal", "b.jpg", "18", "0.80"),
        ("p1", "normal", "a.jpg", "19", "0.80"), ("p1", "normal", "b.jpg", "21", "0.90"),
        ("p1", "normal", "c.jpg", "", ""), ("p1", "night", "d.jpg", "5", "0.10"),
        ("p2", "normal", "a.jpg", "16", "0.90"), ("p2", "normal", "b.jpg", "16", "0.90"),
        ("p4", "normal", "a.jpg", "30", "0.50"), ("p4", "normal", "b.jpg", "30", "0.60"),
    ])
    rows = run_checks(_config(config, tmp_path))
    normal = {row["pipeline"]: row for row in rows if row["scenario"] == "normal_group"}
    assert normal["p1"]["value"] == "psnr=20.00; ssim=0.8500"
    assert normal["p1"]["passed"] is True
    assert normal["p2"]["passed"] is False
    assert normal["p4"]["passed"] is False
    # Для p3 метрик немає взагалі.
    assert normal["p3"]["value"] == "немає даних"
