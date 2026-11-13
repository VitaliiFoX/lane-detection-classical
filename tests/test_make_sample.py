"""Тести tools/make_sample (F3) на штучному «CULane» у тимчасовій теці."""

import copy
import csv

import pytest

from tools.make_sample import make_sample


def _fake_culane(root):
    """Дві групи по 3 кадри; один кадр нічної групи без смуг."""
    lists = root / "list" / "test_split"
    lists.mkdir(parents=True)
    for group_file, clip in (("test0_normal.txt", "driver_a/clip1.MP4"),
                             ("test8_night.txt", "driver_b/clip2.MP4")):
        lines = []
        for frame in range(3):
            image = root / clip / f"0000{frame}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"jpeg")
            empty = clip.startswith("driver_b") and frame == 0
            image.with_suffix(".lines.txt").write_text("" if empty else "1 2 3 4\n",
                                                       encoding="utf-8")
            lines.append(f"/{clip}/0000{frame}.jpg")
        (lists / group_file).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _config(base, root):
    config = copy.deepcopy(base)
    config["data"].update(culane_root=str(root / "CULane"), sample_dir=str(root / "sample"),
                          sample_csv=str(root / "sample.csv"), per_group=2,
                          groups={"normal": "test0_normal.txt", "night": "test8_night.txt"})
    return config


def _rows(root):
    with open(root / "sample.csv", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_sample_is_copied_and_listed(config, tmp_path):
    _fake_culane(tmp_path / "CULane")
    config = _config(config, tmp_path)
    assert make_sample(config) == 4

    rows = _rows(tmp_path)
    assert [row["group"] for row in rows] == ["normal", "normal", "night", "night"]
    # Кадр без смуг (driver_b/…/00000) у вибірку не потрапляє.
    assert "driver_b/clip2.MP4/00000.jpg" not in {row["source"] for row in rows}
    for row in rows:
        assert (tmp_path / "sample" / row["image"]).is_file()
        assert "__" in row["image"]
        lines = (tmp_path / "sample" / row["image"]).with_suffix(".lines.txt")
        assert lines.read_text(encoding="utf-8") == "1 2 3 4\n"


def test_same_seed_same_sample(config, tmp_path):
    _fake_culane(tmp_path / "CULane")
    config = _config(config, tmp_path)
    make_sample(config)
    first = _rows(tmp_path)
    make_sample(config)
    assert _rows(tmp_path) == first


def test_missing_culane(config, tmp_path):
    with pytest.raises(FileNotFoundError):
        make_sample(_config(config, tmp_path))


def test_not_enough_frames(config, tmp_path):
    _fake_culane(tmp_path / "CULane")
    config = _config(config, tmp_path)
    config["data"]["per_group"] = 3
    with pytest.raises(ValueError):
        make_sample(config)
