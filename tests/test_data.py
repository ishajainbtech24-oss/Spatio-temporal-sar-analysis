import numpy as np
import pytest

from sar_flood.data.sen1floods11 import (
    DatasetError,
    Sample,
    detect_data_source,
    discover_samples,
    group_split,
    load_chip,
    make_splits,
    validate_dataset,
)


def test_discover_pairs(synthetic_root):
    samples = discover_samples(synthetic_root)
    assert len(samples) == 18
    assert all(s.image_path.exists() and s.label_path.exists() for s in samples)
    assert {s.event for s in samples} == {"Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"}


def test_missing_dataset_has_helpful_error(tmp_path):
    with pytest.raises(DatasetError, match="DATASET.md"):
        discover_samples(tmp_path)


def test_unpaired_files_are_skipped(synthetic_root, tmp_path):
    import shutil

    root = tmp_path / "copy"
    shutil.copytree(synthetic_root, root)
    victim = next((root / "data/flood_events/HandLabeled/LabelHand").glob("*.tif"))
    victim.unlink()
    assert len(discover_samples(root)) == 17
    report = validate_dataset(root)
    assert len(report["unpaired"]["images_without_label"]) == 1


def test_load_chip_shapes_and_nodata(synthetic_root):
    s = next(s for s in discover_samples(synthetic_root) if s.chip_id.endswith("_1000"))
    chip = load_chip(s)
    assert chip.image.shape == (2, 64, 64) and chip.image.dtype == np.float32
    assert chip.label.shape == (64, 64)
    assert np.isnan(chip.image).any()          # chip 1000 has a no-data wedge
    assert set(np.unique(chip.label)) <= {-1, 0, 1}
    assert chip.profile["crs"] is not None


def test_official_split_used_and_disjoint(synthetic_root):
    splits, strategy = make_splits(discover_samples(synthetic_root), synthetic_root, "auto")
    assert strategy == "official"
    ids = [set(x.chip_id for x in v) for v in splits.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_group_split_keeps_events_together(synthetic_root):
    samples = discover_samples(synthetic_root)
    splits = group_split(samples, 0.15, 0.15, seed=3)
    events = {k: {s.event for s in v} for k, v in splits.items()}
    assert not (events["train"] & events["test"]) and not (events["train"] & events["val"])
    assert all(splits.values())
    assert sum(len(v) for v in splits.values()) == len(samples)


def test_group_split_needs_three_events(tmp_path):
    samples = [Sample(f"A_{i}", tmp_path, tmp_path) for i in range(4)]
    with pytest.raises(DatasetError):
        group_split(samples, 0.2, 0.2, 0)


def test_validate_dataset_report(synthetic_root):
    report = validate_dataset(synthetic_root)
    assert report["n_pairs"] == 18 and report["n_ok"] == 18
    assert report["n_with_any_nodata"] > 0


def test_validate_flags_bad_label(synthetic_root, tmp_path):
    import shutil

    import rasterio

    root = tmp_path / "bad"
    shutil.copytree(synthetic_root, root)
    lab = next((root / "data/flood_events/HandLabeled/LabelHand").glob("*.tif"))
    with rasterio.open(lab, "r+") as dst:
        arr = dst.read(1)
        arr[0, 0] = 7
        dst.write(arr, 1)
    report = validate_dataset(root)
    assert report["n_with_problems"] == 1


def test_detect_data_source(synthetic_root, tmp_path):
    assert detect_data_source(synthetic_root) == "synthetic"
    assert detect_data_source(tmp_path) == "sen1floods11"
