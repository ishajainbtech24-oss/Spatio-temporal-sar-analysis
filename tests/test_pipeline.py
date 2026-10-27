"""End-to-end: train -> evaluate -> predict on a tiny synthetic dataset (CPU, ~1 min)."""

import json

import numpy as np
import pytest
from conftest import requires_tf


@pytest.fixture(scope="module")
def trained_run(synthetic_root, tmp_path_factory):
    from sar_flood.config import load_config
    from sar_flood.training.trainer import train

    out = tmp_path_factory.mktemp("out")
    cfg = load_config("configs/smoke.yaml", {
        "data": {"root": str(synthetic_root)}, "output_dir": str(out), "run_name": "t",
        "model": {"base_filters": 4, "depth": 2},
        "train": {"epochs": 2, "ae_epochs": 1, "crop_size": 32, "batch_size": 2},
    })
    return train(cfg, data_source="synthetic"), cfg


@requires_tf
def test_training_writes_reproducible_artifacts(trained_run):
    run_dir, _ = trained_run
    for name in ["model.keras", "metadata.json", "config.yaml", "splits.json", "history.csv",
                 "autoencoder.keras"]:
        assert (run_dir / name).exists(), name
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["data_source"] == "synthetic" and meta["encoder_init"] == "autoencoder"
    assert 0 < meta["threshold"] < 1
    splits = json.loads((run_dir / "splits.json").read_text())
    assert not set(splits["train"]) & set(splits["test"])


@requires_tf
def test_evaluate_writes_json_and_csv(trained_run, tmp_path):
    from sar_flood.evaluation.evaluate import evaluate

    run_dir, cfg = trained_run
    out = evaluate(cfg, split="test", run_dir=run_dir, out_dir=tmp_path / "ev", data_source="synthetic")
    metrics = json.loads((out / "metrics.json").read_text())
    assert set(metrics["models"]) == {"unet", "otsu_vv", "fixed_vv_-15db"}
    assert metrics["n_chips"] == 3
    assert (out / "per_image_unet.csv").exists()
    # the test split matches the one recorded at training time
    assert metrics["split_strategy"] == "official"


def test_evaluate_baselines_without_tensorflow_model(synthetic_root, tmp_path):
    from sar_flood.config import load_config
    from sar_flood.evaluation.evaluate import evaluate

    cfg = load_config(None, {"data": {"root": str(synthetic_root)}})
    out = evaluate(cfg, split="test", out_dir=tmp_path / "b")
    m = json.loads((out / "metrics.json").read_text())["models"]["otsu_vv"]["micro"]
    assert 0.5 < m["iou"] <= 1.0  # synthetic water is very separable


@requires_tf
def test_segmenter_inference_shape(trained_run, synthetic_root):
    from sar_flood.data.sen1floods11 import discover_samples, load_chip
    from sar_flood.inference.segment import Segmenter

    run_dir, _ = trained_run
    seg = Segmenter.from_run_dir(run_dir)
    chip = load_chip(discover_samples(synthetic_root)[0])
    out = seg.predict_chip(chip.image[:, :60, :50])  # non-multiple size
    assert out["prob"].shape == out["mask"].shape == (60, 50)
    assert not out["mask"][~out["valid"]].any()


def test_missing_checkpoint_message(tmp_path):
    pytest.importorskip("tensorflow")
    from sar_flood.inference.segment import Segmenter

    with pytest.raises(FileNotFoundError, match="sar-flood train"):
        Segmenter.from_run_dir(tmp_path)


def test_analyze_geotiff_with_label(synthetic_root, tmp_path):
    from sar_flood.data.sen1floods11 import discover_samples
    from sar_flood.inference.pipeline import analyze_file

    s = discover_samples(synthetic_root)[0]
    res = analyze_file(s.image_path, tmp_path, label_path=s.label_path, use_legacy=False)
    assert {"otsu", "fixed"} <= set(res["methods"])
    assert res["calibrated_db"] and res["map"] is not None
    assert "metrics" in res["methods"]["otsu"]
    for f in [res["images"]["vv"], res["methods"]["otsu"]["overlay"], res["methods"]["otsu"]["geotiff"]]:
        assert (tmp_path / f).exists()


def test_analyze_png_is_flagged_uncalibrated(tmp_path):
    import cv2

    from sar_flood.inference.pipeline import analyze_file

    img = np.full((64, 64), 180, np.uint8)
    img[:, :20] = 20
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), img)
    res = analyze_file(p, tmp_path / "o", use_legacy=False)
    assert not res["calibrated_db"] and list(res["methods"]) == ["otsu"]
    assert res["warnings"]


@pytest.mark.parametrize("content,suffix,msg", [
    (b"not an image", ".png", "decoded"),
    (b"not a tiff", ".tif", "GeoTIFF"),
    (b"x", ".exe", "Unsupported"),
])
def test_analyze_rejects_bad_files(tmp_path, content, suffix, msg):
    from sar_flood.inference.pipeline import InputError, analyze_file

    p = tmp_path / f"bad{suffix}"
    p.write_bytes(content)
    with pytest.raises(InputError, match=msg):
        analyze_file(p, tmp_path / "o", use_legacy=False)


def test_label_shape_mismatch_rejected(synthetic_root, tmp_path):
    from sar_flood.data.sen1floods11 import discover_samples
    from sar_flood.data.synthetic import write_dataset
    from sar_flood.inference.pipeline import InputError, analyze_file

    other = discover_samples(write_dataset(tmp_path / "big", n_per_event=1, size=96))[0]
    s = discover_samples(synthetic_root)[0]
    with pytest.raises(InputError, match="must match"):
        analyze_file(s.image_path, tmp_path / "o", label_path=other.label_path, use_legacy=False)
