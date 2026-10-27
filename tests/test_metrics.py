import math

import numpy as np
import pytest

from sar_flood.evaluation.baselines import fixed_threshold_water, otsu_water
from sar_flood.evaluation.evaluate import roc_auc
from sar_flood.evaluation.metrics import Confusion, MetricAccumulator, best_threshold


def test_confusion_known_values():
    pred = np.array([[1, 1, 0, 0]], bool)
    target = np.array([[1, 0, 1, 0]])
    m = Confusion().update(pred, target).metrics()
    assert m["iou"] == pytest.approx(1 / 3)
    assert m["dice_f1"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert m["pixel_accuracy"] == pytest.approx(0.5)


def test_ignore_index_excluded():
    pred = np.array([1, 1, 1], bool)
    target = np.array([1, -1, -1])
    m = Confusion().update(pred, target).metrics()
    assert m["iou"] == 1.0 and m["valid_pixels"] == 1


def test_empty_masks_are_undefined_not_zero():
    m = Confusion().update(np.zeros(4, bool), np.zeros(4)).metrics()
    assert math.isnan(m["iou"]) and m["pixel_accuracy"] == 1.0


def test_accumulator_micro_vs_per_image():
    acc = MetricAccumulator()
    acc.add("a", np.array([1, 0], bool), np.array([1, 0]))      # IoU 1
    acc.add("b", np.array([0, 0], bool), np.array([0, 0]))      # empty union -> excluded
    acc.add("c", np.array([1, 1], bool), np.array([1, 0]))      # IoU 0.5
    s = acc.summary()
    assert s["per_image"]["n_scored"] == 2 and s["per_image"]["n_empty_union_excluded"] == 1
    assert s["per_image"]["mean_iou"] == pytest.approx(0.75)
    assert s["micro"]["iou"] == pytest.approx(2 / 3)


def test_shape_mismatch():
    with pytest.raises(ValueError):
        Confusion().update(np.zeros(3, bool), np.zeros(4))


def test_best_threshold_finds_separating_value():
    probs = np.array([0.1, 0.2, 0.35, 0.8, 0.9])
    target = np.array([0, 0, 1, 1, -1])
    th, iou = best_threshold(probs, target)
    assert 0.2 < th <= 0.35 and iou == 1.0


def test_roc_auc():
    assert roc_auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == pytest.approx(0.75)
    assert roc_auc(np.array([0.5, 0.5]), np.array([0, 1])) == pytest.approx(0.5)
    assert math.isnan(roc_auc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_threshold_baselines_find_dark_water(rng):
    img = rng.normal(-8, 1, size=(2, 32, 32)).astype(np.float32)
    img[:, :, :10] = rng.normal(-22, 1, size=(2, 32, 10))
    img[:, 0, 31] = np.nan
    mask, th = otsu_water(img)
    assert mask[:, :10].mean() > 0.95 and mask[:, 12:].mean() < 0.05
    assert -22 < th < -8
    fixed = fixed_threshold_water(img)
    assert fixed[:, :10].all() and not fixed[0, 31]


def test_otsu_on_constant_image():
    mask, th = otsu_water(np.full((2, 8, 8), -10.0, np.float32))
    assert not mask.any() and math.isnan(th)
