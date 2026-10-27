"""Binary segmentation metrics with an ignore index.

Two aggregations are reported, because they answer different questions:

* **micro** (a.k.a. dataset-level): one confusion matrix over *all* valid pixels of
  all chips. This is what Sen1Floods11 papers usually report as IoU.
* **per-image mean**: IoU averaged over chips. Chips where both prediction and label
  contain no water have an undefined IoU (0/0); they are excluded from the mean and
  counted separately instead of being silently scored as 0 or 1.

Pixel accuracy is reported but is misleading for this task: with ~10-20% water, a model
that predicts "no water" everywhere still scores 80-90%.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IGNORE_INDEX = -1


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, pred: np.ndarray, target: np.ndarray, ignore_index: int = IGNORE_INDEX) -> Confusion:
        pred = np.asarray(pred).astype(bool)
        target = np.asarray(target)
        if pred.shape != target.shape:
            raise ValueError(f"prediction {pred.shape} and target {target.shape} differ")
        valid = target != ignore_index
        p, t = pred[valid], target[valid].astype(bool)
        self.tp += int(np.sum(p & t))
        self.fp += int(np.sum(p & ~t))
        self.fn += int(np.sum(~p & t))
        self.tn += int(np.sum(~p & ~t))
        return self

    def __add__(self, other: Confusion) -> Confusion:
        return Confusion(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn)

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    def metrics(self) -> dict[str, float]:
        def div(a: float, b: float) -> float:
            return float(a / b) if b > 0 else float("nan")

        return {
            "iou": div(self.tp, self.tp + self.fp + self.fn),
            "dice_f1": div(2 * self.tp, 2 * self.tp + self.fp + self.fn),
            "precision": div(self.tp, self.tp + self.fp),
            "recall": div(self.tp, self.tp + self.fn),
            "pixel_accuracy": div(self.tp + self.tn, self.n),
            "water_fraction_true": div(self.tp + self.fn, self.n),
            "water_fraction_pred": div(self.tp + self.fp, self.n),
            "valid_pixels": self.n,
        }


@dataclass
class MetricAccumulator:
    total: Confusion = field(default_factory=Confusion)
    per_image: list[dict] = field(default_factory=list)

    def add(self, chip_id: str, pred: np.ndarray, target: np.ndarray) -> dict:
        c = Confusion().update(pred, target)
        self.total = self.total + c
        row = {"chip_id": chip_id, **c.metrics(), "tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn}
        row["empty_union"] = (c.tp + c.fp + c.fn) == 0
        self.per_image.append(row)
        return row

    def summary(self) -> dict:
        ious = [r["iou"] for r in self.per_image if not r["empty_union"]]
        dices = [r["dice_f1"] for r in self.per_image if not r["empty_union"]]
        return {
            "micro": self.total.metrics(),
            "per_image": {
                "n_images": len(self.per_image),
                "n_scored": len(ious),
                "n_empty_union_excluded": len(self.per_image) - len(ious),
                "mean_iou": float(np.mean(ious)) if ious else float("nan"),
                "median_iou": float(np.median(ious)) if ious else float("nan"),
                "mean_dice_f1": float(np.mean(dices)) if dices else float("nan"),
            },
            "confusion": {"tp": self.total.tp, "fp": self.total.fp,
                          "fn": self.total.fn, "tn": self.total.tn},
        }


def best_threshold(probs: np.ndarray, target: np.ndarray, grid=None) -> tuple[float, float]:
    """Threshold maximising micro IoU on (flattened, ignore-aware) validation pixels."""
    grid = np.arange(0.1, 0.91, 0.05) if grid is None else grid
    valid = target != IGNORE_INDEX
    p, t = probs[valid], target[valid].astype(bool)
    best = (0.5, -1.0)
    for th in grid:
        pred = p >= th
        tp, fp, fn = np.sum(pred & t), np.sum(pred & ~t), np.sum(~pred & t)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        if iou > best[1]:
            best = (float(th), float(iou))
    return best
