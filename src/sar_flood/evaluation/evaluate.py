"""Evaluate the U-Net and the baselines on one split and write JSON/CSV results."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..config import Config
from ..data.sen1floods11 import discover_samples, load_chip, make_splits
from ..utils import environment_info, get_logger, load_json, save_json, utc_now
from .baselines import fixed_threshold_water, otsu_water
from .metrics import MetricAccumulator

log = get_logger(__name__)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC AUC (ties averaged); NaN if only one class present."""
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = np.asarray(scores)[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _resolve_split(cfg: Config, split: str, run_dir: Path | None):
    samples = discover_samples(cfg.data_root)
    if run_dir is not None and (run_dir / "splits.json").exists():
        saved = load_json(run_dir / "splits.json")
        by_id = {s.chip_id: s for s in samples}
        missing = [c for c in saved[split] if c not in by_id]
        if missing:
            log.warning("%d chips listed in the run's splits.json are missing on disk", len(missing))
        return [by_id[c] for c in saved[split] if c in by_id], saved["strategy"]
    splits, strategy = make_splits(samples, cfg.data_root, cfg.data.split_strategy,
                                   cfg.data.val_fraction, cfg.data.test_fraction, cfg.train.seed)
    return splits[split], strategy


def evaluate(cfg: Config, split: str = "test", run_dir: str | Path | None = None,
             out_dir: str | Path | None = None, include_legacy: bool = False,
             data_source: str = "sen1floods11") -> Path:
    run_dir = Path(run_dir) if run_dir else None
    segmenter = None
    if run_dir is not None:
        from ..inference.segment import Segmenter  # noqa: PLC0415

        segmenter = Segmenter.from_run_dir(run_dir)
        cfg = segmenter.cfg  # evaluate with exactly the preprocessing the model was trained with
    samples, strategy = _resolve_split(cfg, split, run_dir)
    if not samples:
        raise RuntimeError(f"split '{split}' is empty")
    out_dir = Path(out_dir) if out_dir else cfg.resolve(cfg.output_dir) / "eval" / (
        (run_dir.name if run_dir else "baselines") + f"_{split}")

    accs = {"otsu_vv": MetricAccumulator(), "fixed_vv_-15db": MetricAccumulator()}
    if segmenter is not None:
        accs["unet"] = MetricAccumulator()
    legacy_rows = []
    legacy_model = None
    if include_legacy:
        from ..models.legacy import legacy_preprocess_vv, load_legacy_classifier  # noqa: PLC0415

        legacy_model = load_legacy_classifier()

    skipped = []
    for s in samples:
        try:
            chip = load_chip(s, cfg.data.bands)
        except Exception as exc:
            skipped.append({"chip_id": s.chip_id, "reason": str(exc)})
            continue
        label = chip.label.astype(np.int16).copy()
        label[~np.isfinite(chip.image).all(axis=0)] = -1  # no-data inputs are not scored
        if not (label >= 0).any():
            skipped.append({"chip_id": s.chip_id, "reason": "no valid labelled pixels"})
            continue
        ms = cfg.preprocess.median_size
        accs["otsu_vv"].add(s.chip_id, otsu_water(chip.image, ms)[0], label)
        accs["fixed_vv_-15db"].add(s.chip_id, fixed_threshold_water(chip.image, -15.0, ms), label)
        if segmenter is not None:
            accs["unet"].add(s.chip_id, segmenter.predict_chip(chip.image)["mask"], label)
        if legacy_model is not None:
            p = float(legacy_model.predict(legacy_preprocess_vv(chip.image[0]), verbose=0)[0, 0])
            legacy_rows.append({"chip_id": s.chip_id, "p_no_water": p,
                                "has_water": bool((label == 1).any())})

    results = {
        "created": utc_now(),
        "split": split,
        "split_strategy": strategy,
        "data_source": data_source,
        "n_chips": len(samples) - len(skipped),
        "skipped": skipped,
        "run": run_dir.name if run_dir else None,
        "unet_threshold": segmenter.threshold if segmenter else None,
        "unet_trained_on": segmenter.metadata.get("data_source") if segmenter else None,
        "models": {name: acc.summary() for name, acc in accs.items()},
        "environment": environment_info(),
        "notes": [
            "micro = one confusion matrix over all valid pixels; per_image excludes chips "
            "whose prediction and label are both empty.",
            "Pixels with no-data input or label -1 are ignored.",
        ],
    }
    if legacy_rows:
        y = np.array([not r["has_water"] for r in legacy_rows])  # positive = no water
        p = np.array([r["p_no_water"] for r in legacy_rows])
        from ..models.legacy import LEGACY_THRESHOLD_NO_WATER  # noqa: PLC0415

        pred = p >= LEGACY_THRESHOLD_NO_WATER
        tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
        results["legacy_scene_classifier"] = {
            "task": "chip-level: does the chip contain NO labelled water?",
            "WARNING": "The legacy model was trained on ALL hand-labelled chips (notebook 07 "
                       "final run), so these numbers are in-sample and optimistic.",
            "n": len(y), "n_no_water": int(y.sum()),
            "roc_auc": roc_auc(p, y),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "threshold": LEGACY_THRESHOLD_NO_WATER,
        }
        _write_csv(legacy_rows, out_dir / "legacy_per_chip.csv")

    for name, acc in accs.items():
        _write_csv(acc.per_image, out_dir / f"per_image_{name}.csv")
    save_json(results, out_dir / "metrics.json")
    for name, summ in results["models"].items():
        m = summ["micro"]
        log.info("%-15s IoU %.3f  Dice %.3f  P %.3f  R %.3f", name, m["iou"], m["dice_f1"],
                 m["precision"], m["recall"])
    return out_dir
