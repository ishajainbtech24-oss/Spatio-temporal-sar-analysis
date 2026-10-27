"""Load and preprocess a split into memory (Sen1Floods11 hand-labelled set is ~1 GB)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..preprocessing.transforms import preprocess_from_config
from ..utils import get_logger
from .sen1floods11 import Sample, load_chip

log = get_logger(__name__)


@dataclass
class SplitArrays:
    ids: list[str]
    images: list[np.ndarray]       # each (H, W, C) float32 in [0, 1]
    labels: list[np.ndarray]       # each (H, W) int8 in {-1, 0, 1}
    valids: list[np.ndarray]       # each (H, W) bool, True where the SAR input is valid
    raw: list[np.ndarray] | None   # optional raw dB (C, H, W) for baselines
    skipped: list[dict]


def load_split(samples: Sequence[Sample], cfg, keep_raw: bool = False,
               max_samples: int | None = None) -> SplitArrays:
    ids, images, labels, valids, raws, skipped = [], [], [], [], [], []
    for s in list(samples)[: max_samples or None]:
        try:
            chip = load_chip(s, cfg.data.bands)
            proc = preprocess_from_config(chip.image, chip.label, cfg.preprocess)
        except Exception as exc:  # corrupted / unreadable chip: skip, but record why
            skipped.append({"chip_id": s.chip_id, "reason": str(exc)})
            log.warning("skipping %s: %s", s.chip_id, exc)
            continue
        if not (proc.label >= 0).any():
            skipped.append({"chip_id": s.chip_id, "reason": "no valid labelled pixels"})
            continue
        ids.append(s.chip_id)
        images.append(proc.image)
        labels.append(proc.label)
        valids.append(proc.valid)
        if keep_raw:
            raws.append(chip.image)
    if not ids:
        raise RuntimeError("No usable samples in this split (all were skipped)")
    return SplitArrays(ids, images, labels, valids, raws if keep_raw else None, skipped)
