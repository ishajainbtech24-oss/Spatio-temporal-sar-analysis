"""SAR preprocessing shared by training, validation, testing and inference.

The single entry point :func:`preprocess_chip` is used everywhere so the four stages can
never drift apart (the original notebooks/app used three different variants).

Steps, in order:

1. **Validity mask** - a pixel is valid if every selected band is finite. Sen1Floods11
   uses NaN for no data; the original notebook normalised with ``np.min``/``np.max``,
   which turns a whole chip into NaN if a single pixel is NaN (that is why 45 chips
   later looked "solid black" and were dropped).
2. **Speckle reduction** - NaN-aware median filter per band (no-data pixels are filled
   with the band median for filtering only, then masked again).
3. **Fixed-range scaling** - clip backscatter to [db_min, db_max] dB and scale to [0, 1].
   A *global* range keeps absolute brightness, which is the main physical cue for water
   (smooth water reflects radar away -> very low dB). Per-image min-max (used before)
   throws that cue away: a dry chip and a water chip both get stretched to 0..1.
4. **Fill** - invalid pixels get ``fill_value`` in the input and ``-1`` (ignore) in the label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

IGNORE_INDEX = -1


@dataclass
class Processed:
    image: np.ndarray      # (H, W, C) float32 in [0, 1]
    valid: np.ndarray      # (H, W) bool
    label: np.ndarray | None  # (H, W) int8 in {-1, 0, 1}


def validity_mask(image_chw: np.ndarray) -> np.ndarray:
    return np.isfinite(image_chw).all(axis=0)


def nan_median_filter(band: np.ndarray, size: int) -> np.ndarray:
    """Median filter that ignores NaNs (they are restored afterwards)."""
    if size is None or size <= 1:
        return band.astype(np.float32, copy=True)
    finite = np.isfinite(band)
    if not finite.any():
        return band.astype(np.float32, copy=True)
    filled = np.where(finite, band, np.nanmedian(band)).astype(np.float32)
    out = median_filter(filled, size=size, mode="reflect")
    out[~finite] = np.nan
    return out


def scale_db(image_chw: np.ndarray, db_min: float, db_max: float) -> np.ndarray:
    clipped = np.clip(image_chw, db_min, db_max)
    return ((clipped - db_min) / (db_max - db_min)).astype(np.float32)


def preprocess_chip(
    image_chw: np.ndarray,
    label: np.ndarray | None = None,
    median_size: int = 3,
    db_min: float = -50.0,
    db_max: float = 1.0,
    fill_value: float = 0.0,
) -> Processed:
    """Apply the full preprocessing; returns channels-last float32 + validity + label."""
    if image_chw.ndim != 3:
        raise ValueError(f"expected (bands, H, W) image, got shape {image_chw.shape}")
    if label is not None and label.shape != image_chw.shape[1:]:
        raise ValueError(f"label shape {label.shape} != image shape {image_chw.shape[1:]}")
    img = image_chw.astype(np.float32)
    img[~np.isfinite(img)] = np.nan
    valid = validity_mask(img)
    img = np.stack([nan_median_filter(b, median_size) for b in img])
    img = scale_db(img, db_min, db_max)
    img[:, ~valid] = fill_value
    img = np.nan_to_num(img, nan=fill_value)  # safety: never feed NaN to a network
    out_label = None
    if label is not None:
        out_label = label.astype(np.int8).copy()
        out_label[(out_label != 0) & (out_label != 1)] = IGNORE_INDEX
        out_label[~valid] = IGNORE_INDEX
    return Processed(image=np.transpose(img, (1, 2, 0)), valid=valid, label=out_label)


def preprocess_from_config(image_chw, label, pre_cfg) -> Processed:
    return preprocess_chip(image_chw, label, median_size=pre_cfg.median_size,
                           db_min=pre_cfg.db_min, db_max=pre_cfg.db_max,
                           fill_value=pre_cfg.fill_value)


def pad_to_multiple(image_hwc: np.ndarray, multiple: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Reflect-pad H and W up to a multiple of ``multiple``; returns padded image and (H, W)."""
    h, w = image_hwc.shape[:2]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        image_hwc = np.pad(image_hwc, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    return image_hwc, (h, w)


def random_crop_and_flip(image, label, crop: int, rng: np.random.Generator, augment: bool = True):
    """Joint random crop + flips/rot90 of an (H,W,C) image and its (H,W) label."""
    h, w = label.shape
    if h < crop or w < crop:
        image, _ = pad_to_multiple(image, crop)
        label = np.pad(label, ((0, image.shape[0] - h), (0, image.shape[1] - w)),
                       constant_values=IGNORE_INDEX)
        h, w = label.shape
    top, left = rng.integers(0, h - crop + 1), rng.integers(0, w - crop + 1)
    image = image[top:top + crop, left:left + crop]
    label = label[top:top + crop, left:left + crop]
    if augment:
        if rng.random() < 0.5:
            image, label = image[:, ::-1], label[:, ::-1]
        if rng.random() < 0.5:
            image, label = image[::-1], label[::-1]
        k = int(rng.integers(0, 4))
        image, label = np.rot90(image, k), np.rot90(label, k)
    return np.ascontiguousarray(image), np.ascontiguousarray(label)
