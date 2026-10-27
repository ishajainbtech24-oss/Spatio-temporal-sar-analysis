"""Classical, training-free water detectors used as baselines.

Open water is a specular reflector at C-band, so it shows up as very low VV backscatter.
Thresholding VV is the standard operational starting point for SAR flood mapping, which
makes it the right yardstick: a learned model has to beat it to be worth its complexity.
"""

from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu

from ..preprocessing.transforms import nan_median_filter


def _vv(image_chw: np.ndarray, median_size: int) -> np.ndarray:
    return nan_median_filter(image_chw[0], median_size)


def otsu_water(image_chw: np.ndarray, median_size: int = 3) -> tuple[np.ndarray, float]:
    """Per-chip Otsu threshold on VV dB. Returns (water mask, threshold in dB).

    Otsu assumes a bimodal histogram; on chips with little or no water it will still
    split the land pixels into two groups (false alarms) - a known weakness this
    baseline is meant to expose.
    """
    vv = _vv(image_chw, median_size)
    finite = np.isfinite(vv)
    if finite.sum() < 2 or np.nanmax(vv) == np.nanmin(vv):
        return np.zeros(vv.shape, dtype=bool), float("nan")
    th = float(threshold_otsu(vv[finite]))
    return (vv < th) & finite, th


def fixed_threshold_water(image_chw: np.ndarray, threshold_db: float = -15.0,
                          median_size: int = 3) -> np.ndarray:
    """Global VV threshold (default -15 dB, a common rule of thumb for open water)."""
    vv = _vv(image_chw, median_size)
    return np.isfinite(vv) & (vv < threshold_db)
