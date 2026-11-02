"""Scene-level flood verdict and rule-based land-surface screening from Sentinel-1.

Flood verdict
-------------
Uses the project's own definition from ``notebooks/02_preprocessing.ipynb``: a chip was
labelled "Flood" when its mask contained water pixels. Here the mask is the selected
method's prediction, and a minimum share of :data:`FLOOD_MIN_WATER_FRACTION` of valid
pixels is required so that a few speckle pixels cannot flip the verdict. Because the
Sen1Floods11 labels include permanent water, "flood-affected" means "water-covered at
acquisition time"; telling flood water from rivers needs a pre-event image.

Land-surface screening (NOT a trained model)
--------------------------------------------
Neither the notebooks nor any saved model classify land cover or plantation suitability,
and Sen1Floods11 has no land-cover labels to train or test one. This module therefore uses
transparent, physically motivated C-band backscatter rules on non-water pixels:

* **Built-up / very bright** - VV >= :data:`BUILT_VV_MIN_DB`. Walls and ground form corner
  reflectors (double bounce) that return far more energy than natural surfaces.
* **Vegetation** - VH >= :data:`VEG_VH_MIN_DB`. Cross-polarised return mainly comes from
  volume scattering inside canopies; bare ground depolarises little.
* **Bare / sparse** - everything else that is not water.

The thresholds are common rules of thumb, not calibrated on labelled data, and they
shift with incidence angle, soil moisture, crop stage and terrain. The plantation
screening built on them is a first-pass triage that tells you where a field survey might
be worthwhile; it does not assess soil, slope, climate, drainage or land rights.
"""

from __future__ import annotations

import numpy as np

from ..preprocessing.transforms import nan_median_filter

FLOOD_MIN_WATER_FRACTION = 0.01   # >= 1 % of valid pixels classified as water
VEG_VH_MIN_DB = -19.0
BUILT_VV_MIN_DB = -6.0

NODATA, WATER, VEGETATION, BARE, BUILT = -1, 0, 1, 2, 3
CLASS_KEYS = {WATER: "water", VEGETATION: "vegetation", BARE: "bare", BUILT: "built"}
CLASS_LABELS = {"water": "Water-covered", "vegetation": "Vegetation",
                "bare": "Bare / sparse land", "built": "Built-up / very bright"}
CLASS_RGB = {"water": (255, 122, 102), "vegetation": (126, 200, 92),
             "bare": (214, 178, 112), "built": (168, 156, 240), "nodata": (52, 62, 78)}

RULES = [
    "Water: pixels the selected method classified as water",
    f"Built-up / very bright: VV ≥ {BUILT_VV_MIN_DB:g} dB (double-bounce scattering)",
    f"Vegetation: VH ≥ {VEG_VH_MIN_DB:g} dB (canopy volume scattering)",
    "Bare / sparse: remaining valid pixels",
]


# ----------------------------------------------------------------------------- flood
def flood_verdict(counts: dict | None) -> dict | None:
    if not counts or not counts.get("valid"):
        return None
    frac = counts["water"] / counts["valid"]
    flooded = frac >= FLOOD_MIN_WATER_FRACTION
    return {
        "flooded": flooded,
        "label": "Flood-affected" if flooded else "No flood water detected",
        "water_fraction": round(frac, 6),
        "min_fraction": FLOOD_MIN_WATER_FRACTION,
        "rule": f"Flood-affected when at least {100 * FLOOD_MIN_WATER_FRACTION:g}% of valid pixels "
                "are classified as water (notebook 02 labelled a chip 'Flood' when it contained "
                "water; the minimum share keeps isolated speckle pixels from deciding the verdict).",
    }


def legacy_agreement(verdict: dict | None, legacy: dict | None) -> bool | None:
    """Does the original scene classifier (correctly interpreted) agree with the verdict?"""
    if not verdict or not legacy:
        return None
    return (legacy.get("label") == "Surface water present") == verdict["flooded"]


# ----------------------------------------------------------------------------- land cover
def classify_land(image_chw: np.ndarray, water_mask: np.ndarray, valid: np.ndarray,
                  median_size: int = 3) -> np.ndarray:
    """Per-pixel class map (int8): -1 no data, 0 water, 1 vegetation, 2 bare, 3 built."""
    if image_chw.shape[0] < 2:
        raise ValueError("land-surface screening needs VV and VH bands")
    vv = nan_median_filter(image_chw[0], median_size)
    vh = nan_median_filter(image_chw[1], median_size)
    out = np.full(valid.shape, BARE, dtype=np.int8)
    out[np.nan_to_num(vh, nan=-99) >= VEG_VH_MIN_DB] = VEGETATION
    out[np.nan_to_num(vv, nan=-99) >= BUILT_VV_MIN_DB] = BUILT
    out[water_mask] = WATER
    out[~valid] = NODATA
    return out


def land_counts(classes: np.ndarray) -> dict:
    counts = {key: int((classes == code).sum()) for code, key in CLASS_KEYS.items()}
    counts["valid"] = int((classes >= 0).sum())
    counts["nodata"] = int((classes < 0).sum())
    return counts


def land_rgb(classes: np.ndarray) -> np.ndarray:
    out = np.empty(classes.shape + (3,), dtype=np.uint8)
    out[:] = CLASS_RGB["nodata"]
    for code, key in CLASS_KEYS.items():
        out[classes == code] = CLASS_RGB[key]
    return out


def plantation_screening(counts: dict | None, flooded: bool | None) -> dict | None:
    """Indicative triage from land-cover shares. Thresholds are shown to the user verbatim."""
    if not counts or not counts.get("valid"):
        return None
    n = counts["valid"]
    share = {k: counts[k] / n for k in ("water", "vegetation", "bare", "built")}
    usable = share["vegetation"] + share["bare"]
    notes = []
    if share["water"] >= 0.5:
        code, label = "no", "Not suitable now"
        reason = "Most of the area is water-covered at acquisition time."
    elif share["built"] >= 0.5:
        code, label = "no", "Not suitable"
        reason = "Most of the area shows built-up or very bright surfaces."
    elif usable >= 0.5:
        code, label = "yes", "Potentially suitable — survey recommended"
        kind = "already vegetated" if share["vegetation"] >= share["bare"] else "mostly open, bare land"
        reason = f"{100 * usable:.0f}% of the area is land without water or built-up signal ({kind})."
    else:
        code, label = "maybe", "Inconclusive"
        reason = "The area is a mix of water, built-up and open land with no dominant usable share."
    if flooded and code != "no":
        notes.append("Part of the area is water-covered; check drainage and flood recurrence first.")
    return {
        "code": code, "label": label, "reason": reason, "notes": notes,
        "usable_fraction": round(usable, 6),
        "criteria": [
            "Not suitable if ≥ 50% is water-covered or ≥ 50% is built-up / very bright",
            "Potentially suitable if ≥ 50% is vegetation or bare land",
            "Otherwise inconclusive",
        ],
        "caveat": "Rule-based screening from radar backscatter only. It does not assess soil, "
                  "slope, rainfall, drainage, crop type or land ownership, and the thresholds "
                  "have not been validated against labelled land-cover data.",
    }
