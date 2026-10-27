"""Sen1Floods11 hand-labelled chips: discovery, splitting, reading and validation.

On-disk layout (identical in Sen1Floods11 v1.1 and the "v1.2 essentials" archive used
in the original notebooks)::

    <root>/data/flood_events/HandLabeled/S1Hand/<Event>_<id>_S1Hand.tif     2 bands, float32 dB (VV, VH), NaN = no data
    <root>/data/flood_events/HandLabeled/LabelHand/<Event>_<id>_LabelHand.tif 1 band, int16: -1 no data, 0 not water, 1 water
    <root>/splits/flood_handlabeled/flood_{train,valid,test,bolivia}_data.csv  (optional, official splits)

Important: ``LabelHand`` marks *all* surface water visible in the post-event image
(permanent rivers and lakes as well as flood water). A model trained on it is a
surface-water segmenter; "flood" maps require subtracting a permanent-water layer.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

S1_SUBDIR = Path("data/flood_events/HandLabeled/S1Hand")
LABEL_SUBDIR = Path("data/flood_events/HandLabeled/LabelHand")
SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "val": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
}
IGNORE_INDEX = -1
BAND_NAMES = {1: "VV", 2: "VH"}


@dataclass(frozen=True)
class Sample:
    """One image/label pair. ``chip_id`` is e.g. ``Ghana_103272``."""

    chip_id: str
    image_path: Path
    label_path: Path | None

    @property
    def event(self) -> str:
        """Flood event / country, used as the grouping key for leakage-free splits."""
        return self.chip_id.rsplit("_", 1)[0]


@dataclass
class Chip:
    image: np.ndarray            # (bands, H, W) float32, dB, NaN where no data
    label: np.ndarray | None     # (H, W) int16 in {-1, 0, 1}
    profile: dict = field(default_factory=dict)  # rasterio profile (crs, transform, ...)


class DatasetError(RuntimeError):
    """Raised for missing or unusable dataset files with an actionable message."""


# --------------------------------------------------------------------------- discovery
def discover_samples(root: str | Path) -> list[Sample]:
    """Pair every ``*_S1Hand.tif`` with its ``*_LabelHand.tif``. Unpaired files are skipped."""
    root = Path(root)
    s1_dir, lab_dir = root / S1_SUBDIR, root / LABEL_SUBDIR
    if not s1_dir.is_dir() or not lab_dir.is_dir():
        raise DatasetError(
            f"Expected Sen1Floods11 folders not found under '{root}'.\n"
            f"  missing: {s1_dir if not s1_dir.is_dir() else lab_dir}\n"
            "See docs/DATASET.md for download instructions, or set SAR_DATA_ROOT / data.root."
        )
    images = {p.name[: -len("_S1Hand.tif")]: p for p in s1_dir.glob("*_S1Hand.tif")}
    labels = {p.name[: -len("_LabelHand.tif")]: p for p in lab_dir.glob("*_LabelHand.tif")}
    common = sorted(images.keys() & labels.keys())
    return [Sample(k, images[k], labels[k]) for k in common]


def unpaired_files(root: str | Path) -> dict[str, list[str]]:
    root = Path(root)
    images = {p.name[: -len("_S1Hand.tif")] for p in (root / S1_SUBDIR).glob("*_S1Hand.tif")}
    labels = {p.name[: -len("_LabelHand.tif")] for p in (root / LABEL_SUBDIR).glob("*_LabelHand.tif")}
    return {"images_without_label": sorted(images - labels), "labels_without_image": sorted(labels - images)}


# --------------------------------------------------------------------------- splitting
def find_official_split_dir(root: str | Path) -> Path | None:
    root = Path(root)
    for candidate in root.rglob(SPLIT_FILES["train"]):
        if all((candidate.parent / f).exists() for f in SPLIT_FILES.values()):
            return candidate.parent
    return None


def _read_split_csv(path: Path) -> list[str]:
    ids = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if row and row[0].strip().endswith("_S1Hand.tif"):
                ids.append(Path(row[0].strip()).name[: -len("_S1Hand.tif")])
    return ids


def group_split(
    samples: Sequence[Sample], val_fraction: float, test_fraction: float, seed: int
) -> dict[str, list[Sample]]:
    """Assign whole flood events to train/val/test so no event is shared between splits.

    Events are shuffled with ``seed`` and greedily assigned to the split that is furthest
    below its target share of chips. With only ~11 events this is coarse, which is the
    point: it measures generalisation to unseen events/regions.
    """
    events: dict[str, list[Sample]] = {}
    for s in samples:
        events.setdefault(s.event, []).append(s)
    if len(events) < 3:
        raise DatasetError("Group split needs at least 3 distinct events")
    rng = np.random.default_rng(seed)
    order = sorted(events)
    rng.shuffle(order)
    total = len(samples)
    targets = {"test": test_fraction * total, "val": val_fraction * total}
    out: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    # Give test and val one event each first so neither can be empty.
    for name in ("test", "val"):
        out[name].extend(events[order.pop(0)])
    for ev in order:
        deficits = {k: targets[k] - len(out[k]) for k in ("test", "val")}
        best = max(deficits, key=deficits.get)
        if deficits[best] >= len(events[ev]) / 2:
            out[best].extend(events[ev])
        else:
            out["train"].extend(events[ev])
    if not out["train"]:
        raise DatasetError("Group split produced an empty training set; lower val/test fractions")
    return out


def make_splits(
    samples: Sequence[Sample],
    root: str | Path,
    strategy: str = "auto",
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[dict[str, list[Sample]], str]:
    """Return ``({"train","val","test"} -> samples, strategy_used)``."""
    by_id = {s.chip_id: s for s in samples}
    split_dir = find_official_split_dir(root) if strategy in ("auto", "official") else None
    if split_dir is not None:
        splits = {}
        for name, fname in SPLIT_FILES.items():
            ids = _read_split_csv(split_dir / fname)
            splits[name] = [by_id[i] for i in ids if i in by_id]
        overlap = {s.chip_id for s in splits["train"]} & {s.chip_id for s in splits["test"]}
        if overlap:
            raise DatasetError("Official split files overlap between train and test")
        return splits, "official"
    if strategy == "official":
        raise DatasetError(f"split_strategy=official but no {SPLIT_FILES['train']} found under {root}")
    return group_split(samples, val_fraction, test_fraction, seed), "group"


# --------------------------------------------------------------------------- reading
def read_image(path: str | Path, bands: Sequence[int] = (1, 2)) -> tuple[np.ndarray, dict]:
    import rasterio  # noqa: PLC0415  (import lazily: keeps CLI start-up fast)

    with rasterio.open(path) as src:
        if max(bands) > src.count:
            raise DatasetError(f"{Path(path).name}: has {src.count} band(s), need band {max(bands)}")
        arr = src.read(list(bands)).astype(np.float32)
        nodata = src.nodata
        profile = {"crs": src.crs, "transform": src.transform, "width": src.width,
                   "height": src.height, "bounds": tuple(src.bounds)}
    if nodata is not None and not np.isnan(nodata):
        arr[arr == nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan  # treat +/-inf as no data too
    return arr, profile


def read_label(path: str | Path) -> np.ndarray:
    import rasterio  # noqa: PLC0415

    with rasterio.open(path) as src:
        lab = src.read(1)
    lab = np.where(np.isfinite(lab), lab, IGNORE_INDEX).astype(np.int16)
    return lab


def load_chip(sample: Sample, bands: Sequence[int] = (1, 2)) -> Chip:
    image, profile = read_image(sample.image_path, bands)
    label = read_label(sample.label_path) if sample.label_path else None
    if label is not None and label.shape != image.shape[1:]:
        raise DatasetError(
            f"{sample.chip_id}: image {image.shape[1:]} and label {label.shape} shapes differ"
        )
    return Chip(image=image, label=label, profile=profile)


# --------------------------------------------------------------------------- validation
def validate_sample(sample: Sample, bands: Sequence[int] = (1, 2)) -> dict:
    """Check one pair and return a report dict with ``ok`` and ``problems``."""
    import rasterio  # noqa: PLC0415

    report: dict = {"chip_id": sample.chip_id, "event": sample.event, "problems": []}
    try:
        chip = load_chip(sample, bands)
    except Exception as exc:  # corrupted file, wrong band count, shape mismatch
        report["problems"].append(f"unreadable: {exc}")
        report["ok"] = False
        return report
    img, lab = chip.image, chip.label
    report["shape"] = list(img.shape)
    nan_frac = float(np.isnan(img).any(axis=0).mean())
    report["nodata_fraction_image"] = round(nan_frac, 4)
    if nan_frac == 1.0:
        report["problems"].append("image is entirely no-data")
    if lab is not None:
        values = set(np.unique(lab).tolist())
        if not values <= {-1, 0, 1}:
            report["problems"].append(f"unexpected label values {sorted(values - {-1, 0, 1})}")
        valid = lab >= 0
        report["label_valid_fraction"] = round(float(valid.mean()), 4)
        report["water_fraction"] = round(float((lab == 1).sum() / max(valid.sum(), 1)), 4)
        if not valid.any():
            report["problems"].append("label has no valid pixels")
        with rasterio.open(sample.image_path) as a, rasterio.open(sample.label_path) as b:
            if a.crs != b.crs or not a.transform.almost_equals(b.transform):
                report["problems"].append("image and label georeferencing differ")
    report["ok"] = not report["problems"]
    return report


def validate_dataset(root: str | Path, bands: Sequence[int] = (1, 2)) -> dict:
    samples = discover_samples(root)
    if not samples:
        raise DatasetError(f"No image/label pairs found under {root}")
    rows = [validate_sample(s, bands) for s in samples]
    events: dict[str, int] = {}
    for s in samples:
        events[s.event] = events.get(s.event, 0) + 1
    return {
        "root": str(root),
        "n_pairs": len(samples),
        "n_ok": sum(r["ok"] for r in rows),
        "n_with_problems": sum(not r["ok"] for r in rows),
        "n_with_any_nodata": sum(r.get("nodata_fraction_image", 0) > 0 for r in rows),
        "n_without_water": sum(r.get("water_fraction", 1) == 0 for r in rows),
        "events": events,
        "unpaired": unpaired_files(root),
        "official_split_dir": str(find_official_split_dir(root) or ""),
        "samples": rows,
    }


def detect_data_source(root: str | Path) -> str:
    """'synthetic' for data written by :mod:`sar_flood.data.synthetic`, else 'sen1floods11'."""
    return "synthetic" if (Path(root) / "SYNTHETIC_DATA_README.txt").exists() else "sen1floods11"
