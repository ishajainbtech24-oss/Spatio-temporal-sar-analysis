"""Analyse one input file with every available method and write quick-looks + results.

Used by both ``sar-flood predict`` and the dashboard, so the two always agree.

Supported inputs
----------------
* **Sentinel-1 GeoTIFF in Sen1Floods11 S1Hand format** (band 1 = VV dB, band 2 = VH dB,
  NaN = no data): U-Net (if a checkpoint exists), both threshold baselines, legacy
  classifier; georeferenced outputs and map bounds.
* **8-bit PNG/JPEG**: has no radiometric calibration, so dB-based methods do not apply.
  Only the legacy scene classifier (which was trained on per-image-stretched data) and
  a relative Otsu split of the grey levels are run, and the result is flagged.

Every number stored in ``result.json`` (pixel counts, fractions, areas, the spatial grid)
is computed here from the binary masks, so the dashboard and the PDF report only display
and never re-derive or estimate them.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..evaluation.baselines import fixed_threshold_water, otsu_water
from ..evaluation.metrics import Confusion
from ..utils import get_logger, save_json, utc_now
from ..visualization import render
from . import landcover

log = get_logger(__name__)

GEOTIFF_EXT = {".tif", ".tiff"}
RASTER_EXT = {".png", ".jpg", ".jpeg"}
ALLOWED_EXT = GEOTIFF_EXT | RASTER_EXT
GRID_MAX = 16  # spatial summary grid is at most GRID_MAX x GRID_MAX cells


class InputError(ValueError):
    """User-facing problem with an input file (message is safe to show)."""


def looks_like_db(image_chw: np.ndarray) -> bool:
    finite = image_chw[np.isfinite(image_chw)]
    if finite.size == 0:
        return False
    med = float(np.median(finite))
    return -45.0 <= med <= 5.0 and float(finite.max()) <= 40.0


def _read_geotiff(path: Path):
    import rasterio  # noqa: PLC0415
    from rasterio.errors import RasterioIOError  # noqa: PLC0415

    try:
        with rasterio.open(path) as src:
            if src.count < 1 or src.width < 16 or src.height < 16:
                raise InputError("GeoTIFF must have at least one band and be at least 16x16 pixels.")
            if src.width * src.height > 4096 * 4096:
                raise InputError("Image is too large for interactive analysis (max 4096x4096).")
            arr = src.read(list(range(1, min(src.count, 2) + 1))).astype(np.float32)
            if src.nodata is not None and not np.isnan(src.nodata):
                arr[arr == src.nodata] = np.nan
            profile = {"crs": src.crs, "transform": src.transform, "bounds": tuple(src.bounds)}
    except RasterioIOError as exc:
        raise InputError("The file could not be read as a GeoTIFF.") from exc
    arr[~np.isfinite(arr)] = np.nan
    return arr, profile


def _read_label(path: Path, shape) -> np.ndarray:
    from ..data.sen1floods11 import read_label  # noqa: PLC0415

    try:
        lab = read_label(path)
    except Exception as exc:
        raise InputError("The ground-truth file could not be read as a GeoTIFF.") from exc
    if lab.shape != shape:
        raise InputError(f"Ground truth is {lab.shape[1]}x{lab.shape[0]} but the image is "
                         f"{shape[1]}x{shape[0]}; they must match.")
    if not set(np.unique(lab).tolist()) <= {-1, 0, 1}:
        raise InputError("Ground truth must contain only -1 (no data), 0 (land) and 1 (water).")
    return lab


def map_bounds(profile: dict) -> dict | None:
    """Lat/lon bounds for a Leaflet overlay, or None if the raster is not georeferenced."""
    crs = profile.get("crs")
    if crs is None or profile.get("transform") is None:
        return None
    from rasterio.warp import transform_bounds  # noqa: PLC0415

    try:
        w, s, e, n = transform_bounds(crs, "EPSG:4326", *profile["bounds"])
    except Exception:
        return None
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        return None
    return {"south": s, "west": w, "north": n, "east": e,
            "exact": str(crs).upper() in {"EPSG:4326", "OGC:CRS84"}}


def pixel_area(profile: dict, bounds: dict | None) -> dict | None:
    """Ground area of one pixel in m², or None when it cannot be derived honestly.

    * Projected CRS in metres: exact |a·e| from the geotransform.
    * Geographic CRS (degrees): spherical approximation at the chip's centre latitude
      (accurate to well under 1 % for chips a few km across).
    * Rotated/sheared rasters, other units, or no georeferencing: None.
    """
    t, crs = profile.get("transform"), profile.get("crs")
    if t is None or crs is None or t.b != 0 or t.d != 0:
        return None
    try:
        if crs.is_projected:
            units = (crs.linear_units or "").lower()
            if units not in {"metre", "meter", "metres", "meters", "m"}:
                return None
            return {"m2": abs(t.a * t.e), "method": "projected CRS (exact)",
                    "x_m": abs(t.a), "y_m": abs(t.e)}
        if crs.is_geographic and bounds is not None:
            lat = math.radians((bounds["south"] + bounds["north"]) / 2)
            x_m = abs(t.a) * 111_320.0 * math.cos(lat)
            y_m = abs(t.e) * 110_574.0
            return {"m2": x_m * y_m, "method": "geographic CRS (spherical approximation)",
                    "x_m": x_m, "y_m": y_m}
    except Exception:
        return None
    return None


def spatial_grid(mask: np.ndarray, valid: np.ndarray, max_cells: int = GRID_MAX) -> dict:
    """Water share per block of a coarse grid; cells with <10 % valid pixels are None."""
    h, w = mask.shape
    rows = max(1, min(max_cells, h // 8))
    cols = max(1, min(max_cells, w // 8))
    r_edges = np.linspace(0, h, rows + 1).astype(int)
    c_edges = np.linspace(0, w, cols + 1).astype(int)
    cells = []
    for i in range(rows):
        row = []
        for j in range(cols):
            v = valid[r_edges[i]:r_edges[i + 1], c_edges[j]:c_edges[j + 1]]
            m = mask[r_edges[i]:r_edges[i + 1], c_edges[j]:c_edges[j + 1]]
            if v.size == 0 or v.mean() < 0.1:
                row.append(None)
            else:
                row.append(round(float(m[v].mean()), 4))
        cells.append(row)
    return {"rows": rows, "cols": cols, "cells": cells}


def pixel_counts(mask: np.ndarray, valid: np.ndarray) -> dict:
    total = int(valid.size)
    n_valid = int(valid.sum())
    water = int((mask & valid).sum())
    return {"water": water, "non_water": n_valid - water, "valid": n_valid,
            "nodata": total - n_valid, "total": total}


def analyze_file(image_path: str | Path, out_dir: str | Path, label_path: str | Path | None = None,
                 segmenter=None, use_legacy: bool = True) -> dict:
    t_start = time.perf_counter()
    image_path, out_dir = Path(image_path), Path(out_dir)
    ext = image_path.suffix.lower()
    if ext not in ALLOWED_EXT:
        raise InputError("Unsupported file type. Use a GeoTIFF (.tif) or PNG/JPEG.")
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"created": utc_now(), "status": "completed", "input": image_path.name,
                    "methods": {}, "images": {}, "warnings": [],
                    "palette_version": render.PALETTE_VERSION}

    if ext in GEOTIFF_EXT:
        image, profile = _read_geotiff(image_path)
        calibrated = looks_like_db(image)
    else:
        import cv2  # noqa: PLC0415

        grey = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if grey is None:
            raise InputError("The file could not be decoded as an image.")
        if min(grey.shape) < 16:
            raise InputError("Image must be at least 16x16 pixels.")
        image, profile, calibrated = grey[None].astype(np.float32), {}, False

    valid = np.isfinite(image).all(axis=0)
    if not valid.any():
        raise InputError("The image contains no valid pixels.")
    h, w = valid.shape
    bounds = map_bounds(profile) if profile else None
    result.update({
        "width": w, "height": h, "bands": int(image.shape[0]),
        "band_names": (["VV", "VH"][: image.shape[0]] if calibrated else ["grey"]),
        "input_format": "GeoTIFF" if ext in GEOTIFF_EXT else ext.lstrip(".").upper(),
        "calibrated_db": calibrated,
        "nodata_fraction": round(float(1 - valid.mean()), 4),
        "crs": str(profile["crs"]) if profile.get("crs") is not None else None,
        "pixel_area": pixel_area(profile, bounds) if profile else None,
        "preprocessing": (
            ["NaN / ±inf marked as no data", "3×3 NaN-aware median filter",
             "clip to −50…1 dB, scale to 0…1 (U-Net input)"]
            if calibrated else ["8-bit grey levels used as-is (no radiometric calibration)"]),
    })
    if not calibrated:
        result["warnings"].append(
            "Input is not calibrated Sentinel-1 backscatter in dB (e.g. an 8-bit PNG). "
            "Segmentation and the dB threshold baseline were skipped; Otsu is applied to "
            "relative grey levels only.")

    label = _read_label(Path(label_path), (h, w)) if label_path else None
    if label is not None:
        label = label.copy()
        label[~valid] = -1
        result["ground_truth"] = pixel_counts(label == 1, label >= 0)

    # ---------------------------------------------------------------- quick-looks
    if calibrated:
        base = render.sar_grey(image, 0)
        result["images"]["vv"] = render.save_png(base, out_dir / "vv.png").name
        if image.shape[0] > 1:
            result["images"]["vh"] = render.save_png(render.sar_grey(image, 1), out_dir / "vh.png").name
            result["images"]["false_colour"] = render.save_png(
                render.sar_false_colour(image), out_dir / "false_colour.png").name
    else:
        g = np.nan_to_num(image[0]).astype(np.uint8)
        base = np.repeat(g[..., None], 3, axis=-1)
        result["images"]["vv"] = render.save_png(base, out_dir / "input.png").name
    if label is not None:
        result["images"]["ground_truth"] = render.save_png(render.label_rgb(label), out_dir / "ground_truth.png").name

    def add_mask_method(name: str, mask: np.ndarray, extra: dict, seconds: float,
                        prob: np.ndarray | None = None):
        mask = mask & valid
        counts = pixel_counts(mask, valid)
        entry = {**extra, "counts": counts,
                 "water_fraction": round(counts["water"] / counts["valid"], 6) if counts["valid"] else None,
                 "inference_ms": round(seconds * 1000, 1),
                 "mask": render.save_png(render.mask_rgb(mask, valid), out_dir / f"{name}_mask.png").name,
                 "overlay": render.save_png(render.overlay(base, mask), out_dir / f"{name}_overlay.png").name,
                 "water_layer": render.save_png(render.water_layer(mask), out_dir / f"{name}_water.png").name,
                 "grid": spatial_grid(mask, valid),
                 "verdict": landcover.flood_verdict(counts)}
        if calibrated and image.shape[0] >= 2:
            classes = landcover.classify_land(image, mask, valid, 3)
            lc_counts = landcover.land_counts(classes)
            entry["land_cover"] = {
                "counts": lc_counts,
                "map": render.save_png(landcover.land_rgb(classes), out_dir / f"{name}_landcover.png").name,
                "rules": landcover.RULES,
                "screening": landcover.plantation_screening(
                    lc_counts, (entry["verdict"] or {}).get("flooded")),
            }
        if prob is not None:
            entry["probability"] = render.save_png(render.probability_rgb(prob), out_dir / f"{name}_prob.png").name
            entry["output_type"] = "probability map, thresholded"
        else:
            entry["output_type"] = "binary threshold rule"
        if label is not None:
            entry["metrics"] = Confusion().update(mask, label).metrics()
            entry["agreement"] = render.save_png(render.error_map(mask, label),
                                                 out_dir / f"{name}_agreement.png").name
        if profile.get("crs") is not None:
            entry["geotiff"] = render.save_mask_geotiff(mask, valid, profile,
                                                        out_dir / f"{name}_mask.tif").name
        result["methods"][name] = entry

    # ---------------------------------------------------------------- methods
    if segmenter is not None and calibrated:
        need = len(segmenter.cfg.data.bands)
        if image.shape[0] < need:
            result["warnings"].append(f"U-Net needs {need} bands (VV+VH); this file has {image.shape[0]}.")
        else:
            t0 = time.perf_counter()
            out = segmenter.predict_chip(image[:need])
            meta, mcfg = segmenter.metadata, segmenter.cfg.model
            add_mask_method("unet", out["mask"], {
                "title": "U-Net (autoencoder-initialised)", "threshold": segmenter.threshold,
                "trained_on": meta.get("data_source", "unknown"),
                "model": {
                    "architecture": f"U-Net, depth {mcfg.depth}, {mcfg.base_filters} base filters",
                    "encoder_init": meta.get("encoder_init"),
                    "checkpoint": meta.get("run_name"),
                    "created": meta.get("created"),
                    "bands": meta.get("bands"),
                    "val_iou": ((meta.get("val_metrics") or {}).get("micro") or {}).get("iou"),
                }}, time.perf_counter() - t0, out["prob"])
    elif segmenter is None:
        result["warnings"].append("No trained U-Net checkpoint found; showing baselines only.")

    if calibrated:
        t0 = time.perf_counter()
        m, th = otsu_water(image, 3)
        add_mask_method("otsu", m, {"title": "Otsu threshold on VV", "threshold_db": round(th, 2),
                                    "model": {"architecture": "Per-chip Otsu threshold on median-filtered VV (dB)"}},
                        time.perf_counter() - t0)
        t0 = time.perf_counter()
        fm = fixed_threshold_water(image, -15.0, 3)
        add_mask_method("fixed", fm, {"title": "Fixed VV threshold", "threshold_db": -15.0,
                                      "model": {"architecture": "VV < −15 dB on median-filtered VV"}},
                        time.perf_counter() - t0)
    else:
        from skimage.filters import threshold_otsu  # noqa: PLC0415

        t0 = time.perf_counter()
        g = np.nan_to_num(image[0])
        th = float(threshold_otsu(g)) if g.max() > g.min() else float("nan")
        add_mask_method("otsu", g < th, {"title": "Otsu on grey levels (uncalibrated)",
                                         "threshold_grey": round(th, 1),
                                         "model": {"architecture": "Otsu threshold on 8-bit grey levels"}},
                        time.perf_counter() - t0)

    if use_legacy:
        try:
            from ..models.legacy import (  # noqa: PLC0415
                interpret_output,
                legacy_preprocess_vv,
                load_legacy_classifier,
            )

            model = _legacy_cache.get("m") or load_legacy_classifier()
            _legacy_cache["m"] = model
            p = float(model.predict(legacy_preprocess_vv(image[0]), verbose=0)[0, 0])
            result["legacy_classifier"] = interpret_output(p)
        except ImportError:
            result["warnings"].append("TensorFlow is not installed; legacy classifier skipped.")
        except Exception as exc:  # keep the rest of the analysis usable
            log.exception("legacy classifier failed")
            result["warnings"].append(f"Legacy classifier failed: {type(exc).__name__}")

    if not calibrated or image.shape[0] < 2:
        result["land_cover_unavailable"] = (
            "Land-surface screening needs a calibrated Sentinel-1 GeoTIFF with both VV and VH bands.")
    for entry in result["methods"].values():
        if entry.get("verdict"):
            entry["verdict"]["legacy_agrees"] = landcover.legacy_agreement(
                entry["verdict"], result.get("legacy_classifier"))
    result["primary_method"] = "unet" if "unet" in result["methods"] else next(iter(result["methods"]))
    result["map"] = bounds
    result["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
    save_json(result, out_dir / "result.json")
    return result


_legacy_cache: dict = {}
