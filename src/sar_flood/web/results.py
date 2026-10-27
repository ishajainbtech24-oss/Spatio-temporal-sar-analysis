"""View model for the results page and the PDF report.

Both render exactly what this module returns, so the page and the report can never show
different numbers. All values come from ``result.json`` written by
:func:`sar_flood.inference.pipeline.analyze_file`. Results created before pixel counts were
stored are re-counted from the saved mask (GeoTIFF if present, otherwise the lossless PNG),
never estimated.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..utils import get_logger, load_json
from ..visualization import render

log = get_logger(__name__)

DONUT_R = 64.0
DONUT_C = 2 * math.pi * DONUT_R

METHOD_SHORT = {"unet": "U-Net", "otsu": "Otsu", "fixed": "Fixed −15 dB"}


class ResultNotFound(LookupError):
    pass


def load_result(run_dir: Path) -> dict:
    path = run_dir / "result.json"
    if not path.exists():
        raise ResultNotFound(run_dir.name)
    return load_json(path)


# ----------------------------------------------------------------------------- re-counting
def _counts_from_geotiff(path: Path) -> dict | None:
    try:
        import rasterio  # noqa: PLC0415

        with rasterio.open(path) as src:
            data = src.read(1)
    except Exception:
        return None
    valid = data != 255
    return {"water": int((data == 1).sum()), "non_water": int((data == 0).sum()),
            "valid": int(valid.sum()), "nodata": int((~valid).sum()), "total": int(data.size)}


def _counts_from_png(path: Path, width: int, height: int, palette_version: int) -> dict | None:
    try:
        import cv2  # noqa: PLC0415

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except Exception:
        return None
    if bgr is None or bgr.shape[1] % width or bgr.shape[0] % height:
        return None
    f = bgr.shape[1] // width
    rgb = bgr[::f, ::f, ::-1]  # undo pixel-repetition upscaling exactly
    if rgb.shape[:2] != (height, width):
        return None
    if palette_version >= 2:
        water, land, ignore = render.WATER_RGB, render.LAND_RGB, render.IGNORE_RGB
    else:
        water, land, ignore = (render.LEGACY_PALETTE[k] for k in ("water", "land", "ignore"))

    def match(c):
        return np.all(rgb == np.array(c, dtype=np.uint8), axis=-1)

    w, l_, ig = match(water), match(land), match(ignore)
    if int(w.sum() + l_.sum() + ig.sum()) != width * height:
        return None  # unexpected colours: refuse rather than guess
    return {"water": int(w.sum()), "non_water": int(l_.sum()), "valid": int(w.sum() + l_.sum()),
            "nodata": int(ig.sum()), "total": width * height}


def method_counts(run_dir: Path, result: dict, method: dict) -> tuple[dict | None, str]:
    if method.get("counts"):
        return method["counts"], "stored at analysis time"
    if method.get("geotiff"):
        c = _counts_from_geotiff(run_dir / method["geotiff"])
        if c:
            return c, "re-counted from the mask GeoTIFF"
    if method.get("mask") and result.get("width") and result.get("height"):
        c = _counts_from_png(run_dir / method["mask"], result["width"], result["height"],
                             result.get("palette_version", 1))
        if c:
            return c, "re-counted from the mask image"
    return None, "unavailable"


# ----------------------------------------------------------------------------- view model
def _fmt_area(m2: float) -> dict:
    return {"m2": m2, "ha": m2 / 1e4, "km2": m2 / 1e6}


def method_view(run_dir: Path, result: dict, key: str, m: dict) -> dict:
    counts, source = method_counts(run_dir, result, m)
    view = {"key": key, "short": METHOD_SHORT.get(key, key), **m,
            "counts": counts, "counts_source": source, "has_counts": counts is not None}
    if counts and counts["valid"] > 0:
        water_pct = 100.0 * counts["water"] / counts["valid"]
        view["water_pct"] = water_pct
        view["dry_pct"] = 100.0 - water_pct
        view["donut"] = {"r": DONUT_R, "c": DONUT_C,
                         "water_len": DONUT_C * water_pct / 100.0,
                         "dry_len": DONUT_C * (100.0 - water_pct) / 100.0}
        pa = result.get("pixel_area")
        if pa and pa.get("m2"):
            view["area"] = {"water": _fmt_area(counts["water"] * pa["m2"]),
                            "dry": _fmt_area(counts["non_water"] * pa["m2"]),
                            "valid": _fmt_area(counts["valid"] * pa["m2"]),
                            "method": pa.get("method")}
    else:
        view["water_pct"] = view["dry_pct"] = None
    return view


def build_view(run_dir: Path, rid: str) -> dict:
    result = load_result(run_dir)
    failed = result.get("status") == "failed"
    methods = [] if failed else [method_view(run_dir, result, k, m)
                                 for k, m in result.get("methods", {}).items()]
    primary = result.get("primary_method")
    if methods and primary not in {m["key"] for m in methods}:
        primary = methods[0]["key"]
    if failed:
        status = {"code": "failed", "label": "Failed"}
    elif result.get("warnings"):
        status = {"code": "warning", "label": "Completed with notes"}
    else:
        status = {"code": "ok", "label": "Completed"}
    return {"rid": rid, "r": result, "methods": methods, "primary": primary, "status": status,
            "has_ground_truth": "ground_truth" in result.get("images", {})}


def fmt_int(n) -> str:
    return "–" if n is None else f"{int(n):,}"


def fmt_area(area: dict | None) -> str:
    if not area:
        return "–"
    if area["km2"] >= 1:
        return f"{area['km2']:,.2f} km²"
    if area["ha"] >= 1:
        return f"{area['ha']:,.1f} ha"
    return f"{area['m2']:,.0f} m²"
