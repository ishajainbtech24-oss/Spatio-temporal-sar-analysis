"""Quick-look rendering of SAR chips, masks and overlays (PNG) and mask export (GeoTIFF).

Colour meaning is fixed across PNGs, the dashboard and the PDF report:
coral = water / flooded, teal = not water, slate = no data or unlabelled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PALETTE_VERSION = 2
WATER_RGB = (255, 122, 102)    # coral  #FF7A66
LAND_RGB = (38, 120, 110)      # teal, darkened so large land areas stay calm
IGNORE_RGB = (52, 62, 78)      # slate
FP_RGB = (242, 181, 68)        # false alarm (amber)
FN_RGB = (110, 170, 255)       # missed water (sky blue)

# Palette of results written before PALETTE_VERSION 2 (needed to re-count old PNG masks).
LEGACY_PALETTE = {"water": (31, 111, 178), "land": (214, 200, 160), "ignore": (120, 120, 120)}


def stretch_db(band: np.ndarray, lo: float = -25.0, hi: float = 0.0) -> np.ndarray:
    """dB band -> uint8 grey; no-data becomes black."""
    out = np.clip((band - lo) / (hi - lo), 0, 1)
    out = np.nan_to_num(out, nan=0.0)
    return (out * 255).astype(np.uint8)


def sar_grey(image_chw: np.ndarray, band: int = 0) -> np.ndarray:
    lo, hi = (-25.0, 0.0) if band == 0 else (-32.0, -5.0)
    g = stretch_db(image_chw[band], lo, hi)
    return np.repeat(g[..., None], 3, axis=-1)


def sar_false_colour(image_chw: np.ndarray) -> np.ndarray:
    """R = VV, G = VH, B = VV - VH (dB ratio)."""
    if image_chw.shape[0] < 2:
        return sar_grey(image_chw)
    vv, vh = image_chw[0], image_chw[1]
    r = stretch_db(vv, -25, 0)
    g = stretch_db(vh, -32, -5)
    b = stretch_db(vv - vh, 0, 15)
    return np.stack([r, g, b], axis=-1)


def label_rgb(label: np.ndarray) -> np.ndarray:
    out = np.empty(label.shape + (3,), dtype=np.uint8)
    out[:] = LAND_RGB
    out[label == 1] = WATER_RGB
    out[label < 0] = IGNORE_RGB
    return out


def mask_rgb(mask: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    lab = mask.astype(np.int8)
    if valid is not None:
        lab[~valid] = -1
    return label_rgb(lab)


def overlay(base_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    water = np.array(WATER_RGB, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * water
    return out.astype(np.uint8)


def water_layer(mask: np.ndarray) -> np.ndarray:
    """RGBA image: opaque coral where water, fully transparent elsewhere."""
    out = np.zeros(mask.shape + (4,), dtype=np.uint8)
    out[mask] = (*WATER_RGB, 255)
    return out


def probability_rgb(prob: np.ndarray) -> np.ndarray:
    """0 -> deep navy, 0.5 -> coral, 1 -> pale peach."""
    stops = np.array([[16, 26, 42], [255, 122, 102], [255, 228, 205]], dtype=np.float32)
    p = np.clip(np.nan_to_num(prob), 0, 1)[..., None]
    low = stops[0] + (stops[1] - stops[0]) * np.clip(p * 2, 0, 1)
    high = stops[1] + (stops[2] - stops[1]) * np.clip(p * 2 - 1, 0, 1)
    return np.where(p < 0.5, low, high).astype(np.uint8)


def error_map(mask: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Agreement map: coral = hit, amber = false alarm, blue = missed, slate = ignored."""
    out = label_rgb(np.where(label == 1, 0, label).astype(np.int8))
    out[(mask) & (label == 1)] = WATER_RGB
    out[(mask) & (label == 0)] = FP_RGB
    out[(~mask) & (label == 1)] = FN_RGB
    return out


def save_png(rgb: np.ndarray, path: str | Path, upscale_to: int | None = 512) -> Path:
    """Write RGB or RGBA uint8 as PNG; small images are upscaled by pixel repetition."""
    import cv2  # noqa: PLC0415

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = rgb
    if upscale_to and max(rgb.shape[:2]) < upscale_to:
        f = upscale_to // max(rgb.shape[:2])
        img = np.repeat(np.repeat(rgb, f, axis=0), f, axis=1)
    code = cv2.COLOR_RGBA2BGRA if img.shape[-1] == 4 else cv2.COLOR_RGB2BGR
    if not cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(img), code)):
        raise OSError(f"could not write {path}")
    return path


def save_mask_geotiff(mask: np.ndarray, valid: np.ndarray, profile: dict, path: str | Path) -> Path:
    """uint8 GeoTIFF: 1 = water, 0 = not water, 255 = no data; keeps CRS/transform."""
    import rasterio  # noqa: PLC0415

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.where(valid, mask.astype(np.uint8), 255).astype(np.uint8)
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                       count=1, dtype="uint8", nodata=255, crs=profile.get("crs"),
                       transform=profile.get("transform"), compress="deflate") as dst:
        dst.write(data[None])
    return path
