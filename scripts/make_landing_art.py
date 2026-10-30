"""Generate the landing-page illustration: a radar-style flood scene and its flood map.

The scene is procedural (noise, a meandering river, field parcels, a town, a flooded
floodplain), rendered with radar-like speckle. It is an ILLUSTRATION, labelled as such on
the page, not real satellite data. Re-run to regenerate:

    python scripts/make_landing_art.py

To use a real image instead, replace the two files in src/sar_flood/web/static/images/
(same size and framing) — for example a Sen1Floods11 chip and its mask exported by
`sar-flood predict` — and update the hero caption in overview.html.
"""

from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

W, H = 1800, 1013
OUT = Path(__file__).resolve().parents[1] / "src" / "sar_flood" / "web" / "static" / "images"
rng = np.random.default_rng(20261030)


def fractal(shape, scales=(220, 90, 35, 12), weights=(1.0, 0.55, 0.28, 0.12)):
    f = np.zeros(shape, np.float32)
    for s, w in zip(scales, weights, strict=True):
        n = gaussian_filter(rng.normal(size=shape).astype(np.float32), s)
        f += w * n / (n.std() + 1e-8)
    return (f - f.mean()) / f.std()


yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

# --- terrain: low valley along a diagonal, rising to the edges
relief = fractal((H, W))
valley = ((yy - (0.62 * H - 0.18 * (xx - W / 2))) / H) ** 2 * 9.0
elev = relief * 0.6 + valley

# --- main river: meandering centreline + a tributary from the top
cy = (0.60 * H - 0.18 * (xx - W / 2) + 70 * np.sin(xx / 170) + 38 * np.sin(xx / 61 + 1.3)
      + 22 * relief * 0)
river = np.abs(yy - cy) < (9 + 3 * np.sin(xx / 90))
tx = 0.36 * W + 55 * np.sin(yy / 95) + 20 * np.sin(yy / 33)
trib = (np.abs(xx - tx) < 5) & (yy < cy)
channels = river | trib

# --- flood: low ground near the channels
dist = distance_transform_edt(~channels).astype(np.float32)
flood_field = 0.35 * elev + dist / 55.0 + 0.55 * fractal((H, W), (45, 14, 5), (1.0, 0.45, 0.18))
flood = (flood_field < np.quantile(flood_field, 0.15)) | channels
# a few isolated ponds
ponds = (fractal((H, W), (26, 8), (1.0, 0.3)) > 2.75) & (dist > 60)
water = flood | ponds
water = cv2.morphologyEx(water.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
water |= channels

# --- land: rotated field parcels with varied brightness and texture
ang = np.deg2rad(17)
u = (xx * np.cos(ang) + yy * np.sin(ang)) / 74.0
v = (-xx * np.sin(ang) + yy * np.cos(ang)) / 41.0
warp = fractal((H, W), (260,), (1.0,)) * 0.45
pid = (np.floor(u + warp) * 7919 + np.floor(v + warp * 0.6) * 104729).astype(np.int64)
lut = np.random.default_rng(7).uniform(0.46, 0.74, size=4096).astype(np.float32)
parcels = lut[np.abs(pid) % 4096]
edges = (np.abs(((u + warp) % 1) - 0.5) > 0.485) | (np.abs(((v + warp * 0.6) % 1) - 0.5) > 0.48)
parcels = gaussian_filter(parcels, 1.2)
land = parcels * (1 - 0.10 * edges) + 0.04 * fractal((H, W), (10, 3), (1.0, 0.5))

# forest on high ground: brighter, rougher
forest = elev > np.quantile(elev, 0.80)
land = np.where(forest, 0.72 + 0.07 * fractal((H, W), (4, 2), (1.0, 0.6)), land)

# town: very bright, blocky cluster on dry ground above the river
town = ((xx - 0.70 * W) ** 2 / (150 ** 2) + (yy - 0.30 * H) ** 2 / (95 ** 2)) < 1
bld = np.zeros((H, W), np.uint8)
brng = np.random.default_rng(3)
for _ in range(900):
    x0, y0 = brng.normal(0.70 * W, 85), brng.normal(0.30 * H, 52)
    w_, h_ = brng.integers(3, 9), brng.integers(3, 8)
    cv2.rectangle(bld, (int(x0), int(y0)), (int(x0 + w_), int(y0 + h_)), 1, -1)
town = bld.astype(bool)
land = np.where(town, 1.25, land)

# roads
road1 = np.abs(yy - (0.18 * H + 0.30 * (xx - 0.2 * W))) < 2.2
road2 = np.abs(xx - (0.78 * W - 0.25 * (yy - 0.1 * H))) < 2.0
land = np.where((road1 | road2) & ~water, 0.95, land)

# terrain shading
gy, gx = np.gradient(gaussian_filter(elev, 6))
land = land * (1 + 6 * (gx * 0.7 - gy * 0.7))

# --- radar look: water dark and smooth, speckle everywhere
base = np.where(water, 0.07 + 0.02 * fractal((H, W), (20,), (1.0,)), land)
looks = 5.0
speckle = rng.gamma(looks, 1.0 / looks, size=(H, W)).astype(np.float32)
img = np.clip(base * speckle, 0, 1.4)
img = gaussian_filter(img, 0.6)
img = np.clip(img / np.quantile(img, 0.995), 0, 1) ** 0.85

# tint to the site's night palette: deep navy -> pale ice
dark, light = np.array([22, 14, 9], np.float32), np.array([242, 234, 226], np.float32)  # BGR
radar = (dark + (light - dark) * img[..., None]).astype(np.uint8)

# --- analysed: same scene, water in coral, crisp edge
coral = np.array([102, 122, 255], np.float32)  # BGR of #FF7A66
analysed = radar.astype(np.float32)
w3 = water[..., None]
analysed = np.where(w3, analysed * 0.25 + coral * 0.75, analysed * 0.82)
edge = cv2.morphologyEx(water.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
analysed[edge] = [150, 170, 255]
analysed = analysed.astype(np.uint8)

OUT.mkdir(parents=True, exist_ok=True)
params = [cv2.IMWRITE_WEBP_QUALITY, 72]
cv2.imwrite(str(OUT / "hero-radar.webp"), radar, params)
cv2.imwrite(str(OUT / "hero-flood.webp"), analysed, params)
# smaller crops reused lower on the page
crop = analysed[int(0.38 * H):int(0.38 * H) + 540, int(0.10 * W):int(0.10 * W) + 810]
cv2.imwrite(str(OUT / "detail-flood.webp"), crop, params)
band = radar[int(0.48 * H):int(0.48 * H) + 420, :]
cv2.imwrite(str(OUT / "band-radar.webp"), cv2.resize(band, (1600, 336)), params)
print(f"water share in illustration: {water.mean():.1%}; files in {OUT}")
