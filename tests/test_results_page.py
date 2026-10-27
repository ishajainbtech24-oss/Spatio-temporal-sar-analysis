"""Results dashboard: numbers must match the masks on disk; edge cases must not crash."""

import io
import json
import re
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def app_and_client(tmp_path, monkeypatch):
    monkeypatch.setenv("SAR_OUTPUT_DIR", str(tmp_path / "outputs"))
    from sar_flood.web.app import create_app

    app = create_app({"TESTING": True, "USE_LEGACY": False, "WEB_RUNS_DIR": tmp_path / "web"})
    return app, app.test_client(), tmp_path / "web"


def _run_sample(client, sample="synthetic/Synthetic_Foxtrot_1000_S1Hand.tif"):
    r = client.post("/analyze", data={"sample": sample, "use_label": "on"})
    assert r.status_code == 302
    return re.search(r"/results/([0-9a-f]{32})", r.headers["Location"]).group(1)


def test_counts_match_mask_geotiff(app_and_client):
    import rasterio

    _, client, web = app_and_client
    rid = _run_sample(client)
    result = json.loads((web / rid / "result.json").read_text())
    for key, m in result["methods"].items():
        with rasterio.open(web / rid / m["geotiff"]) as src:
            data = src.read(1)
        assert m["counts"]["water"] == int((data == 1).sum()), key
        assert m["counts"]["non_water"] == int((data == 0).sum()), key
        assert m["counts"]["nodata"] == int((data == 255).sum()), key
        assert m["counts"]["valid"] == m["counts"]["water"] + m["counts"]["non_water"]
        assert m["water_fraction"] == pytest.approx(m["counts"]["water"] / m["counts"]["valid"], abs=1e-6)


def test_page_shows_computed_percentages_and_counts(app_and_client):
    _, client, web = app_and_client
    rid = _run_sample(client)
    result = json.loads((web / rid / "result.json").read_text())
    c = result["methods"][result["primary_method"]]["counts"]
    html = client.get(f"/results/{rid}").data.decode()
    pct = 100 * c["water"] / c["valid"]
    assert f"{pct:.1f}" in html and f"{100 - pct:.1f}" in html
    assert f"{c['valid']:,}" in html and f"{c['water']:,}" in html
    for text in ["Flood Detection Analysis", "Land classification breakdown", "Flood prediction summary",
                 "Imagery comparison", "Spatial distribution", "Technical details", "Back to dashboard",
                 "Download report", "Ground truth (reference)", 'role="tablist"']:
        assert text in html, text
    assert "not a probability or model confidence" in html


def test_area_uses_pixel_size_for_geographic_crs(app_and_client):
    _, client, web = app_and_client
    rid = _run_sample(client)
    r = json.loads((web / rid / "result.json").read_text())
    pa = r["pixel_area"]
    assert pa and 5 < pa["x_m"] < 20 and 5 < pa["y_m"] < 20  # synthetic chips: ~10 m pixels
    assert "approximation" in pa["method"]


def test_report_pdf_downloads(app_and_client):
    pytest.importorskip("reportlab")
    _, client, _ = app_and_client
    rid = _run_sample(client)
    r = client.get(f"/results/{rid}/report.pdf")
    assert r.status_code == 200 and r.data[:5] == b"%PDF-"
    assert "attachment" in r.headers["Content-Disposition"]


def test_png_input_has_no_map_and_says_so(app_and_client):
    import cv2

    _, client, _ = app_and_client
    img = np.full((64, 64), 200, np.uint8)
    img[:, :30] = 10
    ok, buf = cv2.imencode(".png", img)
    r = client.post("/analyze", data={"image": (io.BytesIO(buf.tobytes()), "scan.png")},
                    content_type="multipart/form-data")
    html = client.get(r.headers["Location"]).data.decode()
    assert "No georeferencing in this file" in html
    assert 'id="map"' not in html
    assert "area needs georeferencing" in html


def test_missing_result_has_helpful_404(app_and_client):
    _, client, _ = app_and_client
    r = client.get("/results/" + "b" * 32)
    assert r.status_code == 404 and b"Analysis not found" in r.data
    assert client.get("/results/" + "b" * 32 + "/report.pdf").status_code == 404
    assert client.get("/results/not-an-id").status_code == 404


def test_unexpected_failure_gets_failed_page(app_and_client, monkeypatch):
    import sar_flood.web.app as webapp

    def boom(*a, **k):
        raise RuntimeError("internal detail /secret/path")

    monkeypatch.setattr(webapp, "analyze_file", boom)
    _, client, _ = app_and_client
    r = client.post("/analyze", data={"sample": "synthetic/Synthetic_Foxtrot_1000_S1Hand.tif"})
    html = client.get(r.headers["Location"]).data.decode()
    assert "did not complete" in html and "status--failed" in html
    assert "/secret/path" not in html and "Traceback" not in html
    assert "Download report" not in html


def test_old_results_are_recounted_from_saved_mask(tmp_path):
    """Results written before counts were stored (old palette, PNG only) are re-counted exactly."""
    from sar_flood.visualization import render
    from sar_flood.web.results import build_view

    run = tmp_path / ("c" * 32)
    mask = np.zeros((32, 40), bool)
    mask[:10] = True
    valid = np.ones_like(mask)
    valid[:, :5] = False
    lab = np.where(valid, mask.astype(np.int8), -1)
    old = np.empty(lab.shape + (3,), np.uint8)
    old[:] = render.LEGACY_PALETTE["land"]
    old[lab == 1] = render.LEGACY_PALETTE["water"]
    old[lab < 0] = render.LEGACY_PALETTE["ignore"]
    render.save_png(old, run / "otsu_mask.png")  # upscaled x12 like the old pipeline did
    (run / "result.json").write_text(json.dumps({
        "input": "x.png", "width": 40, "height": 32, "methods": {"otsu": {
            "title": "Otsu", "water_fraction": 0.3, "mask": "otsu_mask.png"}},
        "images": {}, "warnings": []}))
    v = build_view(run, run.name)
    c = v["methods"][0]["counts"]
    assert c == {"water": int((mask & valid).sum()), "non_water": int((~mask & valid).sum()),
                 "valid": int(valid.sum()), "nodata": int((~valid).sum()), "total": 32 * 40}
    assert v["methods"][0]["counts_source"] == "re-counted from the mask image"


def test_unrecountable_old_result_shows_empty_state(app_and_client):
    app, client, web = app_and_client
    rid = "d" * 32
    (web / rid).mkdir(parents=True)
    (web / rid / "result.json").write_text(json.dumps({
        "input": "x.tif", "width": 10, "height": 10, "methods": {"otsu": {"title": "Otsu",
                                                                         "mask": "missing.png"}},
        "images": {}, "warnings": []}))
    html = client.get(f"/results/{rid}").data.decode()
    assert "Pixel statistics unavailable" in html
    assert "Not available for this input" in html


def test_no_water_mask_renders_single_segment(tmp_path):
    from sar_flood.web.results import method_view

    m = {"title": "x", "counts": {"water": 0, "non_water": 100, "valid": 100, "nodata": 0, "total": 100}}
    v = method_view(Path(tmp_path), {}, "otsu", m)
    assert v["water_pct"] == 0 and v["donut"]["water_len"] == 0 and v["donut"]["dry_len"] > 0


def test_zero_valid_pixels_does_not_divide_by_zero(tmp_path):
    from sar_flood.web.results import method_view

    m = {"title": "x", "counts": {"water": 0, "non_water": 0, "valid": 0, "nodata": 9, "total": 9}}
    v = method_view(Path(tmp_path), {}, "otsu", m)
    assert v["water_pct"] is None and "donut" not in v


def test_spatial_grid_values():
    from sar_flood.inference.pipeline import spatial_grid

    mask = np.zeros((64, 64), bool)
    mask[:, :32] = True
    valid = np.ones_like(mask)
    valid[:8, :8] = False
    g = spatial_grid(mask, valid)
    assert g["rows"] == g["cols"] == 8
    assert g["cells"][0][0] is None
    assert g["cells"][1][0] == 1.0 and g["cells"][1][7] == 0.0
