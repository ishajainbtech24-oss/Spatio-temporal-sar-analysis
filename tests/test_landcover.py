import json
import re

import numpy as np
import pytest

from sar_flood.inference import landcover as lc


def test_flood_verdict_threshold():
    assert lc.flood_verdict({"water": 0, "valid": 1000})["flooded"] is False
    assert lc.flood_verdict({"water": 9, "valid": 1000})["flooded"] is False     # 0.9 %
    assert lc.flood_verdict({"water": 10, "valid": 1000})["flooded"] is True     # 1.0 %
    assert lc.flood_verdict({"water": 0, "valid": 0}) is None
    assert lc.flood_verdict(None) is None


def test_legacy_agreement():
    v = lc.flood_verdict({"water": 500, "valid": 1000})
    assert lc.legacy_agreement(v, {"label": "Surface water present"}) is True
    assert lc.legacy_agreement(v, {"label": "No surface water"}) is False
    assert lc.legacy_agreement(v, None) is None


def test_classify_land_rules():
    vv = np.array([[-20, -10, -10, -3]], np.float32).repeat(3, 0)
    vh = np.array([[-28, -15, -24, -10]], np.float32).repeat(3, 0)
    img = np.stack([vv, vh])
    water = np.zeros(vv.shape, bool)
    water[:, 0] = True
    valid = np.ones(vv.shape, bool)
    valid[2, 3] = False
    c = lc.classify_land(img, water, valid, median_size=0)
    assert (c[:, 0] == lc.WATER).all()
    assert (c[:, 1] == lc.VEGETATION).all()   # VH -15 >= -19
    assert (c[:, 2] == lc.BARE).all()         # VH -24 < -19
    assert (c[:2, 3] == lc.BUILT).all()       # VV -3 >= -6 wins over vegetation
    assert c[2, 3] == lc.NODATA
    counts = lc.land_counts(c)
    assert counts == {"water": 3, "vegetation": 3, "bare": 3, "built": 2, "valid": 11, "nodata": 1}


def test_classify_needs_two_bands():
    with pytest.raises(ValueError):
        lc.classify_land(np.zeros((1, 4, 4)), np.zeros((4, 4), bool), np.ones((4, 4), bool))


@pytest.mark.parametrize("counts,flooded,code", [
    ({"water": 60, "vegetation": 20, "bare": 10, "built": 10, "valid": 100}, True, "no"),
    ({"water": 5, "vegetation": 5, "bare": 30, "built": 60, "valid": 100}, False, "no"),
    ({"water": 10, "vegetation": 60, "bare": 20, "built": 10, "valid": 100}, True, "yes"),
    ({"water": 40, "vegetation": 20, "bare": 5, "built": 35, "valid": 100}, True, "maybe"),
])
def test_plantation_screening(counts, flooded, code):
    sc = lc.plantation_screening(counts, flooded)
    assert sc["code"] == code
    assert sc["criteria"] and sc["caveat"]
    if code == "yes" and flooded:
        assert sc["notes"]  # flood caveat added


def test_screening_no_valid_pixels():
    assert lc.plantation_screening({"water": 0, "vegetation": 0, "bare": 0, "built": 0, "valid": 0}, False) is None


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SAR_OUTPUT_DIR", str(tmp_path / "outputs"))
    from sar_flood.web.app import create_app

    app = create_app({"TESTING": True, "USE_LEGACY": False, "WEB_RUNS_DIR": tmp_path / "web"})
    return app.test_client(), tmp_path / "web"


def test_page_land_cover_matches_stored_counts(client):
    c, web = client
    r = c.post("/analyze", data={"sample": "synthetic/Synthetic_Foxtrot_1000_S1Hand.tif"})
    rid = re.search(r"/results/([0-9a-f]{32})", r.headers["Location"]).group(1)
    res = json.loads((web / rid / "result.json").read_text())
    m = res["methods"][res["primary_method"]]
    land = m["land_cover"]["counts"]
    # land-cover water equals the flood mask's water; classes partition the valid pixels
    assert land["water"] == m["counts"]["water"]
    assert land["water"] + land["vegetation"] + land["bare"] + land["built"] == m["counts"]["valid"]
    assert m["verdict"]["flooded"] == (m["counts"]["water"] / m["counts"]["valid"] >= lc.FLOOD_MIN_WATER_FRACTION)
    html = c.get(f"/results/{rid}").data.decode()
    assert m["verdict"]["label"] in html
    assert m["land_cover"]["screening"]["label"] in html
    for k in ("vegetation", "bare", "built"):
        assert f"{100 * land[k] / land['valid']:.1f}%" in html
    assert (web / rid / m["land_cover"]["map"]).exists()


def test_png_has_verdict_but_no_land_cover(client):
    import io

    import cv2

    c, _ = client
    img = np.full((64, 64), 200, np.uint8)
    img[:, :30] = 10
    _, buf = cv2.imencode(".png", img)
    r = c.post("/analyze", data={"image": (io.BytesIO(buf.tobytes()), "x.png")},
               content_type="multipart/form-data")
    html = c.get(r.headers["Location"]).data.decode()
    assert "Flood-affected" in html
    assert "needs a calibrated Sentinel-1 GeoTIFF with both VV and VH" in html
    assert "Land cover &amp; vegetation analysis" not in html
