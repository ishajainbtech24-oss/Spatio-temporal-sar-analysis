import io
import re

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SAR_OUTPUT_DIR", str(tmp_path / "outputs"))  # no checkpoints here
    from sar_flood.web.app import create_app

    app = create_app({"TESTING": True, "USE_LEGACY": False, "WEB_RUNS_DIR": tmp_path / "web",
                      "MAX_CONTENT_LENGTH": 2 * 1024 * 1024})
    return app.test_client()


@pytest.mark.parametrize("url", ["/", "/analyze", "/evaluation"])
def test_pages_render(client, url):
    r = client.get(url)
    assert r.status_code == 200
    assert b"SAR Flood Intelligence" in r.data


LANDING_JARGON = ["U-Net", "autoencoder", "neural", "CNN", "Transformer", "dB", "backscatter",
                  "Sen1Floods11", "IoU", "segmentation", "model", "pixel", "VV", "VH", "Otsu",
                  "threshold", "chip"]


def test_landing_has_no_jargon(client):
    import re

    html = client.get("/").data.decode()
    text = re.sub(r"<[^>]+>", " ", re.sub(r"<(script|style|head)[\s\S]*?</\1>", " ", html))
    for word in LANDING_JARGON:
        assert not re.search(rf"\b{re.escape(word)}\b", text, re.I), word


def test_landing_structure_and_links(client):
    html = client.get("/").data.decode()
    for anchor in ('id="about"', 'id="how"', 'href="#about"', 'href="#how"'):
        assert anchor in html
    assert html.count('href="/analyze"') >= 3  # nav, hero and closing button all start an analysis
    assert "Illustration" in html  # the hero image is labelled honestly
    for img in ("hero-radar.webp", "hero-flood.webp", "detail-flood.webp", "band-radar.webp"):
        assert img in html
        r = client.get(f"/static/images/{img}")
        assert r.status_code == 200 and r.data[:4] == b"RIFF"


def test_analyze_page_plain_language(client):
    html = client.get("/analyze").data.decode()
    assert "Start an analysis" in html and "Try a sample" in html and "Upload your own" in html
    assert "chip" not in html.lower()


def test_evaluation_empty_state_and_history(client):
    html = client.get("/evaluation").data
    assert b"No evaluation results yet" in html and b"not reproduced" in html


def test_health(client):
    j = client.get("/api/health").get_json()
    assert j["status"] == "ok" and j["unet_run"] is None


def test_unknown_page_404(client):
    assert client.get("/nope").status_code == 404
    assert client.get("/results/../../etc").status_code == 404
    assert client.get("/results/" + "a" * 32).status_code == 404


def test_result_file_name_validation(client):
    assert client.get("/results/" + "a" * 32 + "/files/..%2Fsecret.png").status_code == 404
    assert client.get("/results/" + "a" * 32 + "/files/run.py").status_code == 404


def _result_id(resp):
    m = re.search(r"/results/([0-9a-f]{32})", resp.headers["Location"])
    assert m, resp.headers["Location"]
    return m.group(1)


def test_sample_analysis_end_to_end(client):
    r = client.post("/analyze", data={"sample": "synthetic/Synthetic_Foxtrot_1000_S1Hand.tif",
                                      "use_label": "on"})
    assert r.status_code == 302
    rid = _result_id(r)
    page = client.get(f"/results/{rid}")
    assert page.status_code == 200
    html = page.data.decode()
    assert "Otsu threshold on VV" in html and "IoU" in html and "Geographic view" in html
    img = re.search(rf"/results/{rid}/files/([\w.\-]+\.png)", html).group(1)
    f = client.get(f"/results/{rid}/files/{img}")
    assert f.status_code == 200 and f.data[:4] == b"\x89PNG"


def test_upload_png(client):
    import cv2
    import numpy as np

    img = np.full((64, 64), 200, np.uint8)
    img[:, :30] = 10
    ok, buf = cv2.imencode(".png", img)
    r = client.post("/analyze", data={"image": (io.BytesIO(buf.tobytes()), "my scan.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 302 and "/results/" in r.headers["Location"]
    assert b"uncalibrated" in client.get(r.headers["Location"]).data


@pytest.mark.parametrize("data,expected", [
    ({}, b"Choose a file"),
    ({"image": (io.BytesIO(b"MZ..."), "evil.exe")}, b"Unsupported file type"),
    ({"image": (io.BytesIO(b"garbage"), "x.tif")}, b"could not be read"),
    ({"image": (io.BytesIO(b""), "x.png")}, b"empty"),
    ({"sample": "../../etc/passwd"}, b"Unknown sample"),
])
def test_invalid_submissions_show_message_not_traceback(client, data, expected):
    r = client.post("/analyze", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert r.status_code == 200
    assert expected in r.data
    assert b"Traceback" not in r.data and b"/home/" not in r.data


def test_upload_too_large(client):
    big = io.BytesIO(b"0" * (3 * 1024 * 1024))
    r = client.post("/analyze", data={"image": (big, "big.tif")}, content_type="multipart/form-data",
                    follow_redirects=True)
    assert b"too large" in r.data


def test_old_results_are_cleaned_up(tmp_path, monkeypatch):
    import os
    import time

    monkeypatch.setenv("SAR_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("RESULT_TTL_HOURS", "24")
    from sar_flood.web.app import create_app

    web = tmp_path / "web"
    old, fresh, other = web / ("a" * 32), web / ("b" * 32), web / "keep-me"
    for d in (old, fresh, other):
        d.mkdir(parents=True)
    two_days_ago = time.time() - 48 * 3600
    os.utime(old, (two_days_ago, two_days_ago))
    os.utime(other, (two_days_ago, two_days_ago))
    app = create_app({"TESTING": True, "USE_LEGACY": False, "WEB_RUNS_DIR": web})
    app.test_client().post("/analyze", data={"sample": "synthetic/Synthetic_Foxtrot_1000_S1Hand.tif"})
    assert not old.exists()          # expired analysis removed
    assert fresh.exists()            # recent analysis kept
    assert other.exists()            # anything that is not an analysis folder is never touched


def test_results_kept_by_default(client):
    assert client.application.config["RESULT_TTL_HOURS"] == 0


def test_committed_model_is_found_when_no_local_runs(tmp_path, monkeypatch):
    import sar_flood.web.app as webapp

    monkeypatch.delenv("SAR_FLOOD_RUN_DIR", raising=False)
    monkeypatch.setenv("SAR_OUTPUT_DIR", str(tmp_path / "outputs"))  # no local runs
    monkeypatch.setattr(webapp, "PROJECT_ROOT", tmp_path)
    assert webapp.find_run_dir() is None
    run = tmp_path / "models" / "unet" / "real_run"
    run.mkdir(parents=True)
    (run / "model.keras").write_bytes(b"x")
    (run / "metadata.json").write_text("{}")
    assert webapp.find_run_dir() == run
    local = tmp_path / "outputs" / "runs" / "mine"
    local.mkdir(parents=True)
    (local / "model.keras").write_bytes(b"x")
    (local / "metadata.json").write_text("{}")
    assert webapp.find_run_dir() == local  # your own runs take priority locally
