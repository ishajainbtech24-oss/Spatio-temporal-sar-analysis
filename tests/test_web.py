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


def test_overview_states_no_checkpoint(client):
    assert b"None yet" in client.get("/").data


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
