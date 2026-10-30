"""Flask dashboard. Every number and image it shows comes from the real pipeline:

* /            overview, model status (what checkpoints exist and what they were trained on)
* /analyze     upload a GeoTIFF/PNG (or pick a bundled sample) and run every available method
* /results/id  analysis dashboard: KPIs, class breakdown, imagery comparison, map, details
* /results/id/report.pdf  PDF report built from the same view model
* /evaluation  metrics read from outputs/eval/*/metrics.json + clearly separated history
* /api/health  JSON status

Configuration (environment variables):
  SAR_FLOOD_RUN_DIR   trained run to use (default: newest outputs/runs/*/ with a checkpoint)
  SAR_OUTPUT_DIR      outputs folder (default: <repo>/outputs)
  SAR_FLOOD_LEGACY    "0" disables the legacy classifier
  SECRET_KEY          Flask session key for flash messages (random per process if unset)
  MAX_UPLOAD_MB       upload limit (default 25)
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import uuid
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from jinja2 import Undefined
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from ..config import PROJECT_ROOT
from ..inference.pipeline import ALLOWED_EXT, InputError, analyze_file
from ..utils import get_logger, load_json, save_json, utc_now
from .results import ResultNotFound, build_view, fmt_area, fmt_int

log = get_logger("sar_flood.web")

SAMPLES_DIR = PROJECT_ROOT / "samples"
RESULT_ID = re.compile(r"^[0-9a-f]{32}$")
FILE_NAME = re.compile(r"^[A-Za-z0-9_.\-]+\.(png|tif|json)$")

HISTORICAL_RESULTS = {
    "source": "notebooks/07_Transformer.ipynb (saved outputs; not re-run during this cleanup)",
    "task": "chip-level classification; the positive class is 'chip with NO labelled water' "
            "(labels were inverted relative to their names - see docs/PROJECT_AUDIT.md)",
    "protocol": "5-fold StratifiedGroupKFold by country on 401 chips (45 NaN-corrupted chips dropped)",
    "rows": [
        {"model": "MobileNetV2 + transformer, fine-tuned", "auc": 0.757, "f1": 0.378,
         "precision": 0.29, "recall": 0.55},
        {"model": "MobileNetV2 + transformer, frozen", "auc": 0.723, "f1": 0.19,
         "precision": 0.18, "recall": 0.20},
    ],
}


def _output_dir() -> Path:
    return Path(os.environ.get("SAR_OUTPUT_DIR", PROJECT_ROOT / "outputs")).resolve()


def find_run_dir() -> Path | None:
    explicit = os.environ.get("SAR_FLOOD_RUN_DIR")
    if explicit:
        p = Path(explicit)
        return p if (p / "model.keras").exists() else None
    runs = [p.parent for p in (_output_dir() / "runs").glob("*/metadata.json")
            if (p.parent / "model.keras").exists()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def list_samples() -> list[dict]:
    """Bundled demo inputs: ``samples/<group>/<name>`` (+ optional ``*_LabelHand.tif``)."""
    out = []
    for path in sorted(SAMPLES_DIR.glob("*/*")):
        if path.suffix.lower() not in ALLOWED_EXT or path.name.endswith("_LabelHand.tif"):
            continue
        label = path.with_name(path.name.replace("_S1Hand.tif", "_LabelHand.tif"))
        out.append({"key": f"{path.parent.name}/{path.name}", "name": path.name,
                    "group": path.parent.name, "label": label if label != path and label.exists() else None})
    # samples that come with a correct answer first, then the rest
    return sorted(out, key=lambda s: (s["label"] is None, s["group"] == "legacy", s["name"]))


def collect_evaluations() -> list[dict]:
    evals = []
    for path in sorted((_output_dir() / "eval").glob("*/metrics.json")):
        try:
            data = load_json(path)
        except Exception:
            log.warning("unreadable evaluation file %s", path)
            continue
        data["name"] = path.parent.name
        evals.append(data)
    return evals


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(16),
        MAX_CONTENT_LENGTH=int(os.environ.get("MAX_UPLOAD_MB", "25")) * 1024 * 1024,
        USE_LEGACY=os.environ.get("SAR_FLOOD_LEGACY", "1") != "0",
        WEB_RUNS_DIR=_output_dir() / "web",
        RUN_DIR=None,  # resolved lazily so tests can override
    )
    if test_config:
        app.config.update(test_config)
    state: dict = {"segmenter": None, "segmenter_dir": None, "segmenter_error": None}

    def run_dir() -> Path | None:
        return Path(app.config["RUN_DIR"]) if app.config["RUN_DIR"] else find_run_dir()

    def get_segmenter():
        rd = run_dir()
        if rd is None:
            return None
        if state["segmenter_dir"] != rd:
            try:
                from ..inference.segment import Segmenter  # noqa: PLC0415

                state["segmenter"], state["segmenter_error"] = Segmenter.from_run_dir(rd), None
            except Exception as exc:  # TF missing or broken checkpoint: degrade gracefully
                log.exception("could not load segmenter from %s", rd)
                state["segmenter"], state["segmenter_error"] = None, type(exc).__name__
            state["segmenter_dir"] = rd
        return state["segmenter"]

    def model_status() -> dict:
        rd = run_dir()
        meta = None
        if rd is not None and (rd / "metadata.json").exists():
            meta = load_json(rd / "metadata.json")
        from ..models.legacy import LEGACY_CLASSIFIER  # noqa: PLC0415

        return {"run_name": rd.name if rd else None, "meta": meta,
                "legacy_available": LEGACY_CLASSIFIER.exists() and app.config["USE_LEGACY"],
                "segmenter_error": state["segmenter_error"]}

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def overview():
        return render_template("overview.html", status=model_status(), evals=collect_evaluations(),
                               active="overview")

    @app.get("/analyze")
    def analyze_form():
        return render_template("analyze.html", samples=list_samples(), status=model_status(),
                               max_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024), active="analyze")

    @app.post("/analyze")
    def analyze_submit():
        rid = uuid.uuid4().hex
        work = Path(app.config["WEB_RUNS_DIR"]) / rid
        inputs = work / "input"
        try:
            image_path, label_path = _gather_inputs(inputs)
            result = analyze_file(image_path, work, label_path=label_path,
                                  segmenter=get_segmenter(), use_legacy=app.config["USE_LEGACY"])
        except InputError as exc:
            shutil.rmtree(work, ignore_errors=True)
            flash(str(exc), "error")
            return redirect(url_for("analyze_form"))
        except HTTPException:
            shutil.rmtree(work, ignore_errors=True)
            raise  # e.g. 413 upload too large -> handled below
        except Exception as exc:
            # Keep a failed-run record (no traceback, no paths) so the failure has its own page.
            log.exception("analysis %s failed", rid)
            shutil.rmtree(inputs, ignore_errors=True)
            save_json({"status": "failed", "created": utc_now(), "input": secure_filename(
                           (request.files.get("image").filename if request.files.get("image")
                            else request.form.get("sample", "")) or "unknown"),
                       "error_type": type(exc).__name__, "methods": {}, "images": {}, "warnings": []},
                      work / "result.json")
            return redirect(url_for("result_page", rid=rid))
        log.info("analysis %s done: %s", rid, result["input"])
        return redirect(url_for("result_page", rid=rid))

    def _gather_inputs(inputs: Path) -> tuple[Path, Path | None]:
        sample_key = request.form.get("sample", "").strip()
        upload = request.files.get("image")
        if sample_key:
            match = next((s for s in list_samples() if s["key"] == sample_key), None)
            if match is None:
                raise InputError("Unknown sample.")
            inputs.mkdir(parents=True, exist_ok=True)
            src = SAMPLES_DIR / match["key"]
            image = inputs / src.name
            shutil.copy(src, image)
            label = None
            if match["label"] is not None and request.form.get("use_label", "on") == "on":
                label = inputs / match["label"].name
                shutil.copy(match["label"], label)
            return image, label
        if upload is None or not upload.filename:
            raise InputError("Choose a file to upload or pick a sample.")
        image = _save_upload(upload, inputs, "image")
        label_file = request.files.get("label")
        label = None
        if label_file is not None and label_file.filename:
            label = _save_upload(label_file, inputs, "label")
            if label.suffix.lower() not in {".tif", ".tiff"}:
                raise InputError("Ground truth must be a GeoTIFF (.tif).")
        return image, label

    def _save_upload(storage, folder: Path, stem: str) -> Path:
        name = secure_filename(storage.filename or "")
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise InputError("Unsupported file type. Use a GeoTIFF (.tif) or PNG/JPEG.")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{stem}{ext}"  # never trust the client's file name on disk
        storage.save(path)
        if path.stat().st_size == 0:
            raise InputError("The uploaded file is empty.")
        return path

    @app.get("/results/<rid>")
    def result_page(rid: str):
        if not RESULT_ID.match(rid):
            abort(404)
        try:
            view = build_view(Path(app.config["WEB_RUNS_DIR"]) / rid, rid)
        except ResultNotFound:
            return _missing_result(rid)
        return render_template("result.html", v=view, r=view["r"], rid=rid, active="analyze",
                               status=model_status())

    @app.get("/results/<rid>/report.pdf")
    def result_report(rid: str):
        if not RESULT_ID.match(rid):
            abort(404)
        run = Path(app.config["WEB_RUNS_DIR"]) / rid
        try:
            view = build_view(run, rid)
        except ResultNotFound:
            return _missing_result(rid)
        if view["status"]["code"] == "failed":
            abort(404)
        pdf = run / "report.pdf"
        if not pdf.exists() or pdf.stat().st_mtime < (run / "result.json").stat().st_mtime:
            try:
                from .report import build_report  # noqa: PLC0415

                build_report(view, run, pdf)
            except ImportError:
                flash("PDF reports need the 'reportlab' package: pip install reportlab", "error")
                return redirect(url_for("result_page", rid=rid))
        return send_from_directory(run, "report.pdf", as_attachment=True,
                                   download_name=f"flood-analysis-{rid[:8]}.pdf")

    def _missing_result(rid: str):
        return render_template(
            "error.html", code=404, title="Analysis not found",
            message="No stored analysis matches this ID. Results live in the server's outputs/web "
                    "folder; they may have been deleted, or the link may be from another machine.",
            action_url=url_for("analyze_form"), action_label="Run a new analysis"), 404

    @app.get("/results/<rid>/files/<name>")
    def result_file(rid: str, name: str):
        if not RESULT_ID.match(rid) or not FILE_NAME.match(name):
            abort(404)
        return send_from_directory(Path(app.config["WEB_RUNS_DIR"]) / rid, name)

    @app.get("/evaluation")
    def evaluation():
        return render_template("evaluation.html", evals=collect_evaluations(),
                               history=HISTORICAL_RESULTS, active="evaluation")

    @app.get("/api/health")
    def health():
        s = model_status()
        return jsonify({"status": "ok", "unet_run": s["run_name"],
                        "unet_trained_on": (s["meta"] or {}).get("data_source"),
                        "legacy_available": s["legacy_available"]})

    # ------------------------------------------------------------------ errors
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        flash(f"File too large (limit {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB).", "error")
        return redirect(url_for("analyze_form"))

    @app.errorhandler(404)
    def not_found(_):
        return render_template("error.html", code=404, message="That page or result does not exist."), 404

    @app.errorhandler(500)
    def server_error(_):
        return render_template("error.html", code=500, message="Something went wrong on the server."), 500

    @app.template_filter("pct")
    def pct(value, digits: int = 1):
        if value is None or isinstance(value, Undefined):
            return "–"
        return f"{100 * float(value):.{digits}f}%"

    @app.template_filter("commas")
    def fmt_int_filter(value):
        return fmt_int(None if isinstance(value, Undefined) else value)

    @app.template_filter("area")
    def fmt_area_filter(value):
        return fmt_area(None if isinstance(value, Undefined) else value)

    @app.template_filter("num")
    def num(value, digits: int = 3):
        if value is None or isinstance(value, Undefined):
            return "–"
        return f"{float(value):.{digits}f}"

    return app


def main() -> None:  # python -m sar_flood.web.app
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")))


if __name__ == "__main__":
    main()
