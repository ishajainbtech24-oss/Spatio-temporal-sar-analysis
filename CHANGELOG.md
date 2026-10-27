# Changelog

## 0.4.0 — flood verdict and land-cover / plantation screening

### Added
- **Flood prediction verdict** per method ("Flood-affected" / "No flood water detected"), using notebook 02's definition (a chip is "Flood" when it contains water) with a 1 % minimum water share, plus a check against the original scene classifier.
- **Land cover & vegetation analysis** (`inference/landcover.py`): rule-based classes from VV/VH backscatter (water from the method's mask; built-up VV ≥ −6 dB; vegetation VH ≥ −19 dB; bare/sparse otherwise), a second doughnut chart, a land-cover map and the rules shown on the page.
- **Plantation screening (indicative)**: "Potentially suitable — survey recommended" / "Not suitable" / "Inconclusive" from the land-cover shares, with the criteria and caveats displayed. Not a trained model; the notebooks contain no vegetation or plantation model and Sen1Floods11 has no land-cover labels.
- Verdicts and the land-cover table in the PDF report and `sar-flood predict` output; 11 new tests.

## 0.3.0 — results dashboard redesign

### Added
- Results page rebuilt as an analysis dashboard: header with run ID, status, timestamp, back and report actions; method switcher; KPI strip; land-classification doughnut with keyboard-accessible tooltips; prediction summary with split bar, pixel counts, ground-truth share and IoU/Dice/precision/recall; imagery comparison (SAR, mask, overlay with opacity slider, ground truth) with full-size viewer and missing-image states; spatial water-share grid; map that follows the selected method; collapsible technical details.
- `result.json` now stores per-method pixel counts, inference time, a spatial grid, transparent water layers, pixel size/area, model info and preprocessing steps. Older results are re-counted from their saved masks.
- PDF report (`/results/<id>/report.pdf`, reportlab) built from the same view model as the page.
- Failed analyses get their own page; missing result IDs get a specific 404.
- 12 new tests (counts vs. GeoTIFF masks, page values, report, PNG/no-map, failures, old results, zero-water / zero-valid edge cases).

### Changed
- Dark theme across the dashboard; one colour code everywhere: coral = water, teal = not water, slate = no data (PNG quick-looks included).
- Brand renamed to "SAR Flood Intelligence"; `reportlab` added as a dependency.

## 0.2.0 — portfolio-ready refactor (branch `improve/portfolio-ready`)

### Fixed
- Legacy classifier output was reported with inverted meaning ("Flood" was really "chip without water"); now reported correctly.
- NaN no-data pixels no longer turn whole chips into NaN (NaN-aware filtering and scaling).
- Legacy model segfaulted on Python 3.13 (pickled `Lambda`); now rebuilt in code and loaded from weights (max output difference vs. original 1.25e-6).
- Training/inference preprocessing mismatch: one shared `preprocess_chip` for every stage.
- Dashboard: removed fabricated statistics, alerts and queues; fixed missing `map.html` (500); no tracebacks or paths shown to users; `debug=True` removed.

### Added
- `sar_flood` package with CLI (`validate-data`, `make-synthetic`, `train`, `evaluate`, `predict`, `serve`) and YAML configs.
- Per-pixel water segmentation: autoencoder-pretrained U-Net on VV+VH, masked BCE+Dice loss.
- Leakage-safe splits (official CSVs or group-by-event), val-only checkpoint and threshold selection, saved `splits.json`.
- Otsu and fixed −15 dB baselines; ignore-aware IoU / Dice / precision / recall (micro and per-chip); JSON + CSV outputs.
- GeoTIFF mask export, quick-look PNGs, agreement maps, Leaflet map view for georeferenced inputs.
- Synthetic Sen1Floods11-format data for tests; 71 tests; GitHub Actions (lint, core without TF, full ML + CLI smoke run).
- Documentation: README, audit, architecture, dataset, model card, installation, portfolio notes.

### Changed / moved
- `app/model/*.keras` → `models/legacy/`; `app/uploads/*` → `samples/legacy/`.
- Flask app moved into `src/sar_flood/web/`; `python app.py` and `sar-flood serve` start it.

### Removed
- Empty `src/autoencoder.py`, `src/preprocessing.py`, `src/train.py`, `app/static/style.css`.
- Old `app/` templates and the unvalidated K-means "terrain classification" (recoverable from git history at `18b0647`).

## 0.1.0 — original Colab project (`main` @ 18b0647)
Notebooks 01–07 and the first Flask dashboard.
