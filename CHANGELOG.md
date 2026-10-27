# Changelog

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
