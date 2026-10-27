# Project audit

Audit of `main` at commit `18b0647` ("Created using Colab"), carried out on 2026-10-27 before the
`improve/portfolio-ready` changes. Every finding below was checked against the code, the saved
notebook outputs, or by running the code. Nothing was inferred from the project description alone.

## 1. What the repository contained

| Path | State |
|---|---|
| `README.md` | Empty (0 bytes). |
| `src/autoencoder.py`, `src/preprocessing.py`, `src/train.py` | Empty (0 bytes). No reusable pipeline code existed outside the notebooks. |
| `notebooks/01`–`07` | The actual pipeline. All paths hard-coded to Google Drive / Colab (`/content/drive/MyDrive/...`). |
| `app/app.py`, `app/utils/predict.py`, `app/utils/terrain.py` | Flask dashboard; scene-level "Flood / No Flood" prediction plus a K-means "terrain" heuristic. |
| `app/templates/*.html` | Four pages (Tailwind via CDN). `app/static/style.css` was empty. |
| `app/model/*.keras` | Three Keras 3 models (42 KB, 668 KB, 10.4 MB). |
| `app/uploads/*` | Three 8-bit images of unknown origin. |
| `requirements.txt` | Unpinned; missing `rasterio`, which the notebooks require. |

Nothing was ever trained or evaluated outside Colab notebooks, and there were no tests or CI.

## 2. What the pipeline actually did

1. **Data** (nb 01–02): Sen1Floods11 hand-labelled chips (446 image/label pairs, 512×512, 11 flood
   events). Only **band 1 (VV)** was used; VH was ignored.
2. **Preprocessing** (nb 02): median filter 3×3 → per-image min-max → resize to 128×128.
3. **Labels** (nb 02): each *mask* was collapsed into one *chip-level* label. The project never
   did per-pixel flood segmentation, even though the dataset is a segmentation dataset.
4. **Autoencoder** (nb 03): small conv autoencoder (12 K params) on VV; latent 32×32×16.
5. **Random Forest / XGBoost / CNN** (nb 04, 06) on the 16,384-dim flattened latent features.
6. **Transformer** (nb 07): the version that was deployed is *not* autoencoder-based. It is an
   ImageNet MobileNetV2 (frozen) → 16 tokens → one transformer block → sigmoid, evaluated with
   5-fold group-aware cross-validation, then retrained on **all** data for deployment.
7. **Dashboard**: uploaded PNG/JPEG → that classifier → "Flood"/"No Flood".

Nothing in the project is temporal. Each hand-labelled chip is a single post-event acquisition, so
"spatio-temporal" in the title is not supported by the data or the models.

## 3. Critical findings

### 3.1 The class labels are inverted (critical)

`notebooks/02_preprocessing.ipynb`, cells 17–18:

```python
binary_labels.append("Flood" if np.sum(mask > 0) > 0 else "NoFlood")
encoded_labels = LabelEncoder().fit_transform(binary_labels)   # classes_ = ['Flood', 'NoFlood']
```

`LabelEncoder` sorts alphabetically, so **Flood → 0 and NoFlood → 1**. Notebooks 04, 06 and 07 all
treat `1` as "Flood" (`target_names=["No Flood", "Flood"]`, "probability of Flood").

Consequences:

* The "Flood" class used everywhere downstream is really "chip that contains **no** water pixel".
  It is the 12–13 % minority (45 of 356 training chips) — consistent with Sen1Floods11, whose
  hand-labelled chips were selected to contain water.
* Notebook 07's own sanity check agrees: the "Flood" chips' darkest 16×16 block was *brighter*
  (0.411) than the "No Flood" chips' (0.369, p = 0.011). Water is dark in SAR, so the class called
  "Flood" is the drier one.
* The deployed dashboard therefore reported "Flood" when the model believed the chip was dry.

The legacy model is kept and now reported under its correct meaning: *probability the chip
contains no labelled water* (`src/sar_flood/models/legacy.py`).

### 3.2 One NaN pixel destroyed a whole chip (critical)

S1Hand chips use NaN for no data. `normalize()` used `np.min`/`np.max`, which return NaN if any
pixel is NaN, so the whole chip became NaN. Evidence in the saved outputs:

* nb 03: 573,440 NaN values in `X_train` = exactly 35 full 128×128 chips.
* nb 06: the CNN trained on the un-cleaned arrays logged `loss: nan` for all 10 epochs and
  predicted the majority class only.
* nb 07: after `nan_to_num`, 36 train + 9 test chips were "solid black". They were described as
  sensor dropouts and deleted (7 of the 45 were minority-class chips). They were in fact
  ordinary chips with partial no-data.

Fixed by NaN-aware preprocessing (`preprocessing/transforms.py`) with a regression test.

### 3.3 Data leakage and hidden state

* **nb 04 evaluated on its training data.** `X_test` is loaded from the same
  `latent_features.npy` as `X_train`, and the cells that defined `X_train_rf`/`y_test_rf` are
  missing (the notebook only runs with hidden kernel state). Its 0.875 accuracy is the majority
  class rate; precision/recall/F1 for the positive class were 0.
* **nb 03 and nb 06 use the test set as validation data** during training (`validation_data=(X_test, …)`).
* **Train/test split in nb 02 is random, not by event**, so near-duplicate neighbouring chips from
  the same flood can sit on both sides. nb 07 recognised this and switched to
  `StratifiedGroupKFold` — the most trustworthy numbers in the project come from there.
* **The deployed model was trained on all 401 chips** (nb 07, final cell). Any evaluation of it on
  Sen1Floods11 hand-labelled chips is in-sample.
* nb 07 tunes the decision threshold on an inner validation split (good), but the deployment
  threshold printed was **0.13**, while `app/utils/predict.py` used **0.20** with no recorded source.

### 3.4 The deployed model could not be loaded on current Python (blocking)

`simple_transformer_pretrained_final.keras` contains a `Lambda` layer whose bytecode was pickled
under Python 3.12. Loading it on Python 3.13 **crashes the interpreter (segfault)** — verified with
TensorFlow 2.21 / Keras 3.15 on Python 3.13.17. The dashboard could not start a prediction.

Fix: the architecture is rebuilt in code with the `Lambda` replaced by an equivalent `Rescaling`
layer, and only the weights are loaded. Verified against the original model running under
Python 3.12.16 on 7 inputs (4 random, 3 bundled images): **maximum absolute difference 1.25 × 10⁻⁶**.

### 3.5 Training/inference preprocessing mismatch

* Training: GeoTIFF float dB → median → per-image min-max → `skimage.resize`.
* nb 05 inference: 8-bit image → `cv2.resize` → `/255` (no filter, different scaling).
* Dashboard: 8-bit PNG/JPEG → the training transform. The input domain itself differs: training
  data were calibrated backscatter in dB, uploads were arbitrary 8-bit pictures.

Per-image min-max also removes absolute brightness — the main physical cue for water — so a dry
chip and a wet chip are stretched to the same 0–1 range.

### 3.6 The dashboard displayed fabricated content

Hard-coded values presented as live data: "Data Processed 1.2 PB", "Latency < 45 ms",
"Active Alerts 12 Regions", "3 Floods Detected, Delta Sector 4, 9, 12", "Thermal Anomaly Mount
Merapi +12.4 %", "RADARSAT-2 pass … 0.3 m", an "Active Processing Queue" at 74 %, sensor and
polarisation selectors that did nothing, and a "Supports Sentinel-1 / TerraSAR-X / ICEYE bundles"
claim. `/map` rendered a template that did not exist (`map.html`), producing a 500 error. Errors
were returned as raw HTML containing the exception text; `debug=True` was on.

The terrain classifier (`terrain.py`) labelled K-means clusters as "Water / Vegetation / Urban /
Bare Soil" by brightness and texture with no labelled data or validation; its percentages were
shown next to the model output.

## 4. What worked

* Data discovery and pairing logic (nb 02) and the dataset statistics in nb 01.
* The group-aware cross-validation design of nb 07, including threshold tuning on an inner
  validation split and a fine-tune-vs-frozen ablation — sound methodology, applied to the wrong
  labels.
* All three saved models load on Python 3.12, and the two autoencoder-pipeline models
  (`encoder.keras`, `simple_transformer_final.keras`) also load on 3.13.
* `predict.py` already documented why preprocessing must match training.

## 5. Historical results (as saved in the notebooks; not reproduced)

Positive class = chip with no labelled water (see 3.1). From nb 07, 5-fold `StratifiedGroupKFold`
by country, 401 chips:

| Model | Out-of-fold ROC AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| MobileNetV2 + transformer, fine-tuned | 0.757 | 0.38 | 0.29 | 0.55 |
| MobileNetV2 + transformer, frozen (the deployed variant) | 0.723 | 0.19 | 0.18 | 0.20 |

Per-fold AUC ranged 0.58–0.98 (fine-tuned). nb 06 accuracies (0.88–0.90) equal the majority-class
rate and should not be quoted.

## 6. Plan that was implemented

| Priority | Change |
|---|---|
| P0 | Correct label semantics for the legacy model; stop presenting it as a flood detector. |
| P0 | NaN-aware, fixed-range preprocessing shared by training, evaluation and inference. |
| P0 | Make the legacy model load on any Python (weights-only loader, verified numerically). |
| P0 | Remove fabricated dashboard content; every displayed number comes from code or files. |
| P1 | Reframe the task as what the labels support: per-pixel surface-water **segmentation** with VV+VH. Keep the autoencoder idea as self-supervised pre-training of a U-Net encoder. |
| P1 | Leakage-safe splits (official CSVs, else group-by-event), val-only checkpoint/threshold selection, test touched only by `evaluate`. |
| P1 | Classical baselines (Otsu, fixed −15 dB) and ignore-aware IoU/Dice/precision/recall. |
| P1 | Installable package, CLI, configs, tests, CI. |
| P2 | Documentation, model card, honest README, portfolio notes. |

## 7. Not done / blocked

* **Sen1Floods11 could not be downloaded in the environment used for this work** (outbound access
  to Google Cloud Storage was blocked), so **no model was trained or evaluated on real data**.
  All training and evaluation code was exercised on synthetic chips in the same file format.
* No screenshots of the dashboard were captured (no headless browser could be installed).
* The original notebooks were left unchanged as a historical record; `notebooks/README.md`
  lists their known issues.
