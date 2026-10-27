# Model card

## 1. U-Net surface-water segmenter (`sar-flood train`)

| | |
|---|---|
| Task | Per-pixel binary segmentation: open surface water vs. not water |
| Input | Sentinel-1 GRD chip, 2 bands (VV, VH) in dB, any size (padded to a multiple of 2^depth) |
| Output | Water probability per pixel; binary mask at a threshold chosen on the validation split |
| Architecture | U-Net, depth 4, 16 base filters, 1.95 M parameters; encoder initialised from a convolutional autoencoder trained on training-split images |
| Loss | Masked BCE + soft Dice (pixels labelled −1 or with no-data input are ignored) |
| Training data | Sen1Floods11 hand-labelled chips, official split (or group-by-event split) |
| Framework | TensorFlow / Keras 3, no custom layers |

### Status of trained weights

**No checkpoint trained on real Sen1Floods11 data is distributed with this repository.** The
dataset could not be downloaded in the environment where this code was written, so training was
only exercised on synthetic chips. Train your own with:

```bash
sar-flood train --config configs/default.yaml        # CPU works; a GPU is much faster
sar-flood evaluate --config configs/default.yaml --run-dir outputs/runs/unet_sen1floods11 --split test
```

Each run directory is self-describing: `config.yaml`, `splits.json`, `metadata.json` (data
source, threshold, validation metrics, library versions, timings), `history.csv`, `model.keras`
and `autoencoder.keras`. Fill in section 1.1 from `metrics.json` once a real run exists.

### 1.1 Evaluation

*To be filled in from `outputs/eval/<run>_test/metrics.json` after training on Sen1Floods11.*
Report micro IoU, Dice, precision and recall for the U-Net **and** both baselines on the same
split, the split strategy, the threshold, and the hardware / training time from `metadata.json`.

### Intended use

Research and teaching: comparing learned and classical SAR water detection on a public
benchmark; a starting point for flood-mapping experiments.

### Out of scope

* Operational flood response or any decision affecting people or property. The model has not
  been independently validated, and Sen1Floods11 covers only 11 events.
* Flood extent on its own: labels include permanent water, so outputs need a permanent-water
  layer subtracted to become flood maps.
* Other sensors, bands, polarisations or processing levels than Sentinel-1 GRD VV/VH in dB.
* Uncalibrated images (PNG/JPEG screenshots).

### Known limitations and risks

* **Small, clustered data:** 446 chips from 11 events; performance varies strongly by event.
* **Typical SAR failure modes:** smooth surfaces that look like water (tarmac, sand, dry
  river beds, radar shadow) cause false alarms; flooded vegetation and urban flooding (double
  bounce, brighter than water) cause misses; wind-roughened water can look like land.
* **Label noise:** hand labels are human interpretations and can disagree with what the
  Sentinel-1 image shows (for example where optical imagery used for labelling was acquired at
  a different time or was cloudy). Some "errors" against the reference are reference errors.
* **Threshold** is tuned for IoU on the validation split, not for a specific false-alarm rate.

## 2. Legacy scene classifier (`models/legacy/simple_transformer_pretrained_final.keras`)

| | |
|---|---|
| Origin | Notebook 07 final cell: ImageNet MobileNetV2 (frozen) → 16 tokens → 1 transformer block → sigmoid |
| Input | 128 × 128 VV only, median filter 3, per-image min-max |
| Output, correctly interpreted | Probability that the chip contains **no** labelled water (see `PROJECT_AUDIT.md` §3.1 — the notebooks and old app called this "flood probability") |
| Threshold | 0.13, as printed by notebook 07 for this exact model |
| Training data | All 401 non-corrupted hand-labelled chips (no held-out set) |
| Cross-validated estimate (historical) | Frozen variant: out-of-fold ROC AUC 0.723, F1 0.19 for the "no water" class, 5-fold group CV by country |
| Loading | `sar_flood.models.legacy.load_legacy_classifier()` (weights-only; the saved graph segfaults on Python 3.13) |

Kept for transparency and comparison. It should not be used to make claims about flooding.
