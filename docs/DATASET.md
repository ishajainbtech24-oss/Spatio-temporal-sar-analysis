# Dataset: Sen1Floods11 (hand-labelled subset)

## Source

* Paper: Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). *Sen1Floods11: a
  georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.*
  CVPR Workshops 2020, pp. 210–211.
* Code and download instructions: <https://github.com/cloudtostreet/Sen1Floods11>
* Public bucket: `gs://sen1floods11` (about 14 GB in total; the hand-labelled part used here is
  about 1 GB).
* Licence: mirrors of the dataset list **CC BY 4.0**. Confirm the licence on the original source
  before redistributing any data, and cite the paper above when publishing results.

The original notebooks used a Kaggle-style archive called "Sen1Floods11 Essentials v1.2". It has
the same `data/flood_events/HandLabeled` layout, so it works with this code as well; the official
split files may or may not be included.

## Download

```bash
# requires the Google Cloud SDK (gsutil); no account is needed for this public bucket
mkdir -p data/raw/sen1floods11
gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled data/raw/sen1floods11/v1.1/data/flood_events/HandLabeled
gsutil -m rsync -r gs://sen1floods11/v1.1/splits data/raw/sen1floods11/v1.1/splits
```

If your copy lives elsewhere, point the code at it instead of moving it:

```bash
export SAR_DATA_ROOT=/path/to/v1.1      # Windows PowerShell: $env:SAR_DATA_ROOT="D:\data\v1.1"
sar-flood validate-data
```

## Expected layout

```
<root>/
├── data/flood_events/HandLabeled/
│   ├── S1Hand/      <Event>_<id>_S1Hand.tif      (used)
│   ├── LabelHand/   <Event>_<id>_LabelHand.tif   (used)
│   ├── JRCWaterHand/ ...                          (permanent water; not used yet)
│   └── S2Hand/, S1OtsuLabelHand/ ...              (not used)
└── splits/flood_handlabeled/
    └── flood_train_data.csv, flood_valid_data.csv, flood_test_data.csv (+ flood_bolivia_data.csv)
```

The split files are found by searching under `<root>`, so both `v1.1/splits/...` and an archive
that puts them elsewhere work. If they are missing, the code falls back to a group split (below).

## File contents

| File | Bands | Type | Values |
|---|---|---|---|
| `*_S1Hand.tif` | 1 = VV, 2 = VH | float32 | Sentinel-1 GRD backscatter in dB; **NaN = no data** |
| `*_LabelHand.tif` | 1 | int16 | **−1** no data / not labelled, **0** not water, **1** water |

All chips are 512 × 512 pixels (≈10 m) in EPSG:4326. There are 446 image/label pairs from 11 flood
events (Bolivia, Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri Lanka,
USA); counts per event are in notebook 01.

**`LabelHand` marks all water, not only flood water.** Rivers, lakes and reservoirs are labelled
1 too. Models trained on it are surface-water segmenters. A flood map needs a permanent-water
reference (for example `JRCWaterHand`) to subtract, which this project does not do yet.

## Validation

`sar-flood validate-data` checks every pair and writes `outputs/data_validation.json`:

* image/label pairing (unpaired files are listed),
* readability (corrupted files), band count, shape agreement,
* label values outside {−1, 0, 1},
* no-data fraction per chip (NaN and ±inf), chips with no valid label, chips without water,
* identical CRS and geotransform for image and label,
* whether official split files were found.

## Splits and leakage

* `split_strategy: auto` (default) uses the official CSVs when present, otherwise a **group split**:
  whole flood events are assigned to train/val/test with a fixed seed, so no event appears in two
  splits. Neighbouring chips of one event are highly correlated; a random chip-level split would
  leak.
* The chosen split is saved to `outputs/runs/<run>/splits.json`. `evaluate` re-uses that file, so
  a model is always scored on the split it never saw.
* Training reads only train and val. The autoencoder sees only training images. Checkpoint
  choice, early stopping and the decision threshold use only the validation split.

## Handling of special values

| Situation | Handling |
|---|---|
| NaN / ±inf in any band | pixel marked invalid; input set to `fill_value`; label set to −1 (ignored) |
| Label −1 | ignored in loss and metrics |
| Label values other than −1/0/1 | treated as −1 and reported by `validate-data` |
| Chip with no valid labelled pixel | skipped and recorded in run metadata |
| Unreadable file | skipped and recorded |

## Synthetic data

`sar-flood make-synthetic` writes a small dataset in exactly this layout (with split CSVs and a
`SYNTHETIC_DATA_README.txt` marker). Water is drawn as low, speckled backscatter (≈ −21 dB VV)
against brighter land, with no-data wedges and unlabelled pixels. It exists so the code can be
tested without the real data. Results on it are trivially high and say nothing about real SAR.
