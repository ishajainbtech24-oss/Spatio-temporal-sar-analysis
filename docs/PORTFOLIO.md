# Portfolio and interview notes

Everything here is limited to what the repository can back up today. When you train on the real
data, replace the bracketed placeholders with numbers from `outputs/eval/*/metrics.json`, and
never quote the synthetic smoke-run metrics as results.

## One-line description

Sentinel-1 SAR surface-water segmentation on Sen1Floods11: autoencoder-pretrained U-Net vs.
classical thresholds, with a leakage-safe evaluation pipeline, 83 automated tests and a Flask
dashboard.

## Resume bullets

Pick two or three. Versions A–C are true today; D becomes usable after a real training run.

- **A (audit + correctness).** Audited an end-to-end SAR flood-detection project and found an
  inverted label encoding, NaN propagation that corrupted 45 of 446 chips, and train/test leakage;
  fixed the pipeline and documented each issue with evidence from saved notebook outputs.
- **B (engineering).** Refactored Colab notebooks into an installable Python package with a CLI
  (validate → train → evaluate → predict), YAML configs, run metadata and 83 pytest tests
  running in GitHub Actions without the dataset.
- **C (ML + geospatial).** Built a Sentinel-1 VV/VH water-segmentation pipeline: NaN-aware
  despeckling, autoencoder-pretrained U-Net with masked BCE+Dice loss, Otsu/−15 dB baselines,
  ignore-aware IoU/Dice, GeoTIFF mask export and a Flask/Leaflet dashboard.
- **C′ (model recovery, short).** Restored a Keras model that crashed on Python 3.13 by rebuilding
  its graph and loading weights only, matching the original outputs to within 1.3 × 10⁻⁶.
- **D (after training — fill in).** Trained an autoencoder-initialised U-Net on Sen1Floods11
  reaching [X] test IoU vs. [Y] for Otsu thresholding on the official split.

## Explaining the architecture (about 90 seconds)

1. *Problem:* map open water in Sentinel-1 radar chips. Water is dark in SAR because it reflects
   the pulse away; noise and look-alike surfaces make simple thresholds error-prone.
2. *Data:* Sen1Floods11, 446 hand-labelled 512×512 chips from 11 flood events, two polarisations,
   labels −1/0/1. Split by official files or by whole event, because chips from the same event
   are correlated.
3. *Preprocessing:* one function used everywhere — validity mask from NaNs, 3×3 median filter that
   ignores NaNs, fixed dB range scaling (keeps absolute brightness, unlike per-image min-max).
4. *Model:* a convolutional autoencoder learns to reconstruct training chips without labels; its
   encoder initialises a U-Net. Loss is BCE + Dice over valid pixels only.
5. *Selection & evaluation:* best epoch and decision threshold picked on validation; test scored
   once with IoU/Dice/precision/recall next to Otsu and a −15 dB threshold.
6. *Serving:* the same pipeline powers the CLI and the dashboard, which writes a georeferenced mask
   and shows agreement with ground truth.

## Likely interview questions (with short answers)

**SAR basics**
- *Why does water look dark in SAR?* Specular reflection: smooth water sends the pulse away from
  the side-looking sensor, so little energy returns (low σ⁰, ≈ −20 dB VV).
- *VV vs VH?* Co- vs cross-polarised. VV is the stronger water/land contrast; VH responds to
  volume scattering (vegetation) and helps separate some look-alikes.
- *What is speckle and why median filtering?* Coherent interference noise, multiplicative. A
  median filter removes outliers while keeping edges better than a mean filter; Lee/Refined Lee
  filters are the SAR-specific alternatives.
- *Where does SAR water detection fail?* Radar shadow, smooth tarmac/sand, wind-roughened water,
  flooded vegetation and urban floods (double bounce makes them bright).
- *Why dB?* Backscatter spans orders of magnitude; the log scale makes distributions closer to
  Gaussian and thresholds intuitive.

**Preprocessing and data**
- *Why not per-image min-max?* It removes the absolute level that distinguishes water; a dry chip
  is stretched to look like a wet one. Also, with NaNs it returns NaN for the whole chip.
- *How do you handle no-data?* Validity mask; fill input with a constant; label −1; ignore in loss
  and metrics.
- *How can leakage happen here?* Random chip splits put neighbouring chips of one event in train
  and test; tuning thresholds/epochs on test; self-supervised pre-training on test images;
  training the final model on all data and then evaluating it.

**Autoencoders and segmentation**
- *Why an autoencoder without skip connections for pre-training?* With skips it can copy the input
  and learn trivial features.
- *What does the autoencoder buy you?* Possibly better initialisation from unlabelled pixels; it
  must be shown with an ablation (`use_autoencoder_init: false`), not assumed.
- *Why U-Net?* Skip connections recover spatial detail lost by pooling — needed for thin rivers
  and flood edges.
- *Why BCE + Dice?* Dice directly optimises overlap and copes with class imbalance; BCE gives
  stable per-pixel gradients.

**Transformers (from the original work)**
- *What did the transformer do in the old model?* Treated the 4×4 MobileNetV2 feature map as 16
  tokens with learned position embeddings and self-attention, then pooled to one chip score.
- *Why might that underperform a U-Net here?* It produces one label per chip; the task is spatial.
  16 tokens from 128×128 input also discard fine detail.

**Metrics**
- *Why not accuracy?* With ~10–20 % water, predicting "no water" everywhere gets 80–90 %.
- *Micro vs. per-chip IoU?* Micro pools pixels across chips (large-water chips dominate); per-chip
  averages chips. Chips where both prediction and truth are empty have undefined IoU — exclude and
  count them rather than scoring 0 or 1.
- *How did you choose the threshold?* Sweep on validation pixels for maximum IoU; saved with the
  checkpoint; never tuned on test.

**Debugging stories**
- *Tell me about a bug you found.* The label inversion: `LabelEncoder` sorts alphabetically, so
  "Flood" became 0; downstream code assumed 1 = flood. Evidence: the "flood" class had *brighter*
  dark patches, which contradicts SAR physics.
- *Why did the CNN loss go to NaN?* NaN inputs from the min-max bug reached the network.

## Five-minute demo

1. Open the README status note and the audit summary: show you found and fixed real problems
   rather than polishing numbers.
2. Run `pytest` (20 s) to show the tests, then the quick-start smoke run.
3. `sar-flood serve`, open *Analyse a chip*, pick the synthetic sample with ground truth: walk
   through VV/VH/false colour, the U-Net vs. Otsu masks, the agreement map and the map view.
   Point out the "trained on synthetic data" tag — the system is honest about what it is.
4. Upload one of the legacy PNGs to show the "uncalibrated input" handling.
5. Open *Evaluation* to show metrics come only from evaluation files, with history separated.
6. If you have trained on real data, show `metrics.json` and the U-Net vs. baseline gap.

## Remaining limitations and realistic next steps

| Limitation | Next step |
|---|---|
| No real-data training yet | Run `configs/default.yaml` on Colab/GPU (about 1 GB of data); report test and Bolivia results. |
| Water ≠ flood | Subtract `JRCWaterHand` and report flood-only IoU. |
| Single date | Add a pre-event Sentinel-1 acquisition for change detection. |
| Autoencoder benefit unproven | Run the ablation with `use_autoencoder_init: false`, several seeds. |
| No screenshots | Capture the dashboard locally and add them to `docs/images/`. |
| No licence | Choose one (for example MIT) and add `LICENSE`. |
