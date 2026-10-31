# Installation

Requires Python 3.10–3.13 and about 2 GB of disk for the dependencies (TensorFlow is most of it).
Tested here on Linux with Python 3.13.17 (TensorFlow 2.21) and Python 3.12.16.

## Linux / macOS

```bash
git clone https://github.com/ishajainbtech24-oss/Spatio-temporal-sar-analysis.git
cd Spatio-temporal-sar-analysis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[ml,dev]"      # or: pip install -r requirements.txt && pip install -e .
```

## Windows (PowerShell)

```powershell
git clone https://github.com/ishajainbtech24-oss/Spatio-temporal-sar-analysis.git
cd Spatio-temporal-sar-analysis
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[ml,dev]"
```

`rasterio` ships Windows wheels on PyPI for current Python versions; if `pip` tries to build it
from source, upgrade pip or use Python 3.12. If PowerShell blocks the activate script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## Lightweight install (no TensorFlow)

```bash
pip install -e ".[dev]"
```

Gives data validation, preprocessing, the threshold baselines, evaluation of baselines, the
dashboard (baselines only) and 91 of the 103 tests. Model tests are skipped automatically.

## GPU

The `ml` extra installs `tensorflow-cpu` on Linux/Windows. For an NVIDIA GPU on Linux, replace it:

```bash
pip uninstall -y tensorflow-cpu
pip install "tensorflow[and-cuda]>=2.16"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

TensorFlow no longer supports native Windows GPUs; use WSL2. On Apple silicon,
`pip install tensorflow-metal` adds GPU support. `train.device: auto|cpu|gpu` in the config
controls what is used; `gpu` fails loudly if no GPU is visible.

## Check the install

```bash
pytest                                   # ~20 s on CPU, no dataset needed
sar-flood --help
```

## Configuration

| Setting | Where |
|---|---|
| Dataset location | `data.root` in `configs/*.yaml`, or `SAR_DATA_ROOT` |
| Output folder | `output_dir`, or `SAR_OUTPUT_DIR` |
| Hyper-parameters, seed, device | `configs/*.yaml` (copy `default.yaml` and edit) |
| Dashboard model, upload limit, secret key | environment variables, see `.env.example` |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Expected Sen1Floods11 folders not found` | `data.root` / `SAR_DATA_ROOT` must point at the folder that *contains* `data/flood_events/HandLabeled`. |
| `No trained checkpoint in ...` | Train first, or pass the right `--run-dir` / `SAR_FLOOD_RUN_DIR`. |
| Segmentation fault when loading the old `.keras` file directly | Use `sar_flood.models.legacy.load_legacy_classifier()`; the saved Lambda layer only loads on Python 3.12. |
| Dashboard says "uncalibrated input" | You uploaded an 8-bit image. Use an `*_S1Hand.tif` GeoTIFF for segmentation. |
| Out of memory during training | Lower `train.batch_size` or `train.crop_size`, or set `train.max_train_samples`. |
| Very slow training on CPU | Expected for the full config; try `--epochs 5 --ae-epochs 2` first, or use a GPU / Colab. |
| `ModuleNotFoundError: sar_flood` | Run `pip install -e .` inside the activated environment. |
