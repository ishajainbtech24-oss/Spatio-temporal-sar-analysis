"""Training: optional autoencoder pre-training, then U-Net segmentation.

Leakage rules enforced here:

* the autoencoder is pre-trained on **training-split images only** (no val/test pixels);
* checkpoint selection (best ``val_iou``), early stopping and threshold tuning use the
  **validation split only**;
* the test split is never loaded by this module - it is only touched by ``evaluate``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml

from ..config import Config
from ..data.loader import SplitArrays, load_split
from ..data.sen1floods11 import discover_samples, make_splits
from ..evaluation.metrics import MetricAccumulator, best_threshold
from ..inference.segment import CHECKPOINT_NAME, METADATA_NAME, predict_proba
from ..preprocessing.transforms import random_crop_and_flip
from ..utils import configure_device, environment_info, get_logger, save_json, set_seed, utc_now

log = get_logger(__name__)


def _crop_dataset(split: SplitArrays, cfg: Config, target: str, seed: int):
    """keras.utils.PyDataset yielding random crops (new crops every epoch)."""
    import keras  # noqa: PLC0415

    crop, bs = cfg.train.crop_size, cfg.train.batch_size

    class Crops(keras.utils.PyDataset):
        def __init__(self):
            super().__init__()
            self.rng = np.random.default_rng(seed)
            self.n = len(split.ids)

        def __len__(self):
            return max(1, int(np.ceil(self.n / bs)))

        def on_epoch_end(self):
            pass

        def __getitem__(self, idx):
            xs, ys = [], []
            for _ in range(bs):
                i = int(self.rng.integers(0, self.n))
                if target == "label":
                    x, y = random_crop_and_flip(split.images[i], split.labels[i], crop, self.rng,
                                                cfg.train.augment)
                    ys.append(y.astype(np.float32)[..., None])
                else:  # autoencoder: reconstruct input, validity as extra target channel
                    valid = split.valids[i].astype(np.int8)
                    x, v = random_crop_and_flip(split.images[i], valid, crop, self.rng, cfg.train.augment)
                    v = np.clip(v, 0, 1).astype(np.float32)[..., None]
                    ys.append(np.concatenate([x, v], axis=-1))
                xs.append(x)
            return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)

    return Crops()


def _full_arrays(split: SplitArrays, multiple: int):
    """Stack equally sized validation chips (Sen1Floods11 chips are all 512x512)."""
    shapes = {im.shape for im in split.images}
    if len(shapes) != 1:
        return None
    h, w, _ = next(iter(shapes))
    if h % multiple or w % multiple:
        return None
    return np.stack(split.images), np.stack(split.labels).astype(np.float32)[..., None]


def evaluate_split(model, split: SplitArrays, threshold: float, multiple: int) -> tuple[dict, np.ndarray, np.ndarray]:
    acc = MetricAccumulator()
    probs, targets = [], []
    for cid, img, lab, valid in zip(split.ids, split.images, split.labels, split.valids, strict=True):
        p = predict_proba(model, img, multiple)
        p[~valid] = 0.0
        acc.add(cid, p >= threshold, lab)
        probs.append(p.ravel())
        targets.append(lab.ravel())
    return acc.summary(), np.concatenate(probs), np.concatenate(targets)


def train(cfg: Config, data_source: str = "sen1floods11") -> Path:
    device = configure_device(cfg.train.device)
    set_seed(cfg.train.seed)
    import keras  # noqa: PLC0415

    from ..models.unet import (  # noqa: PLC0415
        MaskedIoU,
        build_autoencoder,
        build_unet,
        masked_bce_dice,
        masked_mse,
        transfer_encoder_weights,
    )

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)

    samples = discover_samples(cfg.data_root)
    splits, strategy = make_splits(samples, cfg.data_root, cfg.data.split_strategy,
                                   cfg.data.val_fraction, cfg.data.test_fraction, cfg.train.seed)
    save_json({"strategy": strategy, **{k: [s.chip_id for s in v] for k, v in splits.items()}},
              run_dir / "splits.json")
    log.info("split strategy=%s train=%d val=%d test=%d (test not loaded during training)",
             strategy, len(splits["train"]), len(splits["val"]), len(splits["test"]))

    train_arr = load_split(splits["train"], cfg, max_samples=cfg.train.max_train_samples)
    val_arr = load_split(splits["val"], cfg)
    in_ch = len(cfg.data.bands)
    multiple = 2 ** cfg.model.depth
    t0 = time.time()
    history: dict = {}

    # ---------------------------------------------------------------- 1) autoencoder
    ae = None
    if cfg.train.use_autoencoder_init and cfg.train.ae_epochs > 0:
        log.info("pre-training autoencoder for %d epochs on %d training chips",
                 cfg.train.ae_epochs, len(train_arr.ids))
        ae = build_autoencoder(in_ch, cfg.model.base_filters, cfg.model.depth)
        ae.compile(optimizer=keras.optimizers.Adam(cfg.train.learning_rate), loss=masked_mse)
        h = ae.fit(_crop_dataset(train_arr, cfg, "reconstruct", cfg.train.seed),
                   epochs=cfg.train.ae_epochs, verbose=2)
        history["autoencoder_loss"] = [float(v) for v in h.history["loss"]]
        ae.save(run_dir / "autoencoder.keras")

    # ---------------------------------------------------------------- 2) U-Net
    unet = build_unet(in_ch, cfg.model.base_filters, cfg.model.depth, cfg.model.dropout)
    copied = transfer_encoder_weights(ae, unet) if ae is not None else []
    log.info("U-Net encoder initialised from autoencoder: %d layers", len(copied))
    unet.compile(optimizer=keras.optimizers.Adam(cfg.train.learning_rate), loss=masked_bce_dice,
                 metrics=[MaskedIoU(name="iou")])
    ckpt_path = run_dir / CHECKPOINT_NAME
    full_val = _full_arrays(val_arr, multiple)
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_iou", mode="max", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_iou", mode="max",
                                      patience=cfg.train.early_stopping_patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
    ]
    if full_val is None:  # unequal or odd-sized val chips: validate on crops instead
        val_data = _crop_dataset(val_arr, cfg, "label", cfg.train.seed + 1)
    else:
        val_data = full_val
    h = unet.fit(_crop_dataset(train_arr, cfg, "label", cfg.train.seed), validation_data=val_data,
                 epochs=cfg.train.epochs, batch_size=1 if full_val is not None else None,
                 callbacks=callbacks, verbose=2)
    history["unet"] = {k: [float(x) for x in v] for k, v in h.history.items()}
    best = keras.models.load_model(ckpt_path, compile=False)

    # ---------------------------------------------------------------- 3) threshold on val
    _, val_probs, val_targets = evaluate_split(best, val_arr, 0.5, multiple)
    threshold, val_iou_at_th = best_threshold(val_probs, val_targets)
    val_summary, _, _ = evaluate_split(best, val_arr, threshold, multiple)

    meta = {
        "created": utc_now(),
        "data_source": data_source,
        "data_root": str(cfg.data.root),
        "split_strategy": strategy,
        "n_train": len(train_arr.ids),
        "n_val": len(val_arr.ids),
        "skipped": {"train": train_arr.skipped, "val": val_arr.skipped},
        "threshold": threshold,
        "threshold_selected_on": "validation split (max micro IoU)",
        "val_iou_at_threshold_flat": val_iou_at_th,
        "val_metrics": val_summary,
        "encoder_init": "autoencoder" if copied else "random",
        "epochs_ran": len(h.history["loss"]),
        "train_seconds": round(time.time() - t0, 1),
        "device": device,
        "bands": [{1: "VV", 2: "VH"}[b] for b in cfg.data.bands],
        "environment": environment_info(),
        "history": history,
    }
    save_json(meta, run_dir / METADATA_NAME)
    log.info("best checkpoint: %s | val micro IoU %.3f at threshold %.2f",
             ckpt_path, val_summary["micro"]["iou"], threshold)
    return run_dir
