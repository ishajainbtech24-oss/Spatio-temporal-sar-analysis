"""Segmentation inference: checkpoint loading and full-chip probability maps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Config
from ..preprocessing.transforms import pad_to_multiple, preprocess_from_config
from ..utils import load_json

CHECKPOINT_NAME = "model.keras"
METADATA_NAME = "metadata.json"


def predict_proba(model, image_hwc: np.ndarray, multiple: int = 16) -> np.ndarray:
    """Water probability (H, W) for one preprocessed (H, W, C) image of any size."""
    padded, (h, w) = pad_to_multiple(image_hwc, multiple)
    prob = model.predict(padded[None], verbose=0)[0, ..., 0]
    return prob[:h, :w].astype(np.float32)


@dataclass
class Segmenter:
    """A trained U-Net plus everything needed to use it consistently."""

    model: object
    cfg: Config
    threshold: float
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> Segmenter:
        import keras  # noqa: PLC0415

        from ..config import load_config  # noqa: PLC0415
        from ..models import unet  # noqa: F401,PLC0415  (registers custom objects)

        run_dir = Path(run_dir)
        ckpt, meta_path = run_dir / CHECKPOINT_NAME, run_dir / METADATA_NAME
        if not ckpt.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No trained checkpoint in '{run_dir.name}'. Train one with `sar-flood train` "
                "(see README) or point --run-dir to an existing run."
            )
        meta = load_json(meta_path)
        meta["run_name"] = run_dir.name
        cfg = load_config(run_dir / "config.yaml")
        model = keras.models.load_model(ckpt, compile=False)
        return cls(model=model, cfg=cfg, threshold=float(meta.get("threshold", 0.5)), metadata=meta)

    @property
    def multiple(self) -> int:
        return 2 ** self.cfg.model.depth

    def predict_chip(self, image_chw: np.ndarray) -> dict:
        proc = preprocess_from_config(image_chw, None, self.cfg.preprocess)
        prob = predict_proba(self.model, proc.image, self.multiple)
        prob[~proc.valid] = 0.0
        mask = (prob >= self.threshold) & proc.valid
        return {"prob": prob, "mask": mask, "valid": proc.valid}
