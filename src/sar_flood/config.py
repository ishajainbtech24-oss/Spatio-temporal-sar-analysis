"""Central, typed configuration.

Configuration is read from a YAML file (see ``configs/``) and can be overridden by
environment variables for paths that differ between machines:

* ``SAR_DATA_ROOT``   -> ``data.root``
* ``SAR_OUTPUT_DIR``  -> ``output_dir``

Every run writes the fully-resolved configuration next to its outputs so a result can
always be traced back to the exact settings that produced it.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    # Directory that contains ``data/flood_events/HandLabeled/{S1Hand,LabelHand}``
    # (the Sen1Floods11 v1.1 / v1.2 layout).
    root: str = "data/raw/sen1floods11/v1.1"
    # "official": use flood_{train,valid,test}_data.csv if found under ``root``.
    # "group": group-aware split by flood event (country) -> no event in two splits.
    # "auto": official if the CSVs exist, otherwise group.
    split_strategy: str = "auto"
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    # Bands of S1Hand chips to use, 1-based (1 = VV, 2 = VH).
    bands: list[int] = field(default_factory=lambda: [1, 2])


@dataclass
class PreprocessConfig:
    # Median filter window for speckle reduction (0 or 1 disables it).
    median_size: int = 3
    # Backscatter (dB) clip range before linear scaling to [0, 1]. This is the range
    # used by the Sen1Floods11 reference baseline.
    db_min: float = -50.0
    db_max: float = 1.0
    # Value written into invalid (no-data) pixels of the network input.
    fill_value: float = 0.0


@dataclass
class ModelConfig:
    base_filters: int = 16
    depth: int = 4
    dropout: float = 0.1


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 8
    epochs: int = 40
    learning_rate: float = 1e-3
    crop_size: int = 256
    augment: bool = True
    early_stopping_patience: int = 8
    # Self-supervised autoencoder pre-training (training split images only).
    ae_epochs: int = 20
    use_autoencoder_init: bool = True
    # "auto" -> GPU if TensorFlow sees one, else CPU. "cpu" forces CPU.
    device: str = "auto"
    # Cap on number of training chips (None = all). Useful for smoke runs.
    max_train_samples: int | None = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "outputs"
    run_name: str = "unet"

    # ------------------------------------------------------------------ helpers
    def resolve(self, path: str | os.PathLike) -> Path:
        """Resolve a possibly-relative path against the project root."""
        p = Path(path).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def data_root(self) -> Path:
        return self.resolve(self.data.root)

    @property
    def run_dir(self) -> Path:
        return self.resolve(self.output_dir) / "runs" / self.run_name

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _merge(dc: Any, values: dict[str, Any]) -> Any:
    """Recursively apply ``values`` onto dataclass instance ``dc`` (unknown keys fail)."""
    names = {f.name: f for f in dataclasses.fields(dc)}
    for key, value in (values or {}).items():
        if key not in names:
            raise KeyError(f"Unknown config key '{key}' for {type(dc).__name__}")
        current = getattr(dc, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(dc, key, value)
    return dc


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Config:
    """Load a :class:`Config` from YAML, then apply env-vars and explicit overrides."""
    cfg = Config()
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            _merge(cfg, yaml.safe_load(fh) or {})
    if os.environ.get("SAR_DATA_ROOT"):
        cfg.data.root = os.environ["SAR_DATA_ROOT"]
    if os.environ.get("SAR_OUTPUT_DIR"):
        cfg.output_dir = os.environ["SAR_OUTPUT_DIR"]
    if overrides:
        _merge(cfg, overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    if cfg.data.split_strategy not in {"auto", "official", "group"}:
        raise ValueError("data.split_strategy must be one of auto|official|group")
    if not cfg.data.bands or any(b not in (1, 2) for b in cfg.data.bands):
        raise ValueError("data.bands must be a non-empty subset of [1, 2] (VV=1, VH=2)")
    if not 0 < cfg.data.val_fraction < 0.5 or not 0 < cfg.data.test_fraction < 0.5:
        raise ValueError("val_fraction and test_fraction must be in (0, 0.5)")
    if cfg.preprocess.db_min >= cfg.preprocess.db_max:
        raise ValueError("preprocess.db_min must be < preprocess.db_max")
    if cfg.train.crop_size % (2 ** cfg.model.depth) != 0:
        raise ValueError("train.crop_size must be divisible by 2**model.depth")
    if cfg.train.batch_size < 1 or cfg.train.epochs < 1:
        raise ValueError("batch_size and epochs must be >= 1")
    if cfg.train.device not in {"auto", "cpu", "gpu"}:
        raise ValueError("train.device must be auto|cpu|gpu")
