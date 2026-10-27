"""Small shared helpers: logging, seeding, JSON I/O, environment info."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "sar_flood") -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=os.environ.get("SAR_LOG_LEVEL", "INFO"), format=LOG_FORMAT)
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # noqa: PLC0415

        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def _default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def _clean_nans(obj: Any) -> Any:
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nans(v) for v in obj]
    return obj


def save_json(obj: Any, path: str | os.PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_clean_nans(obj), fh, indent=2, default=_default)
    return path


def load_json(path: str | os.PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import tensorflow as tf  # noqa: PLC0415

        info["tensorflow"] = tf.__version__
        info["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except ImportError:
        info["tensorflow"] = None
    return info


def configure_device(device: str) -> str:
    """Apply the device preference and return the device that will actually be used."""
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        import tensorflow as tf  # noqa: PLC0415
    except ImportError:
        return "none"
    gpus = tf.config.list_physical_devices("GPU")
    if device == "gpu" and not gpus:
        raise RuntimeError("train.device=gpu was requested but TensorFlow sees no GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return "gpu" if gpus and device != "cpu" else "cpu"
