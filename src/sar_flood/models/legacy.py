"""Loader for the original scene-level classifier (notebook 07), kept for comparison.

``models/legacy/simple_transformer_pretrained_final.keras`` = MobileNetV2 (ImageNet init,
frozen) -> 16 tokens -> 1 transformer block -> sigmoid, on 128x128 single-band (VV) chips.

Two problems with the original artefact are handled here:

1. **It cannot be loaded on Python 3.13.** The graph contains a ``Lambda`` layer whose
   Python bytecode was pickled under Python 3.12 (Colab); deserialising it on 3.13
   crashes the interpreter (segfault). We rebuild the identical architecture in code,
   replacing the Lambda (``t * 2 - 1``) with an equivalent ``Rescaling`` layer, and load
   only the weights. Outputs were checked against the original model under Python 3.12
   (max abs difference recorded in docs/PROJECT_AUDIT.md).

2. **Its output is P(chip has NO water), not P(flood).** In notebook 02 the labels were
   built as "Flood" if any mask pixel > 0 else "NoFlood", then ``LabelEncoder`` mapped
   them alphabetically: Flood -> 0, NoFlood -> 1. Notebook 07 then treated class 1 as
   "Flood". So the minority positive class the model learned is "chip without any
   labelled water", and the dashboard showed inverted results. This module exposes the
   output under its correct meaning.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import numpy as np

LEGACY_DIR = Path(__file__).resolve().parents[3] / "models" / "legacy"
LEGACY_CLASSIFIER = LEGACY_DIR / "simple_transformer_pretrained_final.keras"
# Threshold printed by notebook 07 for this exact saved model (tuned on a validation
# split for F1 of the minority class, i.e. "no water"). The old app used 0.20 with no
# recorded source.
LEGACY_THRESHOLD_NO_WATER = 0.13
INPUT_SIZE = 128


def _position_embedding_cls():
    import keras  # noqa: PLC0415
    from keras import layers, ops  # noqa: PLC0415

    @keras.saving.register_keras_serializable(package="sar_flood_legacy")
    class PositionEmbedding(layers.Layer):
        def __init__(self, sequence_length, embed_dim, **kwargs):
            super().__init__(**kwargs)
            self.sequence_length = sequence_length
            self.embed_dim = embed_dim
            self.position_embedding = layers.Embedding(sequence_length, embed_dim)

        def build(self, input_shape):
            self.position_embedding.build((self.sequence_length,))
            super().build(input_shape)

        def call(self, inputs):
            positions = ops.arange(0, ops.shape(inputs)[1])
            return inputs + self.position_embedding(positions)

        def get_config(self):
            return {**super().get_config(), "sequence_length": self.sequence_length,
                    "embed_dim": self.embed_dim}

    return PositionEmbedding


def build_legacy_classifier():
    """Architecture of notebook 07's ``build_pretrained_model`` (weights not loaded)."""
    import keras  # noqa: PLC0415
    from keras import layers  # noqa: PLC0415

    PositionEmbedding = _position_embedding_cls()
    inputs = layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 1))
    x = layers.Concatenate()([inputs, inputs, inputs])
    x = layers.Rescaling(2.0, offset=-1.0)(x)  # == Lambda(lambda t: t * 2.0 - 1.0)
    backbone = keras.applications.MobileNetV2(input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
                                              include_top=False, weights=None)
    tokens = layers.Reshape((16, 1280))(backbone(x))
    y = layers.Dense(32)(tokens)
    y = PositionEmbedding(16, 32)(y)
    # transformer block (2 heads, key_dim 32)
    h = layers.LayerNormalization()(y)
    h = layers.MultiHeadAttention(num_heads=2, key_dim=32)(h, h)
    y = layers.Add()([y, h])
    z = layers.LayerNormalization()(y)
    z = layers.Dense(64, activation="relu")(z)
    z = layers.Dense(32)(z)
    y = layers.Add()([y, z])
    y = layers.GlobalAveragePooling1D()(y)
    y = layers.Dense(64, activation="relu")(y)
    y = layers.Dropout(0.3)(y)
    outputs = layers.Dense(1, activation="sigmoid")(y)
    return keras.Model(inputs, outputs, name="legacy_mobilenetv2_transformer")


def load_legacy_classifier(path: str | Path = LEGACY_CLASSIFIER):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Legacy model not found: {path.name}")
    model = build_legacy_classifier()
    with zipfile.ZipFile(path) as zf, tempfile.TemporaryDirectory() as tmp:
        zf.extract("model.weights.h5", tmp)
        model.load_weights(Path(tmp) / "model.weights.h5")
    return model


# ----------------------------------------------------------------------------- inputs
def legacy_preprocess_vv(vv: np.ndarray) -> np.ndarray:
    """Training-time preprocessing of notebook 02, made NaN-safe.

    median filter (3) -> per-image min-max -> resize to 128x128 (skimage, anti-aliased).
    """
    from skimage.transform import resize  # noqa: PLC0415

    from ..preprocessing.transforms import nan_median_filter  # noqa: PLC0415

    vv = nan_median_filter(vv.astype(np.float32), 3)
    finite = np.isfinite(vv)
    if not finite.any():
        raise ValueError("image has no valid pixels")
    lo, hi = np.min(vv[finite]), np.max(vv[finite])
    vv = (vv - lo) / (hi - lo + 1e-8)
    vv[~finite] = 0.0  # the notebooks applied nan_to_num -> 0 before training
    vv = resize(vv, (INPUT_SIZE, INPUT_SIZE), anti_aliasing=True)
    return vv[None, :, :, None].astype(np.float32)


def legacy_preprocess_image_file(path: str | Path) -> np.ndarray:
    """Same preprocessing for an 8-bit PNG/JPEG (as the old dashboard accepted)."""
    import cv2  # noqa: PLC0415

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("could not decode image")
    return legacy_preprocess_vv(img.astype(np.float32))


def interpret_output(p_no_water: float, threshold: float = LEGACY_THRESHOLD_NO_WATER) -> dict:
    return {
        "p_no_water": round(float(p_no_water), 4),
        "p_water_present": round(1.0 - float(p_no_water), 4),
        "threshold_no_water": threshold,
        "label": "No surface water" if p_no_water >= threshold else "Surface water present",
    }
