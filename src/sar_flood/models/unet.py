"""Convolutional autoencoder (self-supervised feature learning) and U-Net segmenter.

Both networks share an identically named encoder (``enc{i}_*`` + ``bott_*`` layers), so
the encoder learned by reconstructing unlabelled SAR chips can initialise the U-Net.
This keeps the project's original idea - autoencoder-based feature learning - but puts
the learned features to work on the actual task (per-pixel water segmentation) instead
of flattening them into a 16,384-dim vector for a Random Forest.

No ``Lambda`` layers and no custom layers are used, so saved models load on any Python /
Keras 3 version (the legacy model segfaults on Python 3.13 because of a Lambda layer).
"""

from __future__ import annotations

import keras
from keras import layers, ops

IGNORE_INDEX = -1


# ----------------------------------------------------------------------------- blocks
def _conv_block(x, filters: int, prefix: str, dropout: float = 0.0):
    for j in (1, 2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{prefix}_conv{j}")(x)
        x = layers.BatchNormalization(momentum=0.9, name=f"{prefix}_bn{j}")(x)
        x = layers.Activation("relu", name=f"{prefix}_relu{j}")(x)
    if dropout:
        x = layers.SpatialDropout2D(dropout, name=f"{prefix}_drop")(x)
    return x


def _encoder(inputs, base_filters: int, depth: int, dropout: float):
    skips, x = [], inputs
    for i in range(depth):
        x = _conv_block(x, base_filters * 2**i, f"enc{i}")
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"enc{i}_pool")(x)
    x = _conv_block(x, base_filters * 2**depth, "bott", dropout)
    return x, skips


def encoder_layer_names(model: keras.Model) -> list[str]:
    return [lyr.name for lyr in model.layers if lyr.name.startswith(("enc", "bott")) and lyr.weights]


# ----------------------------------------------------------------------------- models
def build_autoencoder(in_channels: int = 2, base_filters: int = 16, depth: int = 4,
                      dropout: float = 0.0) -> keras.Model:
    """Encoder + skip-free decoder reconstructing the (scaled) SAR input."""
    inputs = layers.Input((None, None, in_channels), name="sar")
    x, _ = _encoder(inputs, base_filters, depth, dropout)
    for i in reversed(range(depth)):
        x = layers.UpSampling2D(2, name=f"aedec{i}_up")(x)
        x = _conv_block(x, base_filters * 2**i, f"aedec{i}")
    outputs = layers.Conv2D(in_channels, 1, activation="sigmoid", name="reconstruction")(x)
    return keras.Model(inputs, outputs, name="sar_autoencoder")


def build_unet(in_channels: int = 2, base_filters: int = 16, depth: int = 4,
               dropout: float = 0.1) -> keras.Model:
    """U-Net producing a per-pixel water probability map of the input's size."""
    inputs = layers.Input((None, None, in_channels), name="sar")
    x, skips = _encoder(inputs, base_filters, depth, dropout)
    for i in reversed(range(depth)):
        x = layers.Conv2DTranspose(base_filters * 2**i, 2, strides=2, name=f"dec{i}_up")(x)
        x = layers.Concatenate(name=f"dec{i}_cat")([x, skips[i]])
        x = _conv_block(x, base_filters * 2**i, f"dec{i}")
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="water_prob")(x)
    return keras.Model(inputs, outputs, name="sar_unet")


def transfer_encoder_weights(src: keras.Model, dst: keras.Model) -> list[str]:
    """Copy weights of every shared ``enc*``/``bott*`` layer from ``src`` to ``dst``."""
    copied = []
    dst_layers = {lyr.name: lyr for lyr in dst.layers}
    for name in encoder_layer_names(src):
        if name in dst_layers:
            dst_layers[name].set_weights(src.get_layer(name).get_weights())
            copied.append(name)
    return copied


# ----------------------------------------------------------------------------- losses
@keras.saving.register_keras_serializable(package="sar_flood")
def masked_bce_dice(y_true, y_pred):
    """BCE + soft Dice over valid pixels only (``y_true == -1`` is ignored)."""
    y_true = ops.cast(y_true, "float32")
    if len(y_true.shape) == 3:
        y_true = ops.expand_dims(y_true, -1)
    valid = ops.cast(ops.greater_equal(y_true, 0.0), "float32")
    target = ops.clip(y_true, 0.0, 1.0)
    p = ops.clip(y_pred, 1e-6, 1.0 - 1e-6)
    bce = -(target * ops.log(p) + (1.0 - target) * ops.log(1.0 - p))
    n_valid = ops.maximum(ops.sum(valid), 1.0)
    bce = ops.sum(bce * valid) / n_valid
    inter = ops.sum(p * target * valid)
    denom = ops.sum(p * valid) + ops.sum(target * valid)
    dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
    return bce + dice


@keras.saving.register_keras_serializable(package="sar_flood")
def masked_mse(y_true, y_pred):
    """Reconstruction loss; ``y_true`` carries the validity mask as its last channel."""
    target, valid = y_true[..., :-1], y_true[..., -1:]
    err = ops.square(target - y_pred) * valid
    return ops.sum(err) / ops.maximum(ops.sum(valid) * ops.cast(ops.shape(target)[-1], "float32"), 1.0)


@keras.saving.register_keras_serializable(package="sar_flood")
class MaskedIoU(keras.metrics.Metric):
    """Streaming micro IoU of the water class at a fixed threshold, ignoring -1 pixels."""

    def __init__(self, threshold: float = 0.5, name: str = "iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_variable(shape=(), initializer="zeros", name="tp")
        self.fp = self.add_variable(shape=(), initializer="zeros", name="fp")
        self.fn = self.add_variable(shape=(), initializer="zeros", name="fn")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(y_true, "float32")
        if len(y_true.shape) == 3:
            y_true = ops.expand_dims(y_true, -1)
        valid = ops.cast(ops.greater_equal(y_true, 0.0), "float32")
        t = ops.cast(ops.equal(y_true, 1.0), "float32")
        p = ops.cast(ops.greater_equal(y_pred, self.threshold), "float32") * valid
        self.tp.assign_add(ops.sum(p * t))
        self.fp.assign_add(ops.sum(p * (1.0 - t) * valid))
        self.fn.assign_add(ops.sum((1.0 - p) * t * valid))

    def result(self):
        return self.tp / ops.maximum(self.tp + self.fp + self.fn, 1.0)

    def reset_state(self):
        for v in (self.tp, self.fp, self.fn):
            v.assign(0.0)

    def get_config(self):
        return {**super().get_config(), "threshold": self.threshold}
