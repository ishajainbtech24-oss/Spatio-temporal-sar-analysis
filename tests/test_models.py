import numpy as np
import pytest
from conftest import requires_tf

pytestmark = requires_tf


def test_unet_forward_shape_and_range():
    from sar_flood.models.unet import build_unet

    model = build_unet(in_channels=2, base_filters=4, depth=3)
    out = model.predict(np.random.rand(2, 48, 64, 2).astype("float32"), verbose=0)
    assert out.shape == (2, 48, 64, 1)
    assert np.all((out >= 0) & (out <= 1))


def test_autoencoder_reconstructs_input_shape():
    from sar_flood.models.unet import build_autoencoder

    ae = build_autoencoder(2, 4, 3)
    assert ae.predict(np.zeros((1, 32, 32, 2), "float32"), verbose=0).shape == (1, 32, 32, 2)


def test_encoder_weight_transfer():
    from sar_flood.models.unet import build_autoencoder, build_unet, transfer_encoder_weights

    ae, unet = build_autoencoder(2, 4, 2), build_unet(2, 4, 2)
    copied = transfer_encoder_weights(ae, unet)
    assert copied and all(n.startswith(("enc", "bott")) for n in copied)
    for name in copied:
        for a, b in zip(ae.get_layer(name).get_weights(), unet.get_layer(name).get_weights(), strict=True):
            np.testing.assert_array_equal(a, b)


def test_masked_loss_ignores_invalid_pixels():
    from sar_flood.models.unet import masked_bce_dice

    pred = np.full((1, 4, 4, 1), 0.9, "float32")
    y = np.ones((1, 4, 4, 1), "float32")
    base = float(masked_bce_dice(y, pred))
    y2 = y.copy()
    pred2 = pred.copy()
    y2[0, 0, 0, 0] = -1
    pred2[0, 0, 0, 0] = 0.0001  # terrible prediction, but on an ignored pixel
    assert float(masked_bce_dice(y2, pred2)) == pytest.approx(base, rel=1e-3)
    assert np.isfinite(float(masked_bce_dice(np.full_like(y, -1), pred)))


def test_masked_iou_metric():
    from sar_flood.models.unet import MaskedIoU

    m = MaskedIoU()
    y = np.array([1, 0, 1, -1], "float32").reshape(1, 2, 2, 1)
    p = np.array([0.9, 0.9, 0.1, 0.9], "float32").reshape(1, 2, 2, 1)
    m.update_state(y, p)
    assert float(m.result()) == pytest.approx(1 / 3)


def test_checkpoint_roundtrip(tmp_path):
    import keras

    from sar_flood.models.unet import build_unet

    model = build_unet(2, 4, 2)
    x = np.random.rand(1, 16, 16, 2).astype("float32")
    path = tmp_path / "m.keras"
    model.save(path)
    loaded = keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(model.predict(x, verbose=0), loaded.predict(x, verbose=0), atol=1e-6)


def test_legacy_classifier_loads_and_outputs_probability():
    from sar_flood.models.legacy import (
        interpret_output,
        legacy_preprocess_vv,
        load_legacy_classifier,
    )

    model = load_legacy_classifier()
    vv = np.random.default_rng(0).normal(-12, 4, (200, 180)).astype("float32")
    vv[:10] = np.nan
    x = legacy_preprocess_vv(vv)
    assert x.shape == (1, 128, 128, 1) and np.isfinite(x).all()
    p = float(model.predict(x, verbose=0)[0, 0])
    assert 0 <= p <= 1
    out = interpret_output(p)
    assert out["p_water_present"] == pytest.approx(1 - out["p_no_water"], abs=1e-4)


def test_legacy_missing_file_error(tmp_path):
    from sar_flood.models.legacy import load_legacy_classifier

    with pytest.raises(FileNotFoundError):
        load_legacy_classifier(tmp_path / "nope.keras")
