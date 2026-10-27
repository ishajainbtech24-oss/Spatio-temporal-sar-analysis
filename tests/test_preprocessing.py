import numpy as np
import pytest

from sar_flood.preprocessing.transforms import (
    nan_median_filter,
    pad_to_multiple,
    preprocess_chip,
    random_crop_and_flip,
)


def test_single_nan_does_not_poison_chip():
    """Regression: the notebook min-max turned a whole chip into NaN for one NaN pixel."""
    img = np.full((2, 32, 32), -10.0, dtype=np.float32)
    img[:, 5, 5] = np.nan
    out = preprocess_chip(img)
    assert np.isfinite(out.image).all()
    assert out.valid.sum() == 32 * 32 - 1
    assert not out.valid[5, 5]


def test_fixed_range_scaling_keeps_absolute_brightness():
    water = np.full((2, 16, 16), -22.0, dtype=np.float32)
    land = np.full((2, 16, 16), -8.0, dtype=np.float32)
    a = preprocess_chip(water, median_size=0).image
    b = preprocess_chip(land, median_size=0).image
    assert a.mean() < b.mean()  # per-image min-max would make both identical
    assert 0.0 <= a.min() and b.max() <= 1.0
    np.testing.assert_allclose(a[0, 0, 0], (-22 + 50) / 51, rtol=1e-6)


def test_output_shapes_and_label_masking(rng):
    img = rng.normal(-12, 3, size=(2, 40, 48)).astype(np.float32)
    img[:, :, :4] = np.nan
    lab = rng.integers(0, 2, size=(40, 48)).astype(np.int16)
    out = preprocess_chip(img, lab)
    assert out.image.shape == (40, 48, 2) and out.image.dtype == np.float32
    assert (out.label[:, :4] == -1).all()
    assert set(np.unique(out.label)) <= {-1, 0, 1}


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        preprocess_chip(np.zeros((2, 8, 8), np.float32), np.zeros((9, 8), np.int16))
    with pytest.raises(ValueError):
        preprocess_chip(np.zeros((8, 8), np.float32))


def test_inf_treated_as_nodata():
    img = np.full((1, 8, 8), -10.0, dtype=np.float32)
    img[0, 0, 0] = np.inf
    assert not preprocess_chip(img).valid[0, 0]


def test_nan_median_filter_preserves_nans_and_removes_spike():
    band = np.full((9, 9), -10.0, dtype=np.float32)
    band[4, 4] = 30.0
    band[0, 0] = np.nan
    out = nan_median_filter(band, 3)
    assert out[4, 4] == -10.0 and np.isnan(out[0, 0])


def test_pad_to_multiple():
    x = np.zeros((50, 61, 2), np.float32)
    padded, (h, w) = pad_to_multiple(x, 16)
    assert padded.shape == (64, 64, 2) and (h, w) == (50, 61)


def test_random_crop_is_aligned(rng):
    img = np.arange(64 * 64, dtype=np.float32).reshape(64, 64, 1)
    lab = (np.arange(64 * 64).reshape(64, 64) % 2).astype(np.int8)
    for _ in range(10):
        ci, cl = random_crop_and_flip(img, lab, 32, rng)
        assert ci.shape == (32, 32, 1) and cl.shape == (32, 32)
        np.testing.assert_array_equal(ci[..., 0].astype(int) % 2, cl)  # same geometry applied
