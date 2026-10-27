import os

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def has_tf() -> bool:
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return False
    return True


requires_tf = pytest.mark.skipif(not has_tf(), reason="TensorFlow not installed")


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    from sar_flood.data.synthetic import write_dataset

    return write_dataset(tmp_path_factory.mktemp("synth"), n_per_event=3, size=64, seed=1)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
