import pytest

from sar_flood.config import load_config


def test_default_config_file_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.data.bands == [1, 2]
    assert cfg.train.crop_size % 2 ** cfg.model.depth == 0


def test_smoke_config_and_overrides():
    cfg = load_config("configs/smoke.yaml", {"train": {"epochs": 1}})
    assert cfg.train.epochs == 1 and cfg.run_name == "smoke_synthetic"


def test_env_overrides_data_root(monkeypatch):
    monkeypatch.setenv("SAR_DATA_ROOT", "/somewhere/else")
    assert load_config().data.root == "/somewhere/else"


@pytest.mark.parametrize("override", [
    {"data": {"bands": [3]}},
    {"data": {"split_strategy": "random"}},
    {"preprocess": {"db_min": 5, "db_max": 1}},
    {"train": {"crop_size": 100}},
    {"nonsense": 1},
])
def test_invalid_config_rejected(override):
    with pytest.raises((ValueError, KeyError)):
        load_config(None, override)


def test_missing_config_file():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")
