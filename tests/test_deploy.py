"""Deployment helpers: publish-model command and the Hugging Face staging step (no network)."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path, source="sen1floods11"):
    run = tmp_path / "outputs" / "runs" / "unet_test"
    run.mkdir(parents=True)
    (run / "model.keras").write_bytes(b"weights")
    (run / "config.yaml").write_text("run_name: unet_test\n")
    (run / "metadata.json").write_text(json.dumps({"data_source": source}))
    return run


def test_publish_model_copies_only_inference_files(tmp_path, monkeypatch):
    import sar_flood.config as config
    from sar_flood.cli import main

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    run = _run(tmp_path)
    (run / "autoencoder.keras").write_bytes(b"big")
    assert main(["publish-model", str(run)]) == 0
    dest = tmp_path / "models" / "unet" / "unet_test"
    assert sorted(p.name for p in dest.iterdir()) == ["config.yaml", "metadata.json", "model.keras"]


def test_publish_model_refuses_synthetic(tmp_path, monkeypatch, capsys):
    import sar_flood.config as config
    from sar_flood.cli import main

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    run = _run(tmp_path, source="synthetic")
    assert main(["publish-model", str(run)]) == 2
    assert "synthetic" in capsys.readouterr().err
    assert not (tmp_path / "models" / "unet").exists()


def _deploy_module():
    spec = importlib.util.spec_from_file_location("deploy_hf_space", ROOT / "scripts" / "deploy_hf_space.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_models_rejects_synthetic(tmp_path):
    mod = _deploy_module()
    run = tmp_path / "models" / "unet" / "smoke"
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({"data_source": "synthetic"}))
    with pytest.raises(SystemExit, match="synthetic"):
        mod.check_models(tmp_path, lite=False)
    (run / "metadata.json").write_text(json.dumps({"data_source": "sen1floods11"}))
    mod.check_models(tmp_path, lite=False)
    with pytest.raises(SystemExit, match="lite"):
        mod.check_models(tmp_path, lite=True)


def test_workflows_are_valid_yaml():
    import yaml

    deploy = yaml.safe_load((ROOT / ".github" / "workflows" / "deploy.yml").read_text())
    triggers = deploy[True]  # YAML parses the key "on" as True
    assert triggers["workflow_run"]["workflows"] == ["tests"]
    assert triggers["workflow_run"]["branches"] == ["main"]
    assert "HF_TOKEN" in json.dumps(deploy) and "HF_SPACE" in json.dumps(deploy)
    tests = yaml.safe_load((ROOT / ".github" / "workflows" / "tests.yml").read_text())
    assert tests["name"] == "tests"
