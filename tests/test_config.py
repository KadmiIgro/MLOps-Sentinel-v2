from pathlib import Path

from src.config import load_config


def test_load_config_applies_environment_overrides(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://example-mlflow:5000")
    monkeypatch.setenv("MODEL_URI", "models:/example/Production")
    monkeypatch.setenv("MODEL_PATH", "/tmp/example-model")

    config = load_config(Path("config.yaml"))

    assert config["mlflow"]["tracking_uri"] == "http://example-mlflow:5000"
    assert config["mlflow"]["model_uri"] == "models:/example/Production"
    assert config["model"]["local_path"] == "/tmp/example-model"
