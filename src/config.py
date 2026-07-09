import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or os.getenv("CONFIG_PATH", "config.yaml"))
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if tracking_uri := os.getenv("MLFLOW_TRACKING_URI"):
        config["mlflow"]["tracking_uri"] = tracking_uri
    if model_uri := os.getenv("MODEL_URI"):
        config["mlflow"]["model_uri"] = model_uri
    if model_path := os.getenv("MODEL_PATH"):
        config["model"]["local_path"] = model_path

    return config


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    return load_config()
