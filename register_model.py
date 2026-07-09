import os
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient
from transformers import AutoModelForSequenceClassification

with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
model_name = config["task"]["name"]
model_path = Path(os.getenv("MODEL_PATH", config["model"]["local_path"]))

if not model_path.exists():
    raise FileNotFoundError(f"Local model path does not exist: {model_path}")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(config["mlflow"]["experiment_name"])

model = AutoModelForSequenceClassification.from_pretrained(model_path)
client = MlflowClient()

with mlflow.start_run() as run:
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name=model_name,
    )
    run_id = run.info.run_id

latest_versions = client.get_latest_versions(model_name)
latest_version = max(int(version.version) for version in latest_versions)
client.set_registered_model_alias(
    name=model_name,
    version=latest_version,
    alias="Production",
)

print(f"Registered {model_name} version {latest_version} from run {run_id}")
print("Alias Production now points to the latest registered version")
