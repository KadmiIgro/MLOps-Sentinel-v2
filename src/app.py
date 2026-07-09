import logging
from pathlib import Path
from typing import Any

import mlflow
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, field_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import get_config

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        if self.is_ready:
            return

        try:
            mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
            model_uri = self.config["mlflow"].get("model_uri")
            local_path = Path(self.config["model"]["local_path"])

            if model_uri:
                logger.info("Loading model from MLflow URI: %s", model_uri)
                self.model = mlflow.pytorch.load_model(model_uri)
            elif local_path.exists():
                logger.info("Loading model from local path: %s", local_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    local_path
                )
            else:
                raise FileNotFoundError(
                    "Model is not available. "
                    f"Train it into {local_path} or set MODEL_URI."
                )

            tokenizer_source = (
                local_path if local_path.exists() else self.config["model"]["name"]
            )
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
            self.model.eval()
            self.load_error = None
        except Exception as exc:
            self.load_error = str(exc)
            logger.exception("Model loading failed")
            raise

    def predict(self, text: str) -> dict[str, Any]:
        self.load()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config["api"]["max_text_length"],
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()

        labels_map = self.config["task"].get("labels", {})
        label = labels_map.get(str(prediction), f"class_{prediction}")
        return {
            "prediction": prediction,
            "confidence": round(confidence, 6),
            "label": label,
        }


config = get_config()
predictor = PredictionService(config)

app = FastAPI(
    title="MLOps Sentinel API",
    description="Text classification API for disaster tweet detection.",
    version="1.0.0",
)
Instrumentator().instrument(app).expose(
    app, endpoint="/metrics", include_in_schema=False
)


class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Text must not be empty")
        return value.strip()


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    label: str


@app.get("/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "model_loaded": predictor.is_ready,
        "model_error": predictor.load_error,
    }


@app.get("/ready")
async def readiness_check() -> dict[str, Any]:
    try:
        predictor.load()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"status": "ready", "model_loaded": predictor.is_ready}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    try:
        return PredictResponse(**predictor.predict(request.text))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "message": "MLOps Sentinel API",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "metrics": "/metrics",
    }


if __name__ == "__main__":
    logging.basicConfig(level=config["api"].get("log_level", "INFO"))
    uvicorn.run(app, host=config["api"]["host"], port=config["api"]["port"])
