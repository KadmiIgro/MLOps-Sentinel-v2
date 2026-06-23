import yaml
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

# Загрузка конфига
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ЗАГРУЖАЕМ МОДЕЛЬ ИЗ ПАПКИ (НЕ ИЗ MLflow!)
model_path = f"models/{config['task']['name']}"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# FastAPI
app = FastAPI(
    title="MLOps Sentinel API",
    description="Классификация текстов на наличие катастроф",
    version="1.0.0"
)

class PredictRequest(BaseModel):
    text: str

    @validator('text')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Текст не может быть пустым')
        return v

class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    label: str

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=config['api']['max_text_length']
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
        
        labels_map = config['task'].get('labels', {})
        label = labels_map.get(prediction, "unknown")
        
        return PredictResponse(
            prediction=prediction,
            confidence=confidence,
            label=label
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "MLOps Sentinel API", "docs": "/docs"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)