import pytest
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_path = f"models/{config['task']['name']}"

def test_model_loading():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    assert model is not None
    assert tokenizer is not None

def test_model_prediction():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    inputs = tokenizer("Fire in the building!", return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    
    assert pred in [0, 1]