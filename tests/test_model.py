from pathlib import Path

import pytest
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

model_path = Path(config["model"]["local_path"])


@pytest.mark.skipif(not model_path.exists(), reason="trained model artifact is absent")
def test_model_loading():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    assert model is not None
    assert tokenizer is not None


@pytest.mark.skipif(not model_path.exists(), reason="trained model artifact is absent")
def test_model_prediction():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    inputs = tokenizer(
        "Fire in the building!",
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()

    assert prediction in [0, 1]
