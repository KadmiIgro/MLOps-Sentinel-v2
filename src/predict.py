import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import get_config


def predict_text(text: str) -> dict[str, float | int | str]:
    config = get_config()
    model_path = Path(config["model"]["local_path"])

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}. Run `python -m src.train`."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=config["api"]["max_text_length"],
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

    labels = config["task"].get("labels", {})
    return {
        "prediction": prediction,
        "confidence": round(confidence, 6),
        "label": labels.get(str(prediction), f"class_{prediction}"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local text classification.")
    parser.add_argument("text", help="Text to classify")
    args = parser.parse_args()

    print(json.dumps(predict_text(args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
