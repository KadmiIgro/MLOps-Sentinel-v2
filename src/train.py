import inspect
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data_loader import TextDataLoader


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["model"]["learning_rate"] = float(config["model"]["learning_rate"])
    config["model"]["epochs"] = int(config["model"]["epochs"])
    config["model"]["batch_size"] = int(config["model"]["batch_size"])
    config["model"]["warmup_steps"] = int(config["model"]["warmup_steps"])
    config["model"]["weight_decay"] = float(config["model"]["weight_decay"])
    return config


def compute_metrics(eval_pred: tuple[Any, Any]) -> dict[str, float]:
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


def build_dataset(texts: pd.Series, labels: pd.Series) -> Dataset:
    return Dataset.from_pandas(pd.DataFrame({"text": texts, "label": labels}))


def main() -> None:
    config = load_config()
    loader = TextDataLoader(config)
    x_train, x_test, y_train, y_test = loader.get_data()

    train_dataset = build_dataset(x_train, y_train)
    test_dataset = build_dataset(x_test, y_test)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    def tokenize_function(examples: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=config["api"]["max_text_length"],
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["name"],
        num_labels=config["model"]["num_labels"],
    )

    training_kwargs = {
        "output_dir": "./results",
        "num_train_epochs": config["model"]["epochs"],
        "per_device_train_batch_size": config["model"]["batch_size"],
        "per_device_eval_batch_size": config["model"]["batch_size"] * 4,
        "warmup_steps": config["model"]["warmup_steps"],
        "weight_decay": config["model"]["weight_decay"],
        "learning_rate": config["model"]["learning_rate"],
        "logging_dir": "./logs",
        "logging_steps": 50,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "report_to": [],
    }
    strategy_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    training_kwargs[strategy_key] = "epoch"
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    model_path = Path(config["model"]["local_path"])
    with mlflow.start_run():
        mlflow.log_params(config["model"])
        trainer.train()
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        eval_results = trainer.evaluate()
        mlflow.log_metrics(eval_results)
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=config["task"]["name"],
        )

    print(f"Model saved to {model_path}")
    print(f"Metrics: {eval_results}")


if __name__ == "__main__":
    main()
