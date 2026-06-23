import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
import mlflow
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from data_loader import TextDataLoader

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

config['model']['learning_rate'] = float(config['model']['learning_rate'])
config['model']['epochs'] = int(config['model']['epochs'])
config['model']['batch_size'] = int(config['model']['batch_size'])
config['model']['warmup_steps'] = int(config['model']['warmup_steps'])
config['model']['weight_decay'] = float(config['model']['weight_decay'])

loader = TextDataLoader(config)
X_train, X_test, y_train, y_test = loader.get_data()

train_df = pd.DataFrame({'text': X_train, 'label': y_train})
test_df = pd.DataFrame({'text': X_test, 'label': y_test})

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=config['api']['max_text_length']
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    config['model']['name'],
    num_labels=config['model']['num_labels']
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": acc, "f1": f1}

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=config['model']['epochs'],
    per_device_train_batch_size=config['model']['batch_size'],
    per_device_eval_batch_size=config['model']['batch_size'] * 4,
    warmup_steps=config['model']['warmup_steps'],
    weight_decay=config['model']['weight_decay'],
    learning_rate=config['model']['learning_rate'],
    logging_dir="./logs",
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to=None,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

with mlflow.start_run():
    mlflow.log_params(config['model'])
    trainer.train()
    model_path = f"models/{config['task']['name']}"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name="disaster_classification"
    )
    eval_results = trainer.evaluate()
    mlflow.log_metrics(eval_results)
    print(f"✅ Model saved to MLflow")
    print(f"📊 Metrics: {eval_results}")