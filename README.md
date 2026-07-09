# MLOps Sentinel

[![CI/CD](https://github.com/KadmiIgro/MLOps-Sentinel-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/KadmiIgro/MLOps-Sentinel-v2/actions/workflows/ci.yml)

Production-style MLOps pipeline for text classification. The service classifies a
tweet as `disaster` or `not_disaster`, exposes predictions through FastAPI,
tracks experiments with MLflow, and publishes Prometheus metrics for Grafana.

## Stack

- Model: `distilbert-base-uncased`
- API: FastAPI, Swagger UI at `/docs`
- Training: Hugging Face Transformers, Datasets, scikit-learn
- Experiment tracking: MLflow
- Monitoring: Prometheus and Grafana
- Packaging: Docker and Docker Compose
- CI: GitHub Actions with tests, linting, and Docker smoke check

`requirements.txt` contains the full development and training environment.
Docker API images use `requirements-api.txt` to avoid shipping test and training
helpers in production runtime containers.

## Project Layout

```text
data/                         CSV datasets
models/                       trained model artifacts, ignored by git
src/app.py                    FastAPI application
src/train.py                  training entrypoint
src/data_loader.py            CSV loading and train/test split
tests/                        unit and API tests
monitoring/prometheus.yml     Prometheus scrape config
monitoring/grafana/           Grafana datasource and dashboard provisioning
docker-compose.yml            local API, MLflow, Prometheus, Grafana stack
```

## Local Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Runtime settings can be overridden with environment variables. See
`.env.example` for the supported names.

Place a dataset at `data/train.csv` with at least:

```csv
text,target
Fire in the building!,1
Sunny day in the park,0
```

## Train

```bash
python -m src.train
```

Training saves the local model to `models/disaster_classification` and logs
parameters, metrics, and the model artifact to MLflow using the URI in
`config.yaml`.

## Run API

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Useful endpoints:

- Swagger: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>
- Readiness: <http://localhost:8000/ready>
- Metrics: <http://localhost:8000/metrics>

Prediction example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Fire in the building! Help!"}'
```

Example response:

```json
{
  "prediction": 1,
  "confidence": 0.94,
  "label": "disaster"
}
```

If the model is not trained yet, `/health` still works and `/predict` returns
`503` with instructions to train the model or set `MODEL_URI`. `/ready` checks
whether the model can be loaded and returns `503` until inference is available.

## Run Local Prediction

After training, classify text without starting the API:

```bash
python -m src.predict "Fire in the building! Help!"
```

## Run Full Stack

```bash
docker compose up --build
```

Services:

- API: <http://localhost:8000>
- MLflow: <http://localhost:5000>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000> (`admin` / `admin`)

Grafana is provisioned with a Prometheus datasource and an `MLOps Sentinel API`
dashboard for request rate, p95 latency, and 5xx errors.

To register the local model in MLflow after training:

```bash
docker compose --profile register up --build register-model
```

## Tests

```bash
pytest -v
black --check src tests register_model.py
flake8 src tests register_model.py --count --max-complexity=10 --statistics
```

Model smoke tests are skipped automatically when `models/disaster_classification`
is absent, which keeps CI fast and deterministic.
