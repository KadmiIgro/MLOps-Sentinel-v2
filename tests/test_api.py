import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    response = client.post(
        "/predict",
        json={"text": "Fire in the building!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert data["prediction"] in [0, 1]

def test_predict_empty():
    response = client.post(
        "/predict",
        json={"text": ""}
    )
    assert response.status_code == 422  # Ожидаем ошибку валидации

def test_predict_no_text():
    response = client.post(
        "/predict",
        json={}
    )
    assert response.status_code == 422