from fastapi.testclient import TestClient

from src import app as app_module

client = TestClient(app_module.app)


class FakePredictor:
    is_ready = True
    load_error = None

    def load(self):
        return None

    def predict(self, text):
        return {
            "prediction": 1,
            "confidence": 0.94,
            "label": "disaster",
        }


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "model_loaded" in response.json()


def test_predict(monkeypatch):
    monkeypatch.setattr(app_module, "predictor", FakePredictor())

    response = client.post("/predict", json={"text": "Fire in the building!"})

    assert response.status_code == 200
    data = response.json()
    assert data == {
        "prediction": 1,
        "confidence": 0.94,
        "label": "disaster",
    }


def test_ready(monkeypatch):
    monkeypatch.setattr(app_module, "predictor", FakePredictor())

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model_loaded": True}


def test_predict_empty():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_no_text():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_root_lists_operational_endpoints():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"
    assert response.json()["ready"] == "/ready"
