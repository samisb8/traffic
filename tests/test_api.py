import pytest
from fastapi.testclient import TestClient
from streamlit_app.backend.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test endpoint racine"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["status"] == "running"

def test_health_endpoint():
    """Test endpoint santé"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data

def test_predict_endpoint():
    """Test endpoint prédictions"""
    response = client.get("/predict")
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "zones" in data
    assert "timestamp" in data
    assert len(data["predictions"]) == len(data["zones"])

def test_metrics_endpoint():
    """Test endpoint métriques"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "model_metrics" in data
    assert "system_metrics" in data
    assert "data_quality" in data

def test_retrain_endpoint():
    """Test endpoint re-entraînement"""
    response = client.post("/retrain")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["status"] == "pending"