import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from streamlit_app.backend.predictor import TrafficPredictor
from streamlit_app.backend.monitor import ModelMonitor
from mlops.train import load_data, generate_synthetic_data
import json

# Fixtures pour les tests
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'hour': [8, 12, 18],
        'day_of_week': [0, 3, 6],
        'weather_score': [0.2, 0.5, 0.8],
        'event_impact': [0.1, 0.3, 0.9],
        'historical_avg': [0.4, 0.6, 0.8]
    })

@pytest.fixture
def predictor():
    return TrafficPredictor(model_path="tests/test_model.pkl")

@pytest.fixture
def monitor():
    return ModelMonitor()

# Tests pour predictor.py
def test_predictor_init(predictor):
    assert predictor.model is not None
    assert Path("tests/test_model.pkl").exists()

def test_predictor_predict(predictor, sample_data):
    predictions = predictor.predict(sample_data.values)
    assert len(predictions) == 3
    assert all(0 <= x <= 1 for x in predictions)

# Tests pour monitor.py
def test_monitor_logging(monitor):
    monitor.log_prediction([0.5, 0.7])
    assert monitor.request_count == 1
    assert len(monitor.predictions_log) == 1

def test_monitor_metrics(monitor):
    metrics = monitor.get_current_metrics()
    assert 0.8 <= metrics["model_metrics"]["accuracy"] <= 1.0
    assert metrics["system_metrics"]["total_requests"] >= 0

# Tests pour train.py
def test_data_loading():
    data = load_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert 'traffic_level' in data.columns

def test_synthetic_data():
    data = generate_synthetic_data()
    assert len(data) == 5000
    assert set(data.columns) == {'hour', 'day_of_week', 'weather_score', 
                               'event_impact', 'historical_avg', 'traffic_level'}

# Tests pour app.py (Streamlit)
def test_app_components():
    from streamlit_app.app import dashboard_page, ml_monitoring_page
    
    # Test des fonctions principales (simulation)
    try:
        dashboard_page()
        ml_monitoring_page()
        assert True
    except Exception as e:
        pytest.fail(f"Erreur dans les fonctions Streamlit: {str(e)}")

# Test d'intégration
def test_integration(predictor, monitor, sample_data):
    # Simulation workflow complet
    predictions = predictor.predict(sample_data.values)
    monitor.log_prediction(predictions)
    
    metrics = monitor.get_current_metrics()
    assert metrics["system_metrics"]["total_requests"] > 0
    assert isinstance(predictions, (list, np.ndarray))