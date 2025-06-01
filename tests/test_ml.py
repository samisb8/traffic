import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import joblib

def test_traffic_predictor():
    """Test du prédicteur de trafic"""
    from streamlit_app.backend.predictor import TrafficPredictor
    
    predictor = TrafficPredictor()
    assert predictor.is_loaded()
    
    # Test prédiction
    sample_data = np.random.rand(4, 5)
    predictions = predictor.predict(sample_data)
    
    assert len(predictions) == 4
    assert all(0 <= p <= 1 for p in predictions)

def test_model_monitor():
    """Test du monitoring de modèle"""
    from streamlit_app.backend.monitor import ModelMonitor
    
    monitor = ModelMonitor()
    
    # Test logging
    predictions = np.array([0.5, 0.7, 0.3, 0.8])
    monitor.log_prediction(predictions)
    
    # Test métriques
    metrics = monitor.get_current_metrics()
    assert "model_metrics" in metrics
    assert "system_metrics" in metrics
    assert "data_quality" in metrics

def test_model_training(sample_traffic_data, temp_data_dir):
    """Test entraînement de modèle"""
    from mlops.train import train_model
    
    # Sauvegarde données test
    data_path = temp_data_dir / "traffic_data.csv"
    sample_traffic_data.to_csv(data_path, index=False)
    
    # Override data path
    import mlops.train
    original_path = Path("data/traffic_data.csv")
    
    # Test training
    metrics = train_model(test_mode=True)
    
    assert "mae" in metrics
    assert "r2_score" in metrics
    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0

def test_data_drift_detection():
    """Test détection de drift"""
    from mlops.monitor import check_data_drift
    
    drift_info = check_data_drift()
    
    assert "drift_score" in drift_info
    assert "status" in drift_info
    assert drift_info["status"] in ["ok", "alert"]