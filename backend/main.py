import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.monitor import ModelMonitor
from backend.predictor import TrafficPredictor
import numpy as np
app = FastAPI(title="Traffic Flow API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation
predictor = TrafficPredictor()
monitor = ModelMonitor()


@app.get("/")
def root():
    return {"message": "Traffic Flow MLOps API", "status": "running"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": predictor.is_loaded()}


@app.get("/predict")
def predict():
    """Prédictions trafic pour démo"""
    # Données factices pour la démo
    sample_data = np.random.rand(4, 5)  # 4 zones, 5 features
    predictions = predictor.predict(sample_data)

    # Log pour monitoring
    monitor.log_prediction(predictions)

    return {
        "predictions": predictions.tolist(),
        "zones": ["Centre-ville", "Maari", "Anfa", "Sidi Bernoussi"],
        "timestamp": "2024-01-15T10:30:00Z",
    }


@app.get("/metrics")
def get_metrics():
    """Métriques modèle"""
    return monitor.get_current_metrics()


@app.post("/retrain")
def trigger_retrain():
    """Déclenche re-entraînement"""
    return {"message": "Retraining triggered", "status": "pending"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
