import time
from collections import deque


class ModelMonitor:
    def __init__(self):
        self.predictions_log = deque(maxlen=1000)
        self.start_time = time.time()
        self.request_count = 0

    def log_prediction(self, predictions):
        """Log une prédiction"""
        self.request_count += 1
        self.predictions_log.append(
            {
                "timestamp": time.time(),
                "predictions": (
                    predictions.tolist()
                    if hasattr(predictions, "tolist")
                    else predictions
                ),
                "request_id": self.request_count,
            }
        )

    def get_current_metrics(self):
        """Retourne métriques actuelles"""
        uptime = time.time() - self.start_time

        # Calcul latence moyenne (simulée)
        avg_latency = 120 + np.random.randint(-20, 30)

        # Métriques modèle (simulées pour démo)
        accuracy = 0.873 + np.random.normal(0, 0.01)
        mae = 0.12 + np.random.normal(0, 0.005)

        return {
            "model_metrics": {
                "accuracy": round(accuracy, 3),
                "mae": round(mae, 3),
                "r2_score": round(0.91 + np.random.normal(0, 0.01), 3),
            },
            "system_metrics": {
                "uptime_seconds": round(uptime, 1),
                "total_requests": self.request_count,
                "avg_latency_ms": avg_latency,
                "requests_per_minute": (
                    round(self.request_count / (uptime / 60), 1) if uptime > 0 else 0
                ),
            },
            "data_quality": {
                "drift_score": round(0.15 + np.random.normal(0, 0.02), 3),
                "missing_values": 0,
                "outliers_detected": np.random.randint(0, 5),
            },
        }
