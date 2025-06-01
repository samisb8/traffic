import requests


class APIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def get_predictions(self):
        try:
            response = requests.get(f"{self.base_url}/predict")
            return response.json()
        except Exception as e:
            print(f"Erreur API predictions: {e}")
            return {
                "error": "API non disponible",
                "predictions": [0.65, 0.78, 0.45]
            }

    def get_metrics(self):
        try:
            response = requests.get(f"{self.base_url}/metrics")
            return response.json()
        except Exception as e:
            print(f"Erreur API metrics: {e}")
            return {
                "accuracy": 0.873,
                "mae": 0.12,
                "latency": 120
            }

    def predict_traffic(self, data):
        """Prédiction du trafic avec données spécifiques"""
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Erreur prédiction trafic: {e}")
            return {
                "error": "Erreur de prédiction",
                "prediction": 0.5,
                "confidence": "low"
            }

    def get_health_status(self):
        """Vérification de l'état de l'API"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"API indisponible: {e}")
            return False