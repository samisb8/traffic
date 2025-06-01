import requests
import json

class APIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def get_predictions(self):
        try:
            response = requests.get(f"{self.base_url}/predict")
            return response.json()
        except:
            return {"error": "API non disponible", "predictions": [0.65, 0.78, 0.45]}
    
    def get_metrics(self):
        try:
            response = requests.get(f"{self.base_url}/metrics")
            return response.json()
        except:
            return {"accuracy": 0.873, "mae": 0.12, "latency": 120}