from pathlib import Path

import joblib
import numpy as np


class TrafficPredictor:
    def __init__(self, model_path="data/model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.load_model()

    def load_model(self):
        """Charge le modèle ML"""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                print(f"✅ Modèle chargé: {self.model_path}")
            else:
                print("⚠️  Modèle non trouvé, création modèle factice")
                self.create_dummy_model()
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            self.create_dummy_model()

    def create_dummy_model(self):
        """Crée un modèle factice pour la démo"""
        from sklearn.datasets import make_regression
        from sklearn.ensemble import RandomForestRegressor

        # Données factices
        X, y = make_regression(n_samples=1000, n_features=5, noise=0.1)

        # Modèle simple
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(X, y)

        # Sauvegarde
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print("✅ Modèle factice créé et sauvé")

    def predict(self, data):
        """Prédiction trafic"""
        if self.model is None:
            return np.array([0.5, 0.7, 0.3, 0.8])  # Fallback

        try:
            # Normalise les prédictions entre 0 et 1 (niveau trafic)
            raw_pred = self.model.predict(data)
            normalized = (raw_pred - raw_pred.min()) / (raw_pred.max() - raw_pred.min())
            return np.clip(normalized, 0, 1)
        except:
            return np.array([0.5, 0.7, 0.3, 0.8])

    def is_loaded(self):
        return self.model is not None
