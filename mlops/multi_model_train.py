# mlops/multi_model_train.py
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import (
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPRegressor


class CasablancaTrafficModelManager:
    def __init__(self):
        self.models = {}
        self.model_configs = {
            "traffic_regressor": {
                "RandomForest": RandomForestRegressor(
                    n_estimators=200, max_depth=15, random_state=42
                ),
                "XGBoost": xgb.XGBRegressor(
                    n_estimators=200, max_depth=6, random_state=42
                ),
                "LightGBM": lgb.LGBMRegressor(
                    n_estimators=200, max_depth=6, random_state=42
                ),
                "GradientBoosting": GradientBoostingRegressor(
                    n_estimators=200, max_depth=6, random_state=42
                ),
                "NeuralNetwork": MLPRegressor(
                    hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
                ),
            },
            "congestion_classifier": {
                "RandomForest": RandomForestClassifier(
                    n_estimators=200, max_depth=15, random_state=42
                ),
                "XGBoost": xgb.XGBClassifier(
                    n_estimators=200, max_depth=6, random_state=42
                ),
            },
            "anomaly_detector": {
                "IsolationForest": IsolationForest(contamination=0.1, random_state=42)
            },
        }

    def prepare_data(self, df):
        """Prépare les données pour différents types de modèles"""
        features = [
            "hour",
            "day_of_week",
            "weather_score",
            "event_impact",
            "historical_avg",
        ]
        X = df[features]

        # Pour la régression (prédiction continue)
        y_regression = df["traffic_level"]

        # Pour la classification (catégories de congestion)
        y_classification = pd.cut(
            df["traffic_level"],
            bins=[0, 0.3, 0.7, 1.0],
            labels=["Fluide", "Modéré", "Congestionné"],
        )

        return X, y_regression, y_classification

    def train_model_ensemble(self, X, y, model_type="traffic_regressor"):
        """Entraîne un ensemble de modèles"""
        results = {}

        mlflow.set_experiment("Casablanca_Multi_Models")

        for model_name, model in self.model_configs[model_type].items():
            print(f"🔄 Entraînement {model_name} pour {model_type}...")

            with mlflow.start_run(
                run_name=f"{model_type}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):
                try:
                    # Entraînement
                    if model_type == "anomaly_detector":
                        model.fit(X)
                        # Pour les détecteurs d'anomalies, pas de métriques standard
                        score = model.score_samples(X).mean()
                        metrics = {"anomaly_score": score}
                    else:
                        from sklearn.model_selection import (
                            cross_val_score,
                            train_test_split,
                        )

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )

                        model.fit(X_train, y_train)

                        # Évaluation
                        if model_type == "traffic_regressor":
                            from sklearn.metrics import mean_absolute_error, r2_score

                            y_pred = model.predict(X_test)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            accuracy = max(0, 1 - mae)

                            metrics = {
                                "mae": round(mae, 4),
                                "r2_score": round(r2, 4),
                                "accuracy": round(accuracy, 4),
                            }
                        else:  # classification
                            from sklearn.metrics import (
                                accuracy_score,
                                classification_report,
                            )

                            y_pred = model.predict(X_test)
                            acc = accuracy_score(y_test, y_pred)

                            metrics = {"classification_accuracy": round(acc, 4)}

                    # MLflow logging
                    mlflow.log_params(
                        {
                            "model_type": model_name,
                            "task_type": model_type,
                            "location": "Casablanca, Morocco",
                            "n_samples": len(X),
                            "n_features": X.shape[1],
                        }
                    )

                    mlflow.log_metrics(metrics)

                    # Log du modèle
                    if model_type != "anomaly_detector":
                        mlflow.sklearn.log_model(
                            model,
                            f"{model_type}_model",
                            registered_model_name=f"Casablanca_{model_type.title()}_{model_name}",
                        )

                    # Sauvegarde locale
                    model_path = f"data/models/{model_type}_{model_name}.pkl"
                    Path(model_path).parent.mkdir(exist_ok=True, parents=True)
                    joblib.dump(model, model_path)

                    results[f"{model_type}_{model_name}"] = {
                        "model": model,
                        "metrics": metrics,
                        "path": model_path,
                    }

                    print(f"✅ {model_name}: {metrics}")

                except Exception as e:
                    print(f"❌ Erreur {model_name}: {e}")
                    continue

        return results

    def train_zone_specific_models(self, df):
        """Entraîne des modèles spécifiques par zone"""
        mlflow.set_experiment("Casablanca_Zone_Models")

        # Charger les données complètes avec info zone
        full_data_path = Path("data/casablanca_full_data.csv")
        if not full_data_path.exists():
            print("⚠️ Données complètes non trouvées")
            return {}

        full_df = pd.read_csv(full_data_path)
        zones = full_df["zone"].unique()

        zone_models = {}
        features = [
            "hour",
            "day_of_week",
            "weather_score",
            "event_impact",
            "historical_avg",
        ]

        for zone in zones:
            print(f"🏙️ Entraînement modèle pour {zone}...")

            zone_data = full_df[full_df["zone"] == zone]
            X_zone = zone_data[features]
            y_zone = zone_data["traffic_level"]

            if len(zone_data) < 100:  # Pas assez de données
                continue

            with mlflow.start_run(
                run_name=f"zone_{zone}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):
                model = RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=42
                )
                model.fit(X_zone, y_zone)

                # Métriques zone
                from sklearn.metrics import mean_absolute_error, r2_score

                y_pred = model.predict(X_zone)
                mae = mean_absolute_error(y_zone, y_pred)
                r2 = r2_score(y_zone, y_pred)
                accuracy = max(0, 1 - mae)

                metrics = {
                    "zone_mae": round(mae, 4),
                    "zone_r2": round(r2, 4),
                    "zone_accuracy": round(accuracy, 4),
                }

                # MLflow logging
                mlflow.log_params(
                    {
                        "zone": zone,
                        "model_type": "ZoneSpecificRandomForest",
                        "zone_samples": len(zone_data),
                    }
                )
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(
                    model,
                    "zone_model",
                    registered_model_name=f"Casablanca_Zone_{zone.replace('_', '')}",
                )

                # Sauvegarde
                model_path = f"data/models/zone_{zone}.pkl"
                joblib.dump(model, model_path)

                zone_models[zone] = {
                    "model": model,
                    "metrics": metrics,
                    "path": model_path,
                }

                print(f"✅ {zone}: Accuracy {accuracy:.3f}")

        return zone_models

    def create_ensemble_predictor(self, models_results):
        """Crée un prédicteur ensemble combinant plusieurs modèles"""

        class EnsemblePredictor:
            def __init__(self, models_dict):
                self.models = {}
                for name, info in models_dict.items():
                    if "traffic_regressor" in name:
                        self.models[name] = info["model"]

            def predict(self, X):
                predictions = []
                for name, model in self.models.items():
                    pred = model.predict(X)
                    predictions.append(pred)

                # Moyenne pondérée (ou vote majoritaire)
                ensemble_pred = np.mean(predictions, axis=0)
                return ensemble_pred

            def predict_with_confidence(self, X):
                predictions = []
                for name, model in self.models.items():
                    pred = model.predict(X)
                    predictions.append(pred)

                predictions_array = np.array(predictions)
                mean_pred = np.mean(predictions_array, axis=0)
                std_pred = np.std(predictions_array, axis=0)

                return mean_pred, std_pred

        ensemble = EnsemblePredictor(models_results)

        # Sauvegarder l'ensemble
        ensemble_path = "data/models/ensemble_predictor.pkl"
        joblib.dump(ensemble, ensemble_path)

        return ensemble


def main():
    """Fonction principale d'entraînement multi-modèles"""
    print("🚀 Démarrage entraînement multi-modèles Casablanca")

    # Chargement données
    from train import load_data

    df = load_data()

    manager = CasablancaTrafficModelManager()
    X, y_reg, y_class = manager.prepare_data(df)

    all_results = {}

    # 1. Modèles de régression (prédiction trafic)
    print("\n📊 === MODÈLES DE RÉGRESSION ===")
    regression_results = manager.train_model_ensemble(X, y_reg, "traffic_regressor")
    all_results.update(regression_results)

    # 2. Modèles de classification (niveau congestion)
    print("\n📊 === MODÈLES DE CLASSIFICATION ===")
    classification_results = manager.train_model_ensemble(
        X, y_class, "congestion_classifier"
    )
    all_results.update(classification_results)

    # 3. Détecteurs d'anomalies
    print("\n📊 === DÉTECTEURS D'ANOMALIES ===")
    anomaly_results = manager.train_model_ensemble(X, y_reg, "anomaly_detector")
    all_results.update(anomaly_results)

    # 4. Modèles par zone
    print("\n📊 === MODÈLES PAR ZONE ===")
    zone_results = manager.train_zone_specific_models(df)
    all_results.update(zone_results)

    # 5. Ensemble predictor
    print("\n📊 === ENSEMBLE PREDICTOR ===")
    ensemble = manager.create_ensemble_predictor(all_results)

    # Résumé final
    print("\n🎉 === RÉSUMÉ ENTRAÎNEMENT ===")
    for name, info in all_results.items():
        print(f"✅ {name}: {info['metrics']}")

    print(f"\n📊 Total modèles entraînés: {len(all_results)}")
    print("🏆 Ensemble predictor créé!")
    print("📋 Vérifiez MLflow: http://localhost:5001")


if __name__ == "__main__":
    main()
