from datetime import datetime, timedelta
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import requests
from streamlit import json


def check_data_drift():
    """Détecte le drift dans les données SANS MLflow tracking imbriqué"""
    print("📊 Vérification drift des données...")

    # Données d'entraînement (référence)
    train_data_path = Path("data/traffic_data.csv")
    if not train_data_path.exists():
        print("⚠️ Données d'entraînement non trouvées")
        return {"drift_score": 0.0, "status": "unknown"}

    train_data = pd.read_csv(train_data_path)

    # Simulation données récentes (dans un vrai cas: API/DB)
    recent_data = generate_recent_data()

    # Calcul drift simple (différence moyennes)
    drift_scores = {}
    features = ["hour", "day_of_week", "weather_score", "event_impact"]

    for feature in features:
        if feature in train_data.columns and feature in recent_data.columns:
            train_mean = train_data[feature].mean()
            recent_mean = recent_data[feature].mean()
            if train_mean != 0:
                drift_scores[feature] = abs(train_mean - recent_mean) / abs(train_mean)
            else:
                drift_scores[feature] = 0.0

    overall_drift = np.mean(list(drift_scores.values())) if drift_scores else 0.0

    # Seuil d'alerte
    drift_threshold = 0.1
    status = "ok" if overall_drift < drift_threshold else "alert"

    print(
        f"📈 Drift score: {overall_drift:.4f} ({'⚠️ ALERT' if status == 'alert' else '✅ OK'})"
    )

    return {
        "drift_score": round(overall_drift, 4),
        "feature_drifts": {k: round(v, 4) for k, v in drift_scores.items()},
        "status": status,
        "threshold": drift_threshold,
        "timestamp": datetime.now().isoformat(),
    }


def generate_recent_data():
    """Génère données récentes simulées"""
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    n_samples = 100

    # Légère dérive simulée
    drift_factor = 1.1

    return pd.DataFrame(
        {
            "hour": np.random.randint(0, 24, n_samples),
            "day_of_week": np.random.randint(0, 7, n_samples),
            "weather_score": np.random.uniform(0, 1, n_samples) * drift_factor,
            "event_impact": np.random.uniform(0, 1, n_samples),
            "historical_avg": np.random.uniform(0.3, 0.9, n_samples),
        }
    )


def check_model_performance():
    """Vérifie performance du modèle en production SANS MLflow tracking imbriqué"""
    print("🤖 Vérification performance modèle...")

    try:
        # Métriques depuis l'API
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        if response.status_code == 200:
            api_metrics = response.json()
            print("✅ Métriques récupérées depuis l'API")
            return api_metrics
    except Exception:
        print("⚠️ API non accessible, utilisation métriques locales")

    # Fallback: métriques locales
    metrics_path = Path("data/model_production_metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            local_metrics = json.load(f)
            print("✅ Métriques locales chargées")
            return local_metrics

    print("⚠️ Aucune métrique disponible")
    return {"error": "Aucune métrique disponible"}


def generate_monitoring_report():
    """Génère rapport de monitoring complet avec UN SEUL run MLflow"""
    print("📋 Génération rapport monitoring...")

    # Forcer la fermeture de tous les runs
    try:
        while mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass

    # S'assurer qu'on est dans la bonne expérience
    mlflow.set_experiment("Casablanca_Traffic_Monitoring")

    # Collecte des données AVANT de démarrer MLflow
    print("🔍 Collecte des données de drift...")
    drift_info = check_data_drift()

    print("🔍 Collecte des données de performance...")
    model_perf = check_model_performance()

    # Génération du rapport
    report = {
        "timestamp": datetime.now().isoformat(),
        "data_drift": drift_info,
        "model_performance": model_perf,
        "recommendations": [],
    }

    # Recommandations automatiques
    if drift_info.get("status") == "alert":
        report["recommendations"].append(
            "🔄 Re-entraînement recommandé (drift détecté)"
        )

    if isinstance(model_perf, dict):
        # Vérifier si on a des métriques de modèle
        accuracy = None
        if "model_metrics" in model_perf:
            accuracy = model_perf["model_metrics"].get("accuracy", 0)
        elif "accuracy" in model_perf:
            accuracy = model_perf.get("accuracy", 0)

        if accuracy and accuracy < 0.8:
            report["recommendations"].append("📉 Performance dégradée, vérifier modèle")

    if not report["recommendations"]:
        report["recommendations"].append("✅ Système stable, aucune action requise")

    # Sauvegarde rapport
    report_path = Path("data/monitoring_report.json")
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"📋 Rapport sauvé: {report_path}")

    # MAINTENANT on log tout dans MLflow avec UN SEUL run
    try:
        with mlflow.start_run(
            run_name=f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            # Log toutes les métriques drift
            mlflow.log_metric("overall_drift_score", drift_info.get("drift_score", 0.0))
            for feature, score in drift_info.get("feature_drifts", {}).items():
                mlflow.log_metric(f"drift_{feature}", score)

            # Log métriques de performance si disponibles
            if isinstance(model_perf, dict) and "accuracy" in model_perf:
                mlflow.log_metric("model_accuracy", model_perf["accuracy"])

            # Log paramètres
            mlflow.log_param("drift_threshold", drift_info.get("threshold", 0.1))
            mlflow.log_param("drift_status", drift_info.get("status", "unknown"))
            mlflow.log_param("recommendations", ", ".join(report["recommendations"]))
            mlflow.log_param("monitoring_type", "full_report")
            mlflow.log_param("timestamp", datetime.now().isoformat())

            # Log du rapport comme artefact
            mlflow.log_artifact(str(report_path))

            print("✅ Rapport enregistré dans MLflow")

    except Exception as e:
        print(f"⚠️ Erreur MLflow: {e}")
        print("📋 Rapport sauvé localement quand même")

    # Affichage résumé
    print("\n📊 RÉSUMÉ MONITORING:")
    print(f"   Drift Score: {drift_info.get('drift_score', 'N/A')}")
    print(f"   Status: {drift_info.get('status', 'N/A')}")
    for rec in report["recommendations"]:
        print(f"   {rec}")

    return report


if __name__ == "__main__":
    # Nettoyage radical au début
    try:
        while mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass

    # Exécution du monitoring
    try:
        report = generate_monitoring_report()
        print("🎉 Monitoring terminé avec succès!")
    except Exception as e:
        print(f"💥 Erreur lors du monitoring: {e}")
        print("🔄 Redémarrez MLflow si le problème persiste")
