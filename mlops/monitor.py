import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import requests

def check_data_drift():
    """Détecte le drift dans les données"""
    print("📊 Vérification drift des données...")
    
    # Données d'entraînement (référence)
    train_data_path = Path("data/traffic_data.csv")
    if not train_data_path.exists():
        print("⚠️  Données d'entraînement non trouvées")
        return {"drift_score": 0.0, "status": "unknown"}
    
    train_data = pd.read_csv(train_data_path)
    
    # Simulation données récentes (dans un vrai cas: API/DB)
    recent_data = generate_recent_data()
    
    # Calcul drift simple (différence moyennes)
    drift_scores = {}
    features = ['hour', 'day_of_week', 'weather_score', 'event_impact']
    
    for feature in features:
        train_mean = train_data[feature].mean()
        recent_mean = recent_data[feature].mean()
        drift_scores[feature] = abs(train_mean - recent_mean) / train_mean
    
    overall_drift = np.mean(list(drift_scores.values()))
    
    # Seuil d'alerte
    drift_threshold = 0.1
    status = "ok" if overall_drift < drift_threshold else "alert"
    
    print(f"📈 Drift score: {overall_drift:.4f} ({'⚠️  ALERT' if status == 'alert' else '✅ OK'})")
    
    return {
        "drift_score": round(overall_drift, 4),
        "feature_drifts": {k: round(v, 4) for k, v in drift_scores.items()},
        "status": status,
        "threshold": drift_threshold,
        "timestamp": datetime.now().isoformat()
    }

def generate_recent_data():
    """Génère données récentes simulées"""
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    n_samples = 100
    
    # Légère dérive simulée
    drift_factor = 1.1
    
    return pd.DataFrame({
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'weather_score': np.random.uniform(0, 1, n_samples) * drift_factor,
        'event_impact': np.random.uniform(0, 1, n_samples),
        'historical_avg': np.random.uniform(0.3, 0.9, n_samples)
    })

def check_model_performance():
    """Vérifie performance du modèle en production"""
    print("🤖 Vérification performance modèle...")
    
    try:
        # Métriques depuis l'API
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        if response.status_code == 200:
            api_metrics = response.json()
            print("✅ Métriques récupérées depuis l'API")
            return api_metrics
    except:
        print("⚠️  API non accessible, utilisation métriques locales")
    
    # Fallback: métriques locales
    metrics_path = Path("data/model_production_metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    
    return {"error": "Aucune métrique disponible"}

def generate_monitoring_report():
    """Génère rapport de monitoring complet"""
    print("📋 Génération rapport monitoring...")
    
    # Collecte données
    drift_info = check_data_drift()
    model_perf = check_model_performance()
    
    # Rapport
    report = {
        "timestamp": datetime.now().isoformat(),
        "data_drift": drift_info,
        "model_performance": model_perf,
        "recommendations": []
    }
    
    # Recommandations automatiques
    if drift_info.get("status") == "alert":
        report["recommendations"].append("🔄 Re-entraînement recommandé (drift détecté)")
    
    if isinstance(model_perf, dict) and "model_metrics" in model_perf:
        accuracy = model_perf["model_metrics"].get("accuracy", 0)
        if accuracy < 0.8:
            report["recommendations"].append("📉 Performance dégradée, vérifier modèle")
    
    if not report["recommendations"]:
        report["recommendations"].append("✅ Système stable, aucune action requise")
    
    # Sauvegarde rapport
    report_path = Path("data/monitoring_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"📋 Rapport sauvé: {report_path}")
    
    # Affichage résumé
    print("\n📊 RÉSUMÉ MONITORING:")
    print(f"   Drift Score: {drift_info.get('drift_score', 'N/A')}")
    print(f"   Status: {drift_info.get('status', 'N/A')}")
    for rec in report["recommendations"]:
        print(f"   {rec}")
    
    return report

if __name__ == "__main__":
    report = generate_monitoring_report()
    print("🎉 Monitoring terminé!")