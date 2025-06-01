import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
from pathlib import Path
import argparse

def load_data():
    """Charge ou génère données de trafic"""
    data_path = Path("data/traffic_data.csv")
    
    if data_path.exists():
        print("📊 Chargement données existantes")
        return pd.read_csv(data_path)
    else:
        print("📊 Génération données factices")
        return generate_synthetic_data()

def generate_synthetic_data():
    """Génère données synthétiques pour démo"""
    np.random.seed(42)
    n_samples = 5000
    
    # Features: heure, jour_semaine, météo, événements, historique
    data = {
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'weather_score': np.random.uniform(0, 1, n_samples),
        'event_impact': np.random.uniform(0, 1, n_samples),
        'historical_avg': np.random.uniform(0.3, 0.9, n_samples)
    }
    
    # Target: niveau de trafic (0-1)
    # Plus de trafic aux heures de pointe et jours ouvrés
    traffic_level = (
        0.3 + 
        0.4 * np.sin(2 * np.pi * data['hour'] / 24) +  # Pattern journalier
        0.2 * (data['day_of_week'] < 5) +  # Plus de trafic en semaine
        0.1 * data['weather_score'] +  # Impact météo
        0.2 * data['event_impact'] +  # Impact événements
        0.1 * np.random.normal(0, 1, n_samples)  # Bruit
    )
    
    data['traffic_level'] = np.clip(traffic_level, 0, 1)
    
    df = pd.DataFrame(data)
    
    # Sauvegarde
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/traffic_data.csv", index=False)
    print(f"✅ Données générées: {len(df)} échantillons")
    
    return df

def train_model(test_mode=False):
    """Entraîne le modèle de prédiction trafic"""
    print("🤖 Début entraînement modèle...")
    
    # Chargement données
    df = load_data()
    
    # Préparation features
    features = ['hour', 'day_of_week', 'weather_score', 'event_impact', 'historical_avg']
    X = df[features]
    y = df['traffic_level']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modèle
    if test_mode:
        model = RandomForestRegressor(n_estimators=10, random_state=42)  # Rapide pour tests
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Entraînement
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = 1 - mae  # Approximation pour démo
    
    metrics = {
        'mae': round(mae, 4),
        'r2_score': round(r2, 4),
        'accuracy': round(accuracy, 4),
        'n_samples': len(df),
        'features': features
    }
    
    print(f"📊 Métriques: MAE={mae:.4f}, R²={r2:.4f}, Accuracy={accuracy:.4f}")
    
    # Sauvegarde modèle
    model_path = Path("data/model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    
    # Sauvegarde métriques
    with open("data/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Modèle sauvé: {model_path}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true", help="Mode test (rapide)")
    args = parser.parse_args()
    
    metrics = train_model(test_mode=args.test_mode)
    print("🎉 Entraînement terminé!")