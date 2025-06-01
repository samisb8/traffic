import argparse
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def load_data():
    """Charge ou génère données de trafic"""
    data_path = Path("data/traffic_data.csv")

    if data_path.exists():
        try:
            print("📊 Chargement données existantes")
            df = pd.read_csv(data_path)
            if len(df) > 0 and all(
                col in df.columns
                for col in [
                    "hour",
                    "day_of_week",
                    "weather_score",
                    "event_impact",
                    "historical_avg",
                    "traffic_level",
                ]
            ):
                print(f"✅ Données valides chargées: {len(df)} échantillons")
                return df
            else:
                print("⚠️ Données invalides, régénération...")
                return generate_casablanca_traffic_data()
        except Exception as e:
            print(f"⚠️ Erreur lecture: {e}, régénération...")
            return generate_casablanca_traffic_data()
    else:
        print("📊 Génération nouvelles données Casablanca")
        return generate_casablanca_traffic_data()


def generate_casablanca_traffic_data():
    """Génère données réalistes de trafic pour Casablanca"""
    print("🇲🇦 Génération données trafic Casablanca...")

    np.random.seed(42)

    # Zones stratégiques de Casablanca
    casablanca_zones = {
        "Centre_Ville": {
            "business_factor": 1.2,
            "rush_multiplier": 1.8,
            "weekend_factor": 0.6,
            "base_traffic": 0.65,
        },
        "Maarif": {
            "business_factor": 1.1,
            "rush_multiplier": 1.6,
            "weekend_factor": 0.8,
            "base_traffic": 0.55,
        },
        "Anfa": {
            "business_factor": 0.9,
            "rush_multiplier": 1.4,
            "weekend_factor": 0.9,
            "base_traffic": 0.45,
        },
        "Sidi_Bernoussi": {
            "business_factor": 1.3,
            "rush_multiplier": 1.7,
            "weekend_factor": 0.4,
            "base_traffic": 0.6,
        },
        "Hay_Hassani": {
            "business_factor": 0.8,
            "rush_multiplier": 1.5,
            "weekend_factor": 0.7,
            "base_traffic": 0.4,
        },
        "Ain_Sebaa": {
            "business_factor": 1.4,
            "rush_multiplier": 1.9,
            "weekend_factor": 0.3,
            "base_traffic": 0.7,
        },
    }

    # Génération sur 60 jours pour avoir suffisamment de données
    data = []
    start_date = datetime.now() - timedelta(days=60)

    for day in range(60):
        current_date = start_date + timedelta(days=day)
        day_of_week = current_date.weekday()  # 0=Lundi, 6=Dimanche

        # Facteurs spéciaux pour Casablanca
        is_friday = day_of_week == 4  # Vendredi = jour de prière
        is_weekend = day_of_week >= 5  # Samedi-Dimanche
        is_ramadan = day % 30 < 3  # Simulation période Ramadan (3 jours sur 30)

        for hour in range(24):
            for zone_name, zone_info in casablanca_zones.items():

                # === PATTERNS HORAIRES CASABLANCA ===
                # Heures de pointe matinales (7h-9h)
                morning_rush = 1.0 + 0.8 * np.exp(-((hour - 8) ** 2) / 2)

                # Heures de pointe du soir (17h-19h)
                evening_rush = 1.0 + 0.9 * np.exp(-((hour - 18) ** 2) / 2)

                # Pause déjeuner (12h-14h)
                lunch_factor = 1.0 + 0.3 * np.exp(-((hour - 13) ** 2) / 4)

                # Trafic nocturne très faible
                if hour >= 23 or hour <= 5:
                    night_factor = 0.1 + 0.1 * np.random.random()
                else:
                    night_factor = 1.0

                # === FACTEURS MÉTÉO CASABLANCA ===
                # Météo océanique: plus de pluie en hiver
                month = (day // 30) % 12
                if month in [0, 1, 2, 10, 11]:  # Hiver
                    rain_probability = 0.3
                    weather_impact = np.random.choice(
                        [0.7, 1.0], p=[rain_probability, 1 - rain_probability]
                    )
                else:  # Été
                    weather_impact = np.random.uniform(0.9, 1.0)

                weather_score = weather_impact

                # === ÉVÉNEMENTS SPÉCIAUX CASABLANCA ===
                event_impact = 0.0

                # Matchs Raja/Wydad (stade Mohammed V)
                if np.random.random() < 0.05:  # 5% chance match important
                    if 19 <= hour <= 23:  # Soir
                        event_impact = 0.4

                # Festivals/événements culturels
                if np.random.random() < 0.02:  # 2% chance événement
                    event_impact = 0.3

                # Manifestations/grèves (rare mais impact fort)
                if np.random.random() < 0.01:  # 1% chance
                    event_impact = 0.6

                # === CALCUL TRAFIC FINAL ===
                base_traffic = zone_info["base_traffic"]

                # Application des facteurs
                traffic_level = base_traffic
                traffic_level *= morning_rush * evening_rush * lunch_factor
                traffic_level *= zone_info["business_factor"]
                traffic_level *= night_factor

                # Ajustements spéciaux Casablanca
                if is_weekend:
                    traffic_level *= zone_info["weekend_factor"]

                if is_friday and 11 <= hour <= 14:  # Prière du vendredi
                    traffic_level *= 1.3

                if is_ramadan:
                    if 4 <= hour <= 6:  # Sahur
                        traffic_level *= 1.4
                    elif 17 <= hour <= 20:  # Iftar
                        traffic_level *= 1.6
                    else:
                        traffic_level *= 0.8

                # Impact météo et événements
                traffic_level *= weather_score
                traffic_level += event_impact

                # Bruit réaliste
                traffic_level += np.random.normal(0, 0.05)

                # Normalisation
                traffic_level = np.clip(traffic_level, 0, 1)

                # Calcul moyenne historique (simulation)
                historical_avg = base_traffic * zone_info["business_factor"] * 0.9
                if is_weekend:
                    historical_avg *= zone_info["weekend_factor"]
                historical_avg = np.clip(
                    historical_avg + np.random.normal(0, 0.02), 0, 1
                )

                # Ajout des données
                data.append(
                    {
                        "hour": hour,
                        "day_of_week": day_of_week,
                        "weather_score": round(weather_score, 3),
                        "event_impact": round(event_impact, 3),
                        "historical_avg": round(historical_avg, 3),
                        "traffic_level": round(traffic_level, 3),
                        # Données supplémentaires pour analyse (optionnel)
                        "zone": zone_name,
                        "is_weekend": is_weekend,
                        "is_ramadan": is_ramadan,
                        "timestamp": current_date.replace(hour=hour),
                    }
                )

    df = pd.DataFrame(data)

    # Garder seulement les colonnes nécessaires pour le modèle
    model_df = df[
        [
            "hour",
            "day_of_week",
            "weather_score",
            "event_impact",
            "historical_avg",
            "traffic_level",
        ]
    ].copy()

    # Sauvegarde
    Path("data").mkdir(exist_ok=True)
    model_df.to_csv("data/traffic_data.csv", index=False)

    # Sauvegarde données complètes pour analyse
    df.to_csv("data/casablanca_full_data.csv", index=False)

    print(f"✅ Données Casablanca générées: {len(model_df)} échantillons")
    print(f"📊 Zones couvertes: {df['zone'].nunique()}")
    print(
        f"📅 Période: {df['timestamp'].min().date()} à {df['timestamp'].max().date()}"
    )
    print(f"🚗 Trafic moyen: {model_df['traffic_level'].mean():.3f}")
    print(
        f"⚡ Heures de pointe détectées: {(model_df['traffic_level'] > 0.8).sum()} cas"
    )

    return model_df


def cleanup_mlflow_runs():
    """Nettoie tous les runs MLflow ouverts"""
    try:
        while mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass


def train_model(test_mode=False):
    """Entraîne le modèle de prédiction trafic"""
    print("🤖 Début entraînement modèle...")

    # Nettoyage MLflow
    cleanup_mlflow_runs()

    # Chargement données
    df = load_data()

    print(f"📊 Dataset chargé: {len(df)} échantillons")
    print("📈 Statistiques trafic:")
    print(f"   Minimum: {df['traffic_level'].min():.3f}")
    print(f"   Maximum: {df['traffic_level'].max():.3f}")
    print(f"   Moyenne: {df['traffic_level'].mean():.3f}")
    print(f"   Médiane: {df['traffic_level'].median():.3f}")

    # Préparation features
    features = [
        "hour",
        "day_of_week",
        "weather_score",
        "event_impact",
        "historical_avg",
    ]
    X = df[features]
    y = df["traffic_level"]

    # Vérification données
    print("🔍 Vérification données:")
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"   Valeurs manquantes: {X.isnull().sum().sum()}")

    # Split train/test stratifié par niveau de trafic
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5, labels=False)
    )

    # Modèle
    if test_mode:
        model = RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=42
        )  # Rapide pour tests
        print("🧪 Mode test: modèle simplifié")
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        )
        print("🚀 Mode production: modèle optimisé")

    # Entraînement
    print("🔄 Entraînement en cours...")
    model.fit(X_train, y_train)

    # Évaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Métriques train
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    # Métriques test
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    # Accuracy approximative (1 - erreur normalisée)
    accuracy_train = max(0, 1 - mae_train)
    accuracy_test = max(0, 1 - mae_test)

    # Importance des features
    feature_importance = dict(zip(features, model.feature_importances_))

    metrics = {
        "mae_train": round(mae_train, 4),
        "mae_test": round(mae_test, 4),
        "r2_train": round(r2_train, 4),
        "r2_test": round(r2_test, 4),
        "accuracy_train": round(accuracy_train, 4),
        "accuracy_test": round(accuracy_test, 4),
        "accuracy": round(accuracy_test, 4),  # Pour compatibilité
        "mae": round(mae_test, 4),  # Pour compatibilité
        "r2_score": round(r2_test, 4),  # Pour compatibilité
        "n_samples": len(df),
        "n_features": len(features),
        "features": features,
        "feature_importance": {k: round(v, 4) for k, v in feature_importance.items()},
        "model_type": "RandomForestRegressor",
        "location": "Casablanca, Morocco",
        "training_date": datetime.now().isoformat(),
    }

    print("\n📊 Résultats d'entraînement:")
    print(f"   🎯 Accuracy Test: {accuracy_test:.4f} ({accuracy_test*100:.1f}%)")
    print(f"   📉 MAE Test: {mae_test:.4f}")
    print(f"   📈 R² Test: {r2_test:.4f}")
    print("\n🔧 Importance des features:")
    for feature, importance in sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"   {feature}: {importance:.4f}")

    # Détection overfitting
    if r2_train - r2_test > 0.1:
        print("⚠️  Possible overfitting détecté!")
    else:
        print("✅ Pas d'overfitting détecté")

    # Sauvegarde modèle
    model_path = Path("data/model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)

    # Sauvegarde métriques
    with open("data/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n💾 Modèle sauvé: {model_path}")
    print("📋 Métriques sauvées: data/model_metrics.json")

    # ========== AJOUT MLFLOW TRACKING ==========
    try:
        print("📊 Enregistrement dans MLflow...")

        # Configuration MLflow
        mlflow.set_experiment("Traffic_Prediction")

        with mlflow.start_run(
            run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            # Log des paramètres du modèle
            if test_mode:
                mlflow.log_params(
                    {
                        "n_estimators": 50,
                        "max_depth": 10,
                        "mode": "test",
                        "random_state": 42,
                    }
                )
            else:
                mlflow.log_params(
                    {
                        "n_estimators": 200,
                        "max_depth": 15,
                        "min_samples_split": 5,
                        "min_samples_leaf": 2,
                        "mode": "production",
                        "random_state": 42,
                    }
                )

            # Log des paramètres généraux
            mlflow.log_params(
                {
                    "model_type": "RandomForestRegressor",
                    "location": "Casablanca, Morocco",
                    "n_samples": len(df),
                    "n_features": len(features),
                    "zones_count": 6,
                    "test_size": 0.2,
                    "stratified": True,
                }
            )

            # Log de TOUTES les métriques
            mlflow.log_metrics(
                {
                    "accuracy_train": accuracy_train,
                    "accuracy_test": accuracy_test,
                    "accuracy": accuracy_test,  # Métrique principale
                    "mae_train": mae_train,
                    "mae_test": mae_test,
                    "mae": mae_test,
                    "r2_train": r2_train,
                    "r2_test": r2_test,
                    "r2_score": r2_test,
                    "overfitting_score": r2_train - r2_test,
                }
            )

            # Log importance des features
            for feature, importance in feature_importance.items():
                mlflow.log_metric(f"feature_importance_{feature}", importance)

            # Log du modèle sklearn
            mlflow.sklearn.log_model(
                model, "traffic_model", registered_model_name="CasablancaTrafficModel"
            )

            # Log des artefacts
            mlflow.log_artifact("data/model_metrics.json")
            if Path("data/traffic_data.csv").exists():
                mlflow.log_artifact("data/traffic_data.csv")
            if Path("data/casablanca_full_data.csv").exists():
                mlflow.log_artifact("data/casablanca_full_data.csv")

            print("✅ Modèle enregistré dans MLflow")

    except Exception as e:
        print(f"⚠️ Erreur MLflow (non bloquante): {e}")
        print("💾 Modèle sauvé localement quand même")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entraînement modèle trafic Casablanca"
    )
    parser.add_argument("--test-mode", action="store_true", help="Mode test (rapide)")
    parser.add_argument(
        "--regenerate", action="store_true", help="Forcer régénération données"
    )
    args = parser.parse_args()

    # Forcer régénération si demandé
    if args.regenerate:
        print("🔄 Régénération forcée des données...")
        df = generate_casablanca_traffic_data()

    # Configuration MLflow
    try:
        mlflow.set_experiment("Traffic_Prediction")
        print("✅ Expérience MLflow configurée")
    except Exception as e:
        print(f"⚠️ MLflow non disponible: {e}")

    # Entraînement
    try:
        metrics = train_model(test_mode=args.test_mode)
        print("\n🎉 Entraînement terminé avec succès!")
        print(f"🏆 Performance finale: {metrics['accuracy']:.1%}")
        print("📊 Vérifiez MLflow: http://localhost:5001")
    except Exception as e:
        print(f"\n💥 Erreur lors de l'entraînement: {e}")
        print("🆘 Essayez: python mlops/train.py --regenerate")
        exit(1)
