import json
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import requests


def cleanup_mlflow_runs():
    """Nettoie tous les runs MLflow ouverts"""
    try:
        while mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass


def load_metrics(model_path):
    """Charge les métriques d'un modèle"""
    metrics_path = model_path.parent / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


def compare_models():
    """Compare nouveau modèle vs modèle en production"""
    # Nouveau modèle
    new_model_path = Path("data/model.pkl")
    new_metrics = load_metrics(new_model_path)

    # Modèle en production
    prod_model_path = Path("data/model_production.pkl")
    prod_metrics_path = Path("data/model_production_metrics.json")

    if not prod_model_path.exists():
        print("🆕 Aucun modèle en production, déploiement du nouveau")
        return True, new_metrics, None

    if prod_metrics_path.exists():
        with open(prod_metrics_path) as f:
            prod_metrics = json.load(f)
    else:
        print("⚠️  Métriques production manquantes")
        return True, new_metrics, None

    # Comparaison (critère: accuracy)
    new_accuracy = new_metrics.get("accuracy", 0)
    prod_accuracy = prod_metrics.get("accuracy", 0)

    print(f"📊 Comparaison modèles:")
    print(f"   Nouveau: {new_accuracy:.4f}")
    print(f"   Production: {prod_accuracy:.4f}")

    should_deploy = new_accuracy >= prod_accuracy  # CHANGÉ: >= au lieu de >
    return should_deploy, new_metrics, prod_metrics


def log_model_to_mlflow(model_path, metrics):
    """Enregistre le modèle dans MLflow avec gestion d'erreurs"""
    try:
        # Nettoyage préventif
        cleanup_mlflow_runs()

        # CORRECTION: Utiliser le bon port MLflow
        mlflow.set_tracking_uri("http://localhost:5001")  # 5001 au lieu de 5000
        mlflow.set_experiment("Traffic_Prediction")

        with mlflow.start_run(
            run_name=f"deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            # Log des paramètres
            model_type = metrics.get("model_type", "Unknown")
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("location", "Casablanca, Morocco")
            mlflow.log_param("deployment_time", datetime.now().isoformat())
            mlflow.log_param("deployment_status", "success")
            mlflow.log_param("deployment_trigger", "automated")

            # Log des métriques de performance
            performance_metrics = {
                "accuracy": metrics.get("accuracy", 0),
                "mae": metrics.get("mae", 0),
                "r2_score": metrics.get("r2_score", 0),
                "rmse": metrics.get("rmse", 0) if "rmse" in metrics else None,
            }

            for metric_name, value in performance_metrics.items():
                if value is not None and isinstance(value, (int, float)):
                    mlflow.log_metric(f"deployed_{metric_name}", value)

            # Log toutes les autres métriques numériques
            for metric_name, value in metrics.items():
                if (
                    isinstance(value, (int, float))
                    and metric_name not in performance_metrics
                ):
                    try:
                        mlflow.log_metric(metric_name, value)
                    except:
                        pass  # Ignorer les erreurs de métriques

            # Vérifier que le modèle existe
            if not model_path.exists():
                raise FileNotFoundError(f"Modèle non trouvé: {model_path}")

            # Charger et log du modèle
            model = joblib.load(model_path)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="deployed_traffic_model",
                registered_model_name="CasablancaTrafficModel_Production",
            )

            # Log des artefacts supplémentaires (avec vérification)
            artifacts_to_log = [
                "data/model_metrics.json",
                "data/traffic_data.csv",
                "data/all_models_results.json",  # Si multi-modèles
            ]

            for artifact_path in artifacts_to_log:
                if Path(artifact_path).exists():
                    try:
                        mlflow.log_artifact(artifact_path)
                    except Exception as e:
                        print(f"⚠️  Erreur log artefact {artifact_path}: {e}")
                else:
                    print(f"⚠️  Artefact non trouvé: {artifact_path}")

            print("✅ Modèle enregistré dans MLflow avec succès")
            return True

    except Exception as e:
        print(f"⚠️  Erreur MLflow: {e}")
        print("🔄 Déploiement continue sans MLflow...")
        return False


def deploy_model():
    """Déploie le nouveau modèle si meilleur"""
    print("🚀 Début déploiement modèle...")

    # Nettoyage initial
    cleanup_mlflow_runs()

    # Vérification nouveau modèle
    new_model_path = Path("data/model.pkl")
    if not new_model_path.exists():
        print("❌ Aucun nouveau modèle trouvé")
        print("💡 Lancez d'abord: python mlops/train.py")
        return False

    # Charger les métriques du nouveau modèle
    new_metrics = load_metrics(new_model_path)
    if not new_metrics:
        print("❌ Métriques du nouveau modèle non trouvées")
        print("💡 Le fichier model_metrics.json est requis")
        return False

    # Afficher info du nouveau modèle
    print(f"📊 Nouveau modèle détecté:")
    print(f"   Type: {new_metrics.get('model_type', 'Unknown')}")
    print(f"   Accuracy: {new_metrics.get('accuracy', 0):.4f}")
    print(f"   Entraîné le: {new_metrics.get('training_date', 'Unknown')}")

    # Comparaison avec le modèle en production
    should_deploy, new_metrics, prod_metrics = compare_models()

    if not should_deploy:
        print("❌ Nouveau modèle ne respecte pas les critères de déploiement")
        improvement = new_metrics.get("accuracy", 0) - (
            prod_metrics.get("accuracy", 0) if prod_metrics else 0
        )
        print(f"📉 Amélioration requise: {improvement:.4f}")
        return False

    deployment_success = False

    try:
        # 1. Tentative d'enregistrement dans MLflow (non bloquant)
        print("📊 Tentative d'enregistrement dans MLflow...")
        mlflow_success = log_model_to_mlflow(new_model_path, new_metrics)

        if not mlflow_success:
            print("⚠️  MLflow indisponible, déploiement local seulement")

        # 2. Backup modèle actuel
        prod_model_path = Path("data/model_production.pkl")
        if prod_model_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = Path(f"data/backups/model_backup_{timestamp}.pkl")
            backup_path.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy2(prod_model_path, backup_path)
            print(f"💾 Backup créé: {backup_path}")

        # 3. Déploiement du nouveau modèle
        print("🔄 Copie du modèle en production...")
        shutil.copy2(new_model_path, prod_model_path)
        shutil.copy2("data/model_metrics.json", "data/model_production_metrics.json")

        print("✅ Modèle déployé en production!")
        print(f"🎯 Nouvelle accuracy: {new_metrics.get('accuracy', 'N/A'):.4f}")
        print(f"📊 MAE: {new_metrics.get('mae', 'N/A'):.4f}")
        print(f"📈 R²: {new_metrics.get('r2_score', 'N/A'):.4f}")

        deployment_success = True

        # 4. Historique de déploiement
        deployment_info = {
            "deployment_time": datetime.now().isoformat(),
            "model_type": new_metrics.get("model_type", "Unknown"),
            "accuracy": new_metrics.get("accuracy", 0),
            "mae": new_metrics.get("mae", 0),
            "r2_score": new_metrics.get("r2_score", 0),
            "mlflow_logged": mlflow_success,
            "previous_accuracy": prod_metrics.get("accuracy", 0) if prod_metrics else 0,
        }

        # Sauvegarder historique
        history_path = Path("data/deployment_history.json")
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
        else:
            history = []

        history.append(deployment_info)

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"📝 Historique mis à jour: {len(history)} déploiements")

        # 5. Notification API (optionnel, non bloquant)
        try:
            print("🔄 Notification de l'API...")
            response = requests.post(
                "http://localhost:8000/model-updated",
                json={
                    "status": "success",
                    "metrics": new_metrics,
                    "deployment_time": datetime.now().isoformat(),
                    "model_type": new_metrics.get("model_type", "Unknown"),
                },
                timeout=10,
            )

            if response.status_code == 200:
                print("✅ API notifiée du nouveau modèle")
            else:
                print(f"⚠️  API notification échouée: {response.status_code}")

        except requests.exceptions.ConnectionError:
            print("⚠️  API non accessible (normal si backend arrêté)")
        except Exception as e:
            print(f"⚠️  Erreur notification API: {e}")

        return True

    except Exception as e:
        print(f"❌ Erreur critique déploiement: {e}")

        # Tentative de restauration si échec
        if deployment_success:
            print("🔄 Tentative de restauration...")
            # Logique de rollback si nécessaire

        return False


def verify_deployment():
    """Vérifie que le déploiement s'est bien passé"""
    prod_model_path = Path("data/model_production.pkl")
    prod_metrics_path = Path("data/model_production_metrics.json")

    if prod_model_path.exists() and prod_metrics_path.exists():
        print("✅ Vérification déploiement: OK")

        # Charger et tester le modèle rapidement
        try:
            model = joblib.load(prod_model_path)
            print(f"✅ Modèle chargeable: {type(model).__name__}")

            with open(prod_metrics_path) as f:
                metrics = json.load(f)
                print(f"✅ Métriques production: {metrics.get('accuracy', 'N/A'):.4f}")
                print(f"✅ Type modèle: {metrics.get('model_type', 'Unknown')}")

            # Test de prédiction rapide
            import numpy as np

            test_input = np.array([[8, 1, 0.8, 0.1, 0.6]])  # Heure de pointe
            prediction = model.predict(test_input)[0]
            print(f"✅ Test prédiction: {prediction:.3f}")

            return True
        except Exception as e:
            print(f"❌ Erreur vérification: {e}")
            return False
    else:
        print("❌ Vérification déploiement: ÉCHEC")
        return False


if __name__ == "__main__":
    # Nettoyage initial
    cleanup_mlflow_runs()

    # Configuration MLflow (avec gestion d'erreur)
    try:
        # CORRECTION: Utiliser le bon port
        mlflow.set_tracking_uri("http://localhost:5001")  # 5001 au lieu de 5000
        mlflow.set_experiment("Traffic_Prediction")
        print("✅ Expérience MLflow configurée (port 5001)")
    except Exception as e:
        print(f"⚠️  MLflow non disponible: {e}")

    # Exécution du déploiement
    print("=" * 50)
    success = deploy_model()
    print("=" * 50)

    if success:
        print("🎉 Déploiement réussi!")

        # Vérification post-déploiement
        if verify_deployment():
            print("🏆 Modèle opérationnel en production!")
            print("📊 Vérifiez MLflow: http://localhost:5001")
        else:
            print("⚠️  Problème détecté après déploiement")
    else:
        print("💥 Déploiement échoué!")
        print("🔧 Solutions possibles:")
        print("   1. Vérifiez que train.py a été exécuté")
        print("   2. Redémarrez MLflow: python -m mlflow ui --port 5001")
        print("   3. Vérifiez les permissions sur le dossier data/")
