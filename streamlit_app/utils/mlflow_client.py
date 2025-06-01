import mlflow
from mlflow.tracking import MlflowClient as MLflowTrackingClient


class MLflowClient:
    def __init__(self, tracking_uri="http://localhost:5000"):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MLflowTrackingClient()

    def get_experiments(self):
        """Récupère toutes les expériences"""
        try:
            experiments = self.client.search_experiments()
            return [{"name": exp.name, "id": exp.experiment_id} for exp in experiments]
        except Exception as e:
            print(f"Erreur récupération expériences: {e}")
            return []

    def get_runs(self, experiment_name, max_results=50):
        """Récupère les runs d'une expérience"""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                return pd.DataFrame()

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"],
            )
            return runs
        except Exception as e:
            print(f"Erreur récupération runs: {e}")
            return pd.DataFrame()

    def get_run_details(self, run_id):
        """Récupère les détails d'un run"""
        try:
            run = self.client.get_run(run_id)
            return {
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
            }
        except Exception as e:
            print(f"Erreur détails run: {e}")
            return {"params": {}, "metrics": {}}

    def get_model_versions(self, model_name="CasablancaTrafficModel"):
        """Récupère les versions d'un modèle"""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            return [{"version": v.version, "stage": v.current_stage} for v in versions]
        except Exception as e:
            print(f"Erreur versions modèle: {e}")
            return []

    def get_latest_model(self, model_name="CasablancaTrafficModel", stage="Production"):
        """Récupère le dernier modèle en production"""
        try:
            model_version = self.client.get_latest_versions(model_name, stages=[stage])
            if model_version:
                return model_version[0]
            return None
        except Exception as e:
            print(f"Erreur récupération modèle latest: {e}")
            return None

    def get_experiment_metrics_summary(self, experiment_name):
        """Récupère un résumé des métriques d'une expérience"""
        try:
            runs_df = self.get_runs(experiment_name)
            if runs_df.empty:
                return {}

            metrics_summary = {}
            metric_columns = [
                col for col in runs_df.columns if col.startswith("metrics.")
            ]

            for metric_col in metric_columns:
                metric_name = metric_col.replace("metrics.", "")
                values = runs_df[metric_col].dropna()
                if not values.empty:
                    metrics_summary[metric_name] = {
                        "best": float(values.max()),
                        "worst": float(values.min()),
                        "average": float(values.mean()),
                        "latest": float(values.iloc[0]) if len(values) > 0 else 0.0,
                    }

            return metrics_summary
        except Exception as e:
            print(f"Erreur résumé métriques: {e}")
            return {}

    def is_mlflow_available(self):
        """Vérifie si MLflow est disponible"""
        try:
            self.client.search_experiments()
            return True
        except Exception as e:
            print(f"MLflow indisponible: {e}")
            return False
