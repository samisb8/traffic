import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient as MLflowTrackingClient
import os
from datetime import datetime

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
                order_by=["start_time DESC"]
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
                "end_time": run.info.end_time
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