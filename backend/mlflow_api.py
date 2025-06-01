
import mlflow
from fastapi import APIRouter, HTTPException
from mlflow.tracking import MlflowClient

router = APIRouter(prefix="/mlflow", tags=["mlflow"])

# Configuration MLflow
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()


@router.get("/experiments")
async def get_experiments():
    """Récupère toutes les expériences"""
    try:
        experiments = client.search_experiments()
        return [
            {
                "id": exp.experiment_id,
                "name": exp.name,
                "lifecycle_stage": exp.lifecycle_stage,
            }
            for exp in experiments
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur MLflow: {str(e)}")


@router.get("/experiments/{experiment_name}/runs")
async def get_runs(experiment_name: str, limit: int = 10):
    """Récupère les runs d'une expérience"""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            raise HTTPException(status_code=404, detail="Expérience non trouvée")

        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=limit,
            order_by=["start_time DESC"],
        )

        if runs_df.empty:
            return []

        return runs_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/runs/{run_id}")
async def get_run_details(run_id: str):
    """Récupère les détails d'un run"""
    try:
        run = client.get_run(run_id)
        return {
            "run_id": run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "tags": dict(run.data.tags),
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail="Run non trouvé")


@router.get("/models")
async def get_models():
    """Récupère tous les modèles enregistrés"""
    try:
        models = client.search_registered_models()
        return [
            {
                "name": model.name,
                "description": model.description,
                "latest_version": (
                    model.latest_versions[0].version if model.latest_versions else None
                ),
            }
            for model in models
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/models/{model_name}/versions")
async def get_model_versions(model_name: str):
    """Récupère les versions d'un modèle"""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        return [
            {
                "version": version.version,
                "stage": version.current_stage,
                "status": version.status,
                "run_id": version.run_id,
            }
            for version in versions
        ]
    except Exception as e:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")


@router.post("/models/{model_name}/promote/{version}")
async def promote_model(model_name: str, version: str, stage: str = "Production"):
    """Promeut un modèle vers un stage"""
    try:
        client.transition_model_version_stage(
            name=model_name, version=version, stage=stage
        )
        return {"message": f"Modèle {model_name} v{version} promu vers {stage}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
