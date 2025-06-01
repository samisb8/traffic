import streamlit as st
import mlflow
import pandas as pd
import plotly.express as px
from datetime import datetime
import sys
import os

# Ajouter le chemin utils si nécessaire
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

try:
    import mlflow
    mlflow_available = True
except ImportError:
    mlflow_available = False
    st.error("❌ MLflow n'est pas installé dans l'environnement Streamlit")
    st.stop()

try:
    from streamlit_app.utils.mlflow_client import MLflowClient
    mlflow_client = MLflowClient()
except ImportError:
    st.error("❌ Impossible d'importer mlflow_client")
    st.stop()

st.set_page_config(
    page_title="🧪 Expériences MLflow",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Expériences MLflow - Trafic Casablanca")

# Sidebar - Sélection expérience
st.sidebar.header("🎯 Sélection Expérience")
experiments = mlflow_client.get_experiments()
exp_names = [exp['name'] for exp in experiments]

if exp_names:
    selected_exp = st.sidebar.selectbox("Expérience", exp_names)

    # Récupération des runs
    runs_df = mlflow_client.get_runs(selected_exp)

    if not runs_df.empty:
        st.header(f"📊 Runs - {selected_exp}")

        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        latest_run = runs_df.iloc[0]

        with col1:
            accuracy = latest_run.get('metrics.accuracy', 0)
            st.metric("🎯 Accuracy", f"{accuracy:.3f}")

        with col2:
            mae = latest_run.get('metrics.mae', 0)
            st.metric("📉 MAE", f"{mae:.4f}")

        with col3:
            r2 = latest_run.get('metrics.r2_score', 0)
            st.metric("📈 R²", f"{r2:.3f}")

        with col4:
            st.metric("🏃 Total Runs", len(runs_df))

        # Graphique évolution
        if len(runs_df) > 1:
            st.subheader("📈 Évolution des Performances")
            fig = px.line(
                runs_df.sort_values('start_time'),
                x='start_time',
                y='metrics.accuracy',
                title="Évolution Accuracy dans le temps",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

        # Tableau des runs
        st.subheader("📋 Historique des Runs")
        display_cols = [
            'start_time',
            'metrics.accuracy',
            'metrics.mae',
            'metrics.r2_score',
            'status'
        ]
        available_cols = [col for col in display_cols if col in runs_df.columns]
        st.dataframe(
            runs_df[available_cols].head(10),
            use_container_width=True
        )

        # Détails run sélectionné
        st.subheader("🔍 Détails du Run")
        run_ids = runs_df['run_id'].tolist()
        selected_run_id = st.selectbox("Sélectionner un run", run_ids)

        if selected_run_id:
            run_details = mlflow_client.get_run_details(selected_run_id)

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Paramètres:**")
                st.json(run_details['params'])

            with col2:
                st.write("**Métriques:**")
                st.json(run_details['metrics'])

    else:
        st.warning("Aucun run trouvé pour cette expérience")

else:
    st.warning("⚠️ Aucune expérience MLflow trouvée")
    st.info("Lancez d'abord un entraînement: `python mlops/train.py`")

# Bouton refresh
if st.button("🔄 Actualiser"):
    st.rerun()