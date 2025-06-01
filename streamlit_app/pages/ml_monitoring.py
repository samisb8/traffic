import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from utils.api_client import APIClient

def show():
    st.header("📊 Model Monitoring Dashboard")
    
    # Layout en deux colonnes
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Model Performance")
        
        # Récupération des métriques
        api = APIClient()
        metrics = api.get_metrics()
        
        # Affichage des métriques principales
        st.metric("Model Accuracy", 
                 f"{metrics.get('accuracy', 0.87)*100:.1f}%",
                 delta=f"{metrics.get('accuracy_delta', 0.02)*100:.1f}%")
        
        st.metric("Mean Absolute Error", 
                 f"{metrics.get('mae', 0.12):.3f}",
                 delta=f"{metrics.get('mae_delta', -0.01):.3f}")
        
        st.metric("R² Score", 
                 f"{metrics.get('r2_score', 0.91):.3f}",
                 delta=f"{metrics.get('r2_delta', 0.02):.3f}")
        
        # Historique des performances
        st.subheader("📈 Performance Over Time")
        
        # Génération de données factices pour l'historique
        dates = pd.date_range(end=datetime.today(), periods=30)
        perf_history = pd.DataFrame({
            'Date': dates,
            'Accuracy': 0.85 + 0.05 * (pd.np.random.randn(30).cumsum() * 0.1).clip(0.8, 0.95),
            'MAE': 0.15 - 0.03 * (pd.np.random.randn(30).cumsum() * 0.1).clip(-0.1, 0.1)
        })
        
        perf_fig = px.line(
            perf_history,
            x='Date',
            y=['Accuracy', 'MAE'],
            title="Model Metrics History",
            labels={'value': 'Metric Value', 'variable': 'Metric'}
        )
        st.plotly_chart(perf_fig, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Data Quality")
        
        # Métriques de qualité des données
        st.metric("Data Drift Score", 
                 f"{metrics.get('drift_score', 0.15):.3f}",
                 "⚠️ High" if metrics.get('drift_score', 0.15) > 0.1 else "✅ Normal")
        
        st.metric("Missing Values", 
                 metrics.get('missing_values', 0),
                 "⚠️ Alert" if metrics.get('missing_values', 0) > 5 else "✅ Normal")
        
        st.metric("Outliers Detected", 
                 metrics.get('outliers_detected', 2),
                 "⚠️ Alert" if metrics.get('outliers_detected', 2) > 10 else "✅ Normal")
        
        # Contrôles du modèle
        st.subheader("🛠️ Model Controls")
        
        if st.button("🔄 Check for Data Drift"):
            with st.spinner("Analyzing data drift..."):
                # Simulation d'analyse de drift
                st.success("Drift analysis complete!")
                st.warning("Moderate drift detected in feature 'hour'")
                st.info("Recommendation: Consider retraining the model")
        
        if st.button("📊 Compare Model Versions"):
            with st.spinner("Comparing models..."):
                # Simulation de comparaison
                comparison_data = pd.DataFrame({
                    'Version': ['v1.2 (Production)', 'v1.3 (Candidate)'],
                    'Accuracy': [0.872, 0.891],
                    'MAE': [0.121, 0.112],
                    'Latency': [120, 135]
                })
                st.dataframe(comparison_data)
                
                st.success("New model shows 2.1% better accuracy!")
                st.warning("Latency increased by 15ms")
        
        if st.button("🚀 Trigger Retraining"):
            with st.spinner("Starting retraining pipeline..."):
                # Simulation de réentraînement
                st.success("Retraining job started!")
                st.info("Check MLOps logs for progress")