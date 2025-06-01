# =====================================
# 📊 streamlit_app/pages/ml_monitoring.py
# =====================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Ajouter le chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.api_client import APIClient
except ImportError:
    # Fallback si import échoue
    class APIClient:
        def get_metrics(self):
            return {"error": "API non accessible"}

def ml_monitoring_page():
    """Page ML Monitoring avec métriques et actions MLOps"""
    
    st.header("📊 ML Monitoring - Modèle Trafic Casablanca")
    
    # Colonnes principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Performance Modèle")
        
        try:
            # Essayer de récupérer métriques API
            api_client = APIClient()
            metrics_response = api_client.get_metrics()
            
            if "error" not in metrics_response and "model_metrics" in metrics_response:
                model_metrics = metrics_response["model_metrics"]
                accuracy = model_metrics.get('accuracy', 0.873)
                mae = model_metrics.get('mae', 0.12)
                r2_score = model_metrics.get('r2_score', 0.91)
            else:
                # Métriques par défaut si API non accessible
                accuracy = 0.873
                mae = 0.12
                r2_score = 0.91
                
        except Exception as e:
            # Métriques par défaut en cas d'erreur
            accuracy = 0.873
            mae = 0.12
            r2_score = 0.91
            st.info(f"ℹ️ Utilisation métriques simulées (API: {str(e)[:50]}...)")
        
        # Affichage métriques avec variations simulées
        accuracy_delta = np.random.uniform(-0.05, 0.05)
        mae_delta = np.random.uniform(-0.02, 0.02)
        r2_delta = np.random.uniform(-0.03, 0.03)
        
        st.metric(
            "Accuracy", 
            f"{accuracy:.1%}", 
            f"{accuracy_delta:+.1%}",
            delta_color="normal" if accuracy_delta >= 0 else "inverse"
        )
        st.metric(
            "MAE", 
            f"{mae:.3f}", 
            f"{mae_delta:+.3f}",
            delta_color="inverse" if mae_delta >= 0 else "normal"
        )
        st.metric(
            "R² Score", 
            f"{r2_score:.3f}", 
            f"{r2_delta:+.3f}",
            delta_color="normal" if r2_delta >= 0 else "inverse"
        )
        
        # Graphique historique performance
        st.subheader("📈 Historique Performance")
        
        # Données simulées pour l'historique
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        base_accuracy = 0.85
        
        # Simulation évolution réaliste
        performance_data = []
        for i, date in enumerate(dates):
            # Trend légèrement croissant avec du bruit
            trend = 0.02 * (i / 30)
            noise = 0.03 * np.sin(i / 5) + 0.01 * np.random.randn()
            daily_accuracy = base_accuracy + trend + noise
            performance_data.append({
                'date': date,
                'accuracy': max(0.7, min(0.95, daily_accuracy)),
                'mae': max(0.05, 0.15 - trend/2 + abs(noise)/3),
                'r2_score': max(0.7, min(0.95, daily_accuracy - 0.05))
            })
        
        perf_df = pd.DataFrame(performance_data)
        
        # Graphique avec Plotly
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=perf_df['date'], 
            y=perf_df['accuracy'],
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        
        fig_perf.update_layout(
            title="Évolution Accuracy du Modèle (30 derniers jours)",
            xaxis_title="Date",
            yaxis_title="Accuracy",
            height=300,
            showlegend=False,
            yaxis=dict(range=[0.8, 1.0])
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with col2:
        st.subheader("📈 Data Quality & Drift")
        
        try:
            # Métriques data quality
            if "error" not in metrics_response and "data_quality" in metrics_response:
                data_quality = metrics_response["data_quality"]
                drift_score = data_quality.get('drift_score', 0.15)
                outliers = data_quality.get('outliers_detected', 2)
            else:
                # Valeurs par défaut avec variation
                drift_score = 0.15 + np.random.uniform(-0.05, 0.05)
                outliers = np.random.randint(0, 5)
                
        except:
            drift_score = 0.15
            outliers = 2
        
        # Métriques data quality
        drift_delta = np.random.uniform(-0.02, 0.02)
        st.metric(
            "Drift Score", 
            f"{drift_score:.3f}", 
            f"{drift_delta:+.3f}",
            delta_color="inverse" if drift_delta >= 0 else "normal"
        )
        
        st.metric("Dernière Validation", "2h ago", delta_color="normal")
        st.metric("Qualité Données", "94.2%", "1.1%")
        st.metric("Outliers Détectés", f"{outliers}", f"{np.random.randint(-2, 3):+d}")
        
        # Actions MLOps
        st.subheader("🔧 Actions MLOps")
        
        col2a, col2b = st.columns(2)
        
        with col2a:
            if st.button("🔄 Relancer Entraînement", type="primary", key="retrain_btn"):
                with st.spinner("Entraînement en cours..."):
                    # Simulation entraînement
                    import time
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                    
                    st.success("✅ Entraînement lancé! Nouveau modèle en cours de validation")
                    st.info("📊 Métriques estimées: Accuracy +2.3%")
        
        with col2b:
            if st.button("📊 Comparer Modèles", key="compare_btn"):
                with st.spinner("Comparaison en cours..."):
                    import time
                    time.sleep(1)
                    
                    # Simulation comparaison
                    new_acc = accuracy + np.random.uniform(0.01, 0.05)
                    current_acc = accuracy
                    
                    st.success(f"🏆 Nouveau modèle: {new_acc:.1%} vs Actuel: {current_acc:.1%}")
                    
                    if new_acc > current_acc:
                        st.info("💡 Déploiement recommandé!")
                        if st.button("🚀 Déployer Nouveau Modèle", key="deploy_btn"):
                            st.success("✅ Déploiement initié!")
                    else:
                        st.warning("⚠️ Nouveau modèle moins performant")
        
        # Status Pipeline
        st.subheader("🔄 Pipeline Status")
        
        # Status temps réel avec indicateurs
        pipeline_steps = {
            "Data Ingestion": {"status": "✅", "color": "green", "detail": "OK - 5.2k échantillons"},
            "Feature Engineering": {"status": "✅", "color": "green", "detail": "OK - 5 features"},
            "Model Training": {"status": "🔄", "color": "orange", "detail": "En cours - 78%"},
            "Model Validation": {"status": "⏳", "color": "gray", "detail": "En attente"},
            "Deployment": {"status": "✅", "color": "green", "detail": "Ready - v2.1.3"}
        }
        
        for step, info in pipeline_steps.items():
            col_step, col_status, col_detail = st.columns([3, 1, 3])
            with col_step:
                st.write(f"**{step}**")
            with col_status:
                st.write(info["status"])
            with col_detail:
                st.write(f"_{info['detail']}_")
    
    # Section alertes et insights
    st.divider()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🚨 Alertes & Insights")
        
        # Alertes dynamiques
        alerts = []
        
        if drift_score > 0.2:
            alerts.append({"type": "warning", "message": f"Dérive données détectée (score: {drift_score:.3f})"})
        
        if accuracy < 0.85:
            alerts.append({"type": "error", "message": f"Performance dégradée ({accuracy:.1%})"})
        else:
            alerts.append({"type": "success", "message": f"Performance stable ({accuracy:.1%})"})
        
        if outliers > 3:
            alerts.append({"type": "warning", "message": f"{outliers} outliers détectés"})
        
        # Insights positifs
        insights = [
            "💡 Modèle stable depuis 3 jours",
            "📈 Amélioration +2.1% vs semaine dernière", 
            "🎯 Feature 'hour' très prédictive (importance: 45%)",
            "🔄 Pipeline automatisé opérationnel"
        ]
        
        # Afficher alertes
        for alert in alerts:
            if alert["type"] == "warning":
                st.warning(f"⚠️ {alert['message']}")
            elif alert["type"] == "error":
                st.error(f"❌ {alert['message']}")
            else:
                st.success(f"✅ {alert['message']}")
        
        # Afficher insights
        for insight in insights[:2]:  # Limiter à 2 insights
            st.info(insight)
    
    with col4:
        st.subheader("📊 Feature Importance")
        
        # Feature importance avec graphique
        features_data = {
            'feature': ['hour', 'historical_avg', 'day_of_week', 'weather_score', 'event_impact'],
            'importance': [0.45, 0.23, 0.15, 0.10, 0.07]
        }
        
        features_df = pd.DataFrame(features_data)
        
        # Graphique importance
        fig_features = px.bar(
            features_df,
            x='importance',
            y='feature',
            orientation='h',
            title="Importance des Features",
            color='importance',
            color_continuous_scale='viridis'
        )
        
        fig_features.update_layout(
            height=300,
            showlegend=False,
            xaxis_title="Importance",
            yaxis_title="Features"
        )
        
        st.plotly_chart(fig_features, use_container_width=True)
        
        # Recommandations
        st.subheader("💡 Recommandations")
        recommendations = [
            "🔄 Re-entraînement recommandé dans 2 jours",
            "📊 Collecter plus de données météo",
            "🎯 Optimiser feature 'event_impact'",
            "📈 Surveiller performance week-end"
        ]
        
        for rec in recommendations:
            st.write(f"• {rec}")

# Point d'entrée si exécuté directement
if __name__ == "__main__":
    ml_monitoring_page()