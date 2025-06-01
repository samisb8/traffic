import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from utils.api_client import APIClient

st.set_page_config(page_title="Traffic Flow MLOps", page_icon="🚗", layout="wide")

def main():
    st.title("🚗 Traffic Flow MLOps Demo")
    st.sidebar.title("Navigation")
    
    pages = {
        "🗺️ Dashboard": "pages/dashboard.py",
        "📊 ML Monitoring": "pages/ml_monitoring.py"
    }
    
    selection = st.sidebar.selectbox("Choisir une page", list(pages.keys()))
    
    if selection == "🗺️ Dashboard":
        dashboard_page()
    elif selection == "📊 ML Monitoring":
        ml_monitoring_page()

def dashboard_page():
    st.header("🗺️ Dashboard Trafic Casablanca")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Carte simple avec points
        st.subheader("Carte Trafic")
        casablanca_data = pd.DataFrame({
            'lat': [33.5731, 33.5892, 33.5650, 33.5583],
            'lon': [-7.5898, -7.6031, -7.6114, -7.6387],
            'traffic_level': [0.8, 0.6, 0.9, 0.4],
            'location': ['Centre-ville', 'Maarif', 'Anfa', 'Sidi Bernoussi']
        })
        
        st.map(casablanca_data[['lat', 'lon']])
        
        # Prédictions temps réel
        st.subheader("Prédictions Temps Réel")
        if st.button("🔄 Actualiser Prédictions"):
            api_client = APIClient()
            predictions = api_client.get_predictions()
            st.write(predictions)
    
    with col2:
        st.subheader("📊 Métriques Live")
        st.metric("Trafic Moyen", "65%", "5%")
        st.metric("Incidents", "3", "-1")
        st.metric("Temps Réponse API", "120ms", "10ms")
        
        # Graphique simple
        traffic_data = pd.DataFrame({
            'heure': pd.date_range('08:00', periods=12, freq='H'),
            'trafic': [0.3, 0.5, 0.8, 0.9, 0.7, 0.6, 0.8, 0.9, 0.8, 0.6, 0.4, 0.3]
        })
        
        fig = px.line(traffic_data, x='heure', y='trafic', title="Évolution Trafic")
        st.plotly_chart(fig, use_container_width=True)

def ml_monitoring_page():
    st.header("📊 ML Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Performance Modèle")
        st.metric("Accuracy", "87.3%", "2.1%")
        st.metric("MAE", "0.12", "-0.03")
        st.metric("R² Score", "0.91", "0.05")
        
        # Historique performance
        perf_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'accuracy': 0.85 + 0.05 * pd.np.random.randn(30).cumsum() * 0.1
        })
        
        fig = px.line(perf_data, x='date', y='accuracy', title="Évolution Accuracy")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Data Drift")
        st.metric("Drift Score", "0.15", "0.02")
        st.metric("Dernière Validation", "2h ago", delta_color="normal")
        
        if st.button("🔄 Relancer Entraînement"):
            st.info("Entraînement lancé... Vérifiez les logs MLOps")
        
        if st.button("📊 Comparer Modèles"):
            st.success("Nouveau modèle: 89.1% vs Actuel: 87.3%")

if __name__ == "__main__":
    main()