import streamlit as st
import pandas as pd
import plotly.express as px
from utils.api_client import APIClient

def show():
    st.header("🗺️ Live Traffic Dashboard")
    
    # Configuration des colonnes
    map_col, metrics_col = st.columns([2, 1])
    
    with map_col:
        # Carte interactive avec Plotly pour plus de fonctionnalités
        st.subheader("Traffic Heatmap")
        
        # Données de localisation pour Casablanca
        locations = pd.DataFrame({
            'lat': [33.5731, 33.5892, 33.5650, 33.5583],
            'lon': [-7.5898, -7.6031, -7.6114, -7.6387],
            'location': ['Centre-ville', 'Maarif', 'Anfa', 'Sidi Bernoussi'],
            'traffic': [0.8, 0.6, 0.9, 0.4],
            'incidents': [2, 1, 3, 0]
        })
        
        # Création de la carte avec Plotly
        fig = px.scatter_mapbox(
            locations,
            lat="lat",
            lon="lon",
            color="traffic",
            size="incidents",
            color_continuous_scale=px.colors.sequential.Redor,
            size_max=15,
            zoom=11,
            hover_name="location",
            hover_data=["traffic", "incidents"],
            mapbox_style="carto-positron"
        )
        
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
        
        # Section prédictions
        st.subheader("Traffic Predictions")
        if st.button("🔄 Get Latest Predictions"):
            api = APIClient()
            predictions = api.get_predictions()
            
            if "error" in predictions:
                st.error(predictions["error"])
            else:
                pred_df = pd.DataFrame({
                    'Zone': predictions['zones'],
                    'Prediction': predictions['predictions']
                })
                st.dataframe(pred_df.style.background_gradient(cmap='Reds'))
                
                # Graphique des prédictions
                pred_fig = px.bar(
                    pred_df,
                    x='Zone',
                    y='Prediction',
                    color='Prediction',
                    color_continuous_scale='RdYlGn_r',
                    title="Traffic Level Predictions"
                )
                st.plotly_chart(pred_fig, use_container_width=True)
    
    with metrics_col:
        st.subheader("📊 Live Metrics")
        
        # Récupération des métriques
        api = APIClient()
        metrics = api.get_metrics()
        
        # Affichage des KPI
        st.metric("System Uptime", 
                 f"{metrics.get('uptime_seconds', 0)/3600:.1f} hours",
                 delta=None)
        
        st.metric("API Response Time", 
                 f"{metrics.get('latency', 120)} ms",
                 delta="-10 ms vs last hour" if metrics.get('latency', 120) < 130 else "+5 ms vs last hour")
        
        st.metric("Total Predictions", 
                 metrics.get('total_requests', 0),
                 delta=f"{metrics.get('requests_per_minute', 0):.1f}/min")
        
        # Graphique d'activité
        activity_data = pd.DataFrame({
            'Hour': [f"{h}:00" for h in range(24)],
            'Requests': [max(0, int(50 + 30 * (h-12)**2 / 144 + 10 * (h % 3))) for h in range(24)]
        })
        
        activity_fig = px.area(
            activity_data,
            x='Hour',
            y='Requests',
            title="API Requests by Hour"
        )
        st.plotly_chart(activity_fig, use_container_width=True)