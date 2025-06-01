# =====================================
# 📱 streamlit_app/pages/dashboard.py (Version Harmonisée avec app.py)
# =====================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import random
import time
import math
import sys
from pathlib import Path

# Ajouter le dossier parent au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.api_client import APIClient
except ImportError:
    # Fallback si import échoue
    class APIClient:
        def get_predictions(self):
            return {"zones": ["Zone 1", "Zone 2", "Zone 3", "Zone 4"], "predictions": [0.6, 0.8, 0.4, 0.7]}
        def get_metrics(self):
            return {"system_metrics": {"avg_latency_ms": 120, "uptime_seconds": 3600, "total_requests": 1500}}

# Import du réseau routier depuis app.py
try:
    from app import CasablancaRoadNetwork, road_network
except ImportError:
    # Recréer la classe si import échoue
    class CasablancaRoadNetwork:
        def __init__(self):
            self.road_network = {
                "Boulevard Mohammed V": [(33.5731, -7.5898), (33.5850, -7.5950), (33.5950, -7.6000)],
                "Avenue Hassan II": [(33.5600, -7.5800), (33.5731, -7.5898), (33.5850, -7.6000)],
                "Boulevard Anfa": [(33.5650, -7.6114), (33.5700, -7.6000), (33.5750, -7.5900)],
                "Corniche Ain Diab": [(33.5870, -7.6830), (33.5900, -7.6700), (33.5930, -7.6500)],
                "Boulevard Zerktouni": [(33.5892, -7.6031), (33.5850, -7.5950), (33.5800, -7.5900)],
                "Avenue Mers Sultan": [(33.5500, -7.6200), (33.5600, -7.6100), (33.5700, -7.6000)],
                "Autoroute A3": [(33.5583, -7.6387), (33.5500, -7.6200), (33.5400, -7.6000)],
                "Boulevard Brahim Roudani": [(33.6133, -7.5300), (33.6000, -7.5400), (33.5900, -7.5500)],
            }
            
            self.traffic_incidents = {
                "Boulevard Anfa": {"type": "🚨 Accident", "severity": "high", "delay": 15, "location": (33.5680, -7.6050)},
                "Avenue Hassan II": {"type": "🚧 Travaux", "severity": "medium", "delay": 8, "location": (33.5750, -7.5950)},
                "Boulevard Mohammed V": {"type": "🚗 Embouteillage", "severity": "medium", "delay": 12, "location": (33.5800, -7.5920)},
                "Corniche Ain Diab": {"type": "🚔 Contrôle", "severity": "low", "delay": 3, "location": (33.5890, -7.6750)},
            }
    
    road_network = CasablancaRoadNetwork()

def show():
    """Page dashboard compatible avec la navigation de app.py"""
    
    # Style CSS harmonisé avec app.py
    st.markdown("""
    <style>
        .dashboard-header {
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .route-card {
            border: 2px solid #e0e0e0;
            border-radius: 15px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .route-optimal {
            border-color: #4CAF50;
            background: #f1f8e9;
        }
        .route-traffic {
            border-color: #FF9800;
            background: #fff8e1;
        }
        .route-blocked {
            border-color: #f44336;
            background: #ffebee;
        }
        .alert-danger { 
            background-color: #ffebee; 
            border-left: 4px solid #f44336; 
            padding: 10px; 
            border-radius: 5px;
            margin: 5px 0;
        }
        .alert-warning { 
            background-color: #fff8e1; 
            border-left: 4px solid #ff9800; 
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        .alert-success { 
            background-color: #e8f5e8; 
            border-left: 4px solid #4caf50; 
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        .vehicle-info {
            background: #e3f2fd;
            padding: 10px;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # En-tête du dashboard harmonisé
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="margin: 0; font-size: 2rem;">🗺️ Dashboard Navigation Avancée</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Vue d'ensemble du système de navigation intelligente - Casablanca 🇲🇦
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Si un véhicule est sélectionné dans app.py, l'afficher ici aussi
    if hasattr(st.session_state, 'selected_vehicle') and st.session_state.selected_vehicle:
        show_selected_vehicle_dashboard()
    
    # Configuration des colonnes principales
    main_col, sidebar_col = st.columns([3, 1])
    
    with main_col:
        # ===== SECTION PLANIFICATEUR HARMONISÉE =====
        st.subheader("🧭 Planificateur d'Itinéraire Avancé")
        
        # Utilisation des mêmes locations que app.py
        locations = {
            "Aéroport Mohammed V": (33.3675, -7.5883),
            "Centre-ville (Mohammed V)": (33.5731, -7.5898),
            "Maarif (Twin Center)": (33.5892, -7.6031),
            "Anfa Place": (33.5650, -7.6114),
            "Sidi Bernoussi": (33.5583, -7.6387),
            "Hay Hassani": (33.5200, -7.6400),
            "Ain Sebaa": (33.6133, -7.5300),
            "Marina Casablanca": (33.6061, -7.6261),
            "Université Hassan II": (33.5024, -7.6669),
            "Corniche Ain Diab": (33.5870, -7.6830),
            "Casa Port": (33.6020, -7.6180),
            "Derb Ghallef": (33.5500, -7.6200)
        }
        
        # Interface de planification intégrée
        route_col1, route_col2, route_col3, route_col4 = st.columns([2, 2, 1.5, 1.5])
        
        with route_col1:
            origin = st.selectbox("🚀 Point de départ", list(locations.keys()), index=1, key="dashboard_origin")
        
        with route_col2:
            destination = st.selectbox("🎯 Destination", list(locations.keys()), index=2, key="dashboard_destination")
        
        with route_col3:
            travel_mode = st.selectbox("🚶‍♂️ Mode", ["🚗 Voiture", "🚌 Bus", "🚶‍♂️ Piéton", "🚴‍♂️ Vélo"], key="dashboard_mode")
        
        with route_col4:
            route_preference = st.selectbox("⚡ Préférence", ["⚡ Rapide", "💰 Économique", "🌿 Écologique"], key="dashboard_pref")
        
        # Boutons d'action intégrés
        button_col1, button_col2, button_col3 = st.columns(3)
        
        with button_col1:
            if st.button("🔍 Calculer Routes", type="primary", use_container_width=True, key="dashboard_calc"):
                if origin != destination:
                    show_integrated_route_calculation(origin, destination, travel_mode, route_preference, locations)
                else:
                    st.error("⚠️ Veuillez sélectionner des points différents")
        
        with button_col2:
            if st.button("🗺️ Vue Carte", use_container_width=True, key="dashboard_map"):
                st.session_state.show_dashboard_map = True
        
        with button_col3:
            if st.button("🔄 Actualiser", use_container_width=True, key="dashboard_refresh"):
                st.rerun()
        
        # ===== CARTE INTELLIGENTE INTÉGRÉE =====
        st.subheader("🗺️ Vue Cartographique Intelligente")
        
        if hasattr(st.session_state, 'show_dashboard_map') and st.session_state.show_dashboard_map:
            show_integrated_smart_map(locations, origin, destination)
        else:
            show_traffic_overview_map(locations)
        
        # ===== ANALYSE TRAFIC TEMPS RÉEL =====
        st.subheader("📊 Analyse Trafic Temps Réel")
        
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["🌍 Vue Globale", "🛣️ Par Axes", "📈 Tendances"])
        
        with analysis_tab1:
            show_global_traffic_analysis()
        
        with analysis_tab2:
            show_road_analysis()
        
        with analysis_tab3:
            show_traffic_trends()
    
    with sidebar_col:
        # ===== MÉTRIQUES SYSTÈME HARMONISÉES =====
        st.subheader("⚡ Système Navigation")
        
        api = APIClient()
        metrics = api.get_metrics()
        show_navigation_system_metrics(metrics)
        
        # ===== STATUS VÉHICULES INTÉGRÉ =====
        st.subheader("🚗 Flotte Active")
        show_fleet_status()
        
        # ===== INCIDENTS HARMONISÉS =====
        st.subheader("🚨 Incidents Actifs")
        show_integrated_incidents()
        
        # ===== PRÉDICTIONS ML =====
        st.subheader("🤖 IA Navigation")
        show_ml_predictions_summary()

def show_selected_vehicle_dashboard():
    """Affiche les informations du véhicule sélectionné depuis app.py"""
    
    vehicle = st.session_state.selected_vehicle
    
    st.markdown(f"""
    <div class="vehicle-info">
        <h4>🚗 Véhicule Sélectionné: {vehicle['id']}</h4>
        <div style="display: flex; justify-content: space-between;">
            <div>
                <strong>Type:</strong> {vehicle['type']}<br>
                <strong>Position:</strong> {vehicle['current_location']}
            </div>
            <div>
                <strong>Vitesse:</strong> {vehicle['speed']} km/h<br>
                <strong>Status:</strong> 🟢 En ligne
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Si des routes sont calculées, les afficher
    if hasattr(st.session_state, 'calculated_routes') and st.session_state.calculated_routes:
        st.markdown("#### 🛣️ Routes Calculées")
        
        for i, route in enumerate(st.session_state.calculated_routes[:2]):  # Limite à 2 routes
            color_class = ["route-optimal", "route-traffic"][i] if i < 2 else "route-card"
            
            with st.container():
                st.markdown(f'<div class="route-card {color_class}" style="margin: 5px 0; padding: 10px;">', unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**{route['name']}**")
                    st.write(f"Distance: {route['distance']:.1f} km")
                
                with col2:
                    st.write(f"**{route['duration']:.0f} min**")
                    if route['incidents']:
                        st.write(f"⚠️ {len(route['incidents'])} incident(s)")
                
                st.markdown('</div>', unsafe_allow_html=True)

def show_integrated_route_calculation(origin, destination, travel_mode, preference, locations):
    """Calcul de routes intégré avec le système de app.py"""
    
    with st.spinner("🧭 Calcul des routes optimales..."):
        time.sleep(1.5)
        
        # Utilise la même logique que app.py
        start_coords = locations[origin]
        end_coords = locations[destination]
        
        # Génération de routes réalistes avec le même système
        routes = generate_dashboard_routes(start_coords, end_coords, travel_mode, preference)
        
        st.success(f"✅ {len(routes)} itinéraires trouvés de **{origin}** vers **{destination}**")
        
        # Affichage harmonisé avec app.py
        for i, route in enumerate(routes):
            color_class = ["route-optimal", "route-traffic", "route-blocked"][i] if i < 3 else "route-card"
            
            with st.container():
                st.markdown(f'<div class="route-card {color_class}">', unsafe_allow_html=True)
                
                # En-tête
                header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
                
                with header_col1:
                    st.markdown(f"### {route['icon']} {route['name']}")
                    st.write(route['description'])
                
                with header_col2:
                    st.metric("⏱️ Durée", route['duration'])
                
                with header_col3:
                    st.metric("📏 Distance", route['distance'])
                
                # Détails
                if route.get('incidents'):
                    st.warning(f"⚠️ {len(route['incidents'])} incident(s) sur le trajet")
                
                # Actions
                action_col1, action_col2 = st.columns(2)
                
                with action_col1:
                    if st.button(f"🎯 Choisir", key=f"dash_route_{i}", use_container_width=True):
                        st.balloons()
                        st.success(f"🚀 Route sélectionnée: {route['name']}")
                
                with action_col2:
                    if st.button(f"📍 Voir détails", key=f"dash_detail_{i}", use_container_width=True):
                        show_dashboard_route_details(route)
                
                st.markdown('</div>', unsafe_allow_html=True)

def generate_dashboard_routes(start_coords, end_coords, travel_mode, preference):
    """Génère des routes réalistes pour le dashboard"""
    
    # Calcul distance de base
    def distance_between_points(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) * 111
    
    base_distance = distance_between_points(start_coords, end_coords)
    
    routes = []
    
    # Route 1: Optimale
    optimal_route = road_network.calculate_real_route(start_coords, end_coords, avoid_traffic=False)
    routes.append({
        "name": "Route Optimale",
        "icon": "🟢",
        "duration": f"{optimal_route['duration']:.0f} min",
        "distance": f"{optimal_route['distance']:.1f} km",
        "incidents": optimal_route.get('incidents', []),
        "description": "Route la plus rapide selon les conditions actuelles",
        "efficiency": "95%",
        "traffic_level": "Modéré"
    })
    
    # Route 2: Alternative
    alt_route = road_network.calculate_real_route(start_coords, end_coords, avoid_traffic=True)
    routes.append({
        "name": "Route Alternative",
        "icon": "🟡",
        "duration": f"{alt_route['duration']:.0f} min",
        "distance": f"{alt_route['distance']:.1f} km",
        "incidents": [],
        "description": "Route évitant les zones de trafic dense",
        "efficiency": "87%",
        "traffic_level": "Fluide"
    })
    
    # Route 3: Problématique (si incidents)
    if optimal_route.get('incidents'):
        routes.append({
            "name": "Route Directe",
            "icon": "🔴",
            "duration": f"{optimal_route['duration'] + 15:.0f} min",
            "distance": f"{optimal_route['distance']:.1f} km",
            "incidents": optimal_route['incidents'],
            "description": f"Route directe avec {len(optimal_route['incidents'])} incident(s)",
            "efficiency": "70%",
            "traffic_level": "Dense"
        })
    
    return routes

def show_integrated_smart_map(locations, origin, destination):
    """Carte intelligente intégrée"""
    
    # Création de la carte avec véhicules et incidents
    fig = go.Figure()
    
    # Ajout des locations importantes
    for name, (lat, lon) in locations.items():
        color = "green" if name == origin else "red" if name == destination else "blue"
        size = 15 if name in [origin, destination] else 10
        
        fig.add_trace(go.Scattermapbox(
            lat=[lat],
            lon=[lon],
            mode='markers',
            marker=dict(size=size, color=color),
            text=[name],
            hovertemplate=f"<b>{name}</b><br>Lat: {lat:.4f}<br>Lon: {lon:.4f}<extra></extra>",
            name="Locations" if name not in [origin, destination] else f"{'Départ' if name == origin else 'Arrivée'}",
            showlegend=name in [origin, destination]
        ))
    
    # Ajout des incidents du réseau routier
    for road_name, incident in road_network.traffic_incidents.items():
        incident_color = {"high": "red", "medium": "orange", "low": "blue"}[incident["severity"]]
        
        fig.add_trace(go.Scattermapbox(
            lat=[incident["location"][0]],
            lon=[incident["location"][1]],
            mode='markers',
            marker=dict(size=12, color=incident_color, symbol='triangle-up'),
            text=[f"{incident['type']} - {road_name}"],
            hovertemplate=f"<b>{incident['type']}</b><br>{road_name}<br>Délai: +{incident['delay']} min<extra></extra>",
            name="Incidents",
            showlegend=False
        ))
    
    # Ligne de route si sélectionnée
    if origin and destination and origin != destination:
        origin_coords = locations[origin]
        dest_coords = locations[destination]
        
        fig.add_trace(go.Scattermapbox(
            lat=[origin_coords[0], dest_coords[0]],
            lon=[origin_coords[1], dest_coords[1]],
            mode='lines',
            line=dict(width=4, color='lime'),
            name="Route planifiée",
            hovertemplate="Route: %{text}<extra></extra>",
            text=[f"{origin} → {destination}"]
        ))
    
    # Configuration de la carte
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=33.5731, lon=-7.5898),
            zoom=10
        ),
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_traffic_overview_map(locations):
    """Carte vue d'ensemble du trafic"""
    
    # Génération données trafic par zone
    traffic_data = []
    for name, (lat, lon) in locations.items():
        traffic_level = random.uniform(0.2, 0.9)
        vehicles_count = random.randint(50, 800)
        
        traffic_data.append({
            'lat': lat,
            'lon': lon,
            'location': name,
            'traffic': traffic_level,
            'vehicles': vehicles_count,
            'avg_speed': random.randint(20, 70),
            'incidents': random.randint(0, 4)
        })
    
    df_traffic = pd.DataFrame(traffic_data)
    
    # Carte avec heatmap du trafic
    fig = px.scatter_mapbox(
        df_traffic,
        lat="lat",
        lon="lon",
        color="traffic",
        size="vehicles",
        color_continuous_scale="RdYlGn_r",
        size_max=25,
        zoom=10,
        hover_name="location",
        hover_data={
            "traffic": ":.1%",
            "vehicles": True,
            "avg_speed": True,
            "incidents": True
        },
        mapbox_style="open-street-map",
        title="Vue d'ensemble - Niveau de trafic par zone"
    )
    
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

def show_global_traffic_analysis():
    """Analyse globale du trafic"""
    
    # Métriques globales
    global_col1, global_col2, global_col3, global_col4 = st.columns(4)
    
    with global_col1:
        traffic_density = random.uniform(0.4, 0.8)
        st.metric("🌍 Densité Globale", f"{traffic_density:.1%}", 
                 delta=f"{random.uniform(-0.05, 0.05):.1%}")
    
    with global_col2:
        avg_speed = random.randint(35, 55)
        st.metric("🏃‍♂️ Vitesse Moyenne", f"{avg_speed} km/h",
                 delta=f"{random.randint(-5, 8)} km/h")
    
    with global_col3:
        total_incidents = len(road_network.traffic_incidents)
        st.metric("🚨 Incidents", total_incidents,
                 delta=f"{random.randint(-2, 3)}")
    
    with global_col4:
        efficiency = random.uniform(0.75, 0.95)
        st.metric("⚡ Efficacité", f"{efficiency:.1%}",
                 delta=f"{random.uniform(-0.02, 0.04):.1%}")
    
    # Graphique évolution trafic
    time_data = pd.DataFrame({
        'Heure': [f"{h:02d}:00" for h in range(24)],
        'Trafic': [max(0.1, 0.5 + 0.3 * np.sin((h-7)*np.pi/12) + random.uniform(-0.1, 0.1)) for h in range(24)]
    })
    
    fig_evolution = px.area(
        time_data,
        x='Heure',
        y='Trafic',
        title="📈 Évolution du trafic sur 24h",
        color_discrete_sequence=['#3b82f6']
    )
    
    # Ligne actuelle
    current_hour = datetime.now().hour
    fig_evolution.add_vline(
        x=f"{current_hour:02d}:00",
        line_dash="dash",
        line_color="red",
        annotation_text="Maintenant"
    )
    
    fig_evolution.update_layout(height=300)
    st.plotly_chart(fig_evolution, use_container_width=True)

def show_road_analysis():
    """Analyse par axes routiers"""
    
    # Données par axe routier
    road_data = []
    for road_name in road_network.road_network.keys():
        traffic_level = random.uniform(0.2, 0.9)
        
        # Vérifier s'il y a un incident
        incident_impact = 0
        if road_name in road_network.traffic_incidents:
            incident = road_network.traffic_incidents[road_name]
            impact_multiplier = {"high": 0.3, "medium": 0.15, "low": 0.05}
            incident_impact = impact_multiplier[incident["severity"]]
            traffic_level = min(0.95, traffic_level + incident_impact)
        
        road_data.append({
            'Axe': road_name.replace('Boulevard ', 'Bd ').replace('Avenue ', 'Av '),
            'Trafic': traffic_level,
            'Vitesse': random.randint(15, 70),
            'Status': '🔴' if traffic_level > 0.7 else '🟡' if traffic_level > 0.4 else '🟢',
            'Incident': '⚠️' if road_name in road_network.traffic_incidents else '✅'
        })
    
    df_roads = pd.DataFrame(road_data)
    
    # Graphique en barres
    fig_roads = px.bar(
        df_roads,
        x='Axe',
        y='Trafic',
        color='Trafic',
        color_continuous_scale='RdYlGn_r',
        title="📊 Niveau de trafic par axe routier"
    )
    
    fig_roads.update_layout(
        height=350,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig_roads, use_container_width=True)
    
    # Tableau détaillé
    st.markdown("#### 📋 Détail par axe")
    st.dataframe(
        df_roads.style.format({'Trafic': '{:.1%}'}).background_gradient(subset=['Trafic'], cmap='RdYlGn_r'),
        use_container_width=True
    )

def show_traffic_trends():
    """Tendances et prédictions trafic"""
    
    # Données historiques simulées
    dates = pd.date_range(start='2024-11-01', end='2024-12-01', freq='D')
    historical_data = pd.DataFrame({
        'Date': dates,
        'Trafic_Moyen': [0.6 + 0.1 * np.sin(i*np.pi/7) + random.uniform(-0.05, 0.05) for i in range(len(dates))],
        'Incidents': [random.randint(2, 12) for _ in range(len(dates))],
        'Vitesse_Moyenne': [45 + 10 * np.sin(i*np.pi/7) + random.randint(-5, 5) for i in range(len(dates))]
    })
    
    # Graphique tendances
    fig_trends = go.Figure()
    
    fig_trends.add_trace(go.Scatter(
        x=historical_data['Date'],
        y=historical_data['Trafic_Moyen'],
        mode='lines+markers',
        name='Trafic Moyen',
        line=dict(color='#3b82f6', width=3),
        yaxis='y'
    ))
    
    fig_trends.add_trace(go.Scatter(
        x=historical_data['Date'],
        y=historical_data['Incidents'],
        mode='lines+markers',
        name='Incidents',
        line=dict(color='#ef4444', width=2),
        yaxis='y2'
    ))
    
    fig_trends.update_layout(
        title="📈 Tendances Traffic (30 derniers jours)",
        xaxis_title="Date",
        yaxis=dict(title="Niveau Trafic", side="left", tickformat='.0%'),
        yaxis2=dict(title="Nombre Incidents", side="right", overlaying="y"),
        height=400,
        legend=dict(x=0.01, y=0.99)
    )
    
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Prédictions futures
    st.markdown("#### 🔮 Prédictions Prochaines 4h")
    
    pred_hours = [f"{(datetime.now().hour + i) % 24:02d}:00" for i in range(1, 5)]
    predictions = [random.uniform(0.3, 0.8) for _ in range(4)]
    
    pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
    
    for i, (hour, pred) in enumerate(zip(pred_hours, predictions)):
        with [pred_col1, pred_col2, pred_col3, pred_col4][i]:
            trend = "📈" if pred > 0.6 else "📉" if pred < 0.4 else "➡️"
            st.metric(f"{hour}", f"{pred:.1%}", delta=f"{trend}")

def show_navigation_system_metrics(metrics):
    """Métriques système navigation harmonisées"""
    
    # Métriques de performance du système
    system_metrics = metrics.get('system_metrics', {})
    
    # Uptime
    uptime_seconds = system_metrics.get('uptime_seconds', 3600)
    uptime_hours = uptime_seconds / 3600
    st.metric("🕐 Uptime", f"{uptime_hours:.1f}h", delta="+99.2%")
    
    # Latence API
    latency = system_metrics.get('avg_latency_ms', 120)
    st.metric("⚡ Latence API", f"{latency}ms", delta=f"{random.randint(-20, 10)}ms")
    
    # Requêtes traitées
    requests = system_metrics.get('total_requests', 1500)
    st.metric("📊 Requêtes", f"{requests:,}", delta=f"+{random.randint(50, 200)}")
    
    # Précision ML
    ml_accuracy = random.uniform(0.85, 0.95)
    st.metric("🤖 Précision ML", f"{ml_accuracy:.1%}", delta=f"+{random.uniform(0.01, 0.03):.1%}")
    
    # Status composants
    st.markdown("#### 🔧 Status Composants")
    
    components = {
        "🗺️ Moteur Routes": "✅ Opérationnel",
        "🚨 Détecteur Incidents": "✅ Actif", 
        "📡 Collecteur Données": "🔄 Synchronisation",
        "🤖 Modèle IA": "✅ Entraîné",
        "📱 API Mobile": "✅ Disponible"
    }
    
    for component, status in components.items():
        st.write(f"**{component}**: {status}")

def show_fleet_status():
    """Status de la flotte véhicules"""
    
    # Types de véhicules avec statistiques
    fleet_data = {
        "🚗 Voitures": {"active": random.randint(12000, 18000), "status": "🟢"},
        "🚌 Bus": {"active": random.randint(180, 320), "status": "🟢"},
        "🚖 Taxis": {"active": random.randint(2500, 4500), "status": "🟡"},
        "🚛 Camions": {"active": random.randint(800, 1500), "status": "🟢"},
        "🏍️ Motos": {"active": random.randint(1200, 2200), "status": "🟢"}
    }
    
    for vehicle_type, data in fleet_data.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.metric(vehicle_type, f"{data['active']:,}")
        with col2:
            st.write(data['status'])
    
    # Graphique répartition
    fleet_df = pd.DataFrame([
        {"Type": k, "Nombre": v["active"]} 
        for k, v in fleet_data.items()
    ])
    
    fig_fleet = px.pie(
        fleet_df, 
        values='Nombre', 
        names='Type',
        title="🚗 Répartition Flotte"
    )
    fig_fleet.update_layout(height=250, showlegend=False)
    st.plotly_chart(fig_fleet, use_container_width=True)
    
    # Activité récente
    st.markdown("#### 🕐 Activité Récente")
    recent_activity = [
        {"vehicle": "T-1247", "action": "Nouvel itinéraire", "time": "2 min"},
        {"vehicle": "B-856", "action": "Arrêt programmé", "time": "5 min"},
        {"vehicle": "L-432", "action": "Livraison terminée", "time": "8 min"}
    ]
    
    for activity in recent_activity:
        st.write(f"🚗 **{activity['vehicle']}**: {activity['action']} _{activity['time']}_")

def show_integrated_incidents():
    """Incidents intégrés avec le système de app.py"""
    
    # Utilisation des incidents du réseau routier
    for road_name, incident in road_network.traffic_incidents.items():
        severity_style = {
            "high": "alert-danger",
            "medium": "alert-warning", 
            "low": "alert-success"
        }[incident["severity"]]
        
        severity_icon = {
            "high": "🔴",
            "medium": "🟡",
            "low": "🟢"
        }[incident["severity"]]
        
        st.markdown(f"""
        <div class="{severity_style}">
            <strong>{severity_icon} {road_name}</strong><br>
            {incident['type']} - Délai: +{incident['delay']} min
        </div>
        """, unsafe_allow_html=True)
    
    # Actions rapides
    st.markdown("#### ⚡ Actions Rapides")
    
    action_col1, action_col2 = st.columns(2)
    
    with action_col1:
        if st.button("🚨 Signaler Incident", use_container_width=True, key="report_incident"):
            st.success("✅ Formulaire ouvert!")
    
    with action_col2:
        if st.button("🔄 Actualiser", use_container_width=True, key="refresh_incidents"):
            st.rerun()

def show_ml_predictions_summary():
    """Résumé des prédictions ML"""
    
    # Prédictions par zone
    zones = ["Centre-ville", "Maarif", "Anfa", "Corniche"]
    predictions = [random.uniform(0.3, 0.9) for _ in zones]
    
    st.markdown("#### 🔮 Prédictions 30min")
    
    for zone, pred in zip(zones, predictions):
        color = "🔴" if pred > 0.7 else "🟡" if pred > 0.4 else "🟢"
        st.write(f"{color} **{zone}**: {pred:.1%}")
    
    # Graphique mini prédictions
    pred_df = pd.DataFrame({
        'Zone': zones,
        'Prediction': predictions
    })
    
    fig_pred_mini = px.bar(
        pred_df,
        x='Zone',
        y='Prediction',
        color='Prediction',
        color_continuous_scale='RdYlGn_r',
        title="Prédictions Trafic"
    )
    fig_pred_mini.update_layout(
        height=200,
        showlegend=False,
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_pred_mini, use_container_width=True)
    
    # Recommandations IA
    st.markdown("#### 💡 Recommandations IA")
    
    recommendations = [
        "🚗 Éviter Boulevard Anfa (accident)",
        "⏰ Pic de trafic prévu à 18h",
        "🛣️ Route alternative via Zerktouni"
    ]
    
    for rec in recommendations:
        st.write(f"• {rec}")

def show_dashboard_route_details(route):
    """Détails d'une route du dashboard"""
    
    with st.expander(f"📋 Détails - {route['name']}", expanded=True):
        
        # Informations détaillées
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.write(f"**⏱️ Durée estimée:** {route['duration']}")
            st.write(f"**📏 Distance totale:** {route['distance']}")
            st.write(f"**⚡ Efficacité:** {route.get('efficiency', 'N/A')}")
        
        with detail_col2:
            st.write(f"**🚦 Niveau trafic:** {route.get('traffic_level', 'Inconnu')}")
            st.write(f"**🚨 Incidents:** {len(route.get('incidents', []))}")
            st.write(f"**📊 Score route:** {random.randint(6, 10)}/10")
        
        # Incidents détaillés
        if route.get('incidents'):
            st.markdown("#### 🚨 Incidents sur le trajet")
            for road_name, incident in route['incidents']:
                st.warning(f"**{road_name}**: {incident['type']} (+{incident['delay']} min)")
        
        # Conseils
        st.markdown("#### 💡 Conseils")
        conseils = [
            "🕐 Meilleur moment: Évitez les heures de pointe",
            "⛽ Station service disponible en cours de route",
            "📱 Activez les notifications pour les mises à jour"
        ]
        
        for conseil in conseils:
            st.info(conseil)

# ===== FONCTIONS UTILITAIRES HARMONISÉES =====

@st.cache_data(ttl=300)
def get_cached_dashboard_data():
    """Cache des données dashboard pour performance"""
    return {
        'timestamp': datetime.now(),
        'traffic_data': [random.uniform(0.2, 0.9) for _ in range(10)],
        'incidents_count': len(road_network.traffic_incidents),
        'system_status': 'operational'
    }

def initialize_dashboard_session():
    """Initialisation session dashboard"""
    if 'dashboard_preferences' not in st.session_state:
        st.session_state.dashboard_preferences = {
            'auto_refresh': True,
            'show_predictions': True,
            'preferred_view': 'overview'
        }
    
    if 'dashboard_last_update' not in st.session_state:
        st.session_state.dashboard_last_update = datetime.now()

def check_dashboard_updates():
    """Vérification mises à jour dashboard"""
    if hasattr(st.session_state, 'dashboard_last_update'):
        time_diff = (datetime.now() - st.session_state.dashboard_last_update).seconds
        
        if time_diff > 60:  # Mise à jour toutes les minutes
            st.session_state.dashboard_last_update = datetime.now()
            if st.session_state.get('dashboard_preferences', {}).get('auto_refresh', True):
                st.rerun()

def get_traffic_status_color(level):
    """Couleur selon niveau de trafic"""
    if level < 0.3:
        return "🟢", "Fluide", "#4CAF50"
    elif level < 0.6:
        return "🟡", "Modéré", "#FF9800"
    elif level < 0.8:
        return "🟠", "Dense", "#FF5722"
    else:
        return "🔴", "Très dense", "#F44336"

def calculate_eta_with_incidents(base_time, incidents):
    """Calcul ETA avec prise en compte des incidents"""
    total_delay = sum(incident.get('delay', 0) for _, incident in incidents)
    return base_time + total_delay

def format_duration_smart(minutes):
    """Formatage intelligent de la durée"""
    if minutes < 60:
        return f"{minutes:.0f} min"
    else:
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:.0f}h{mins:02.0f}"

# ===== WIDGETS AVANCÉS =====

def show_route_comparison_widget():
    """Widget de comparaison de routes"""
    
    if hasattr(st.session_state, 'calculated_routes') and len(st.session_state.calculated_routes) >= 2:
        st.markdown("#### ⚖️ Comparaison Routes")
        
        route1, route2 = st.session_state.calculated_routes[:2]
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.markdown(f"**{route1['name']}**")
            st.write(f"⏱️ {route1['duration']:.0f} min")
            st.write(f"📏 {route1['distance']:.1f} km")
        
        with comp_col2:
            st.markdown(f"**{route2['name']}**")
            st.write(f"⏱️ {route2['duration']:.0f} min")
            st.write(f"📏 {route2['distance']:.1f} km")
        
        # Recommandation
        if route1['duration'] < route2['duration']:
            st.success(f"✅ **{route1['name']}** recommandée (-{route2['duration'] - route1['duration']:.0f} min)")
        else:
            st.success(f"✅ **{route2['name']}** recommandée (-{route1['duration'] - route2['duration']:.0f} min)")

def show_weather_traffic_widget():
    """Widget météo et impact trafic"""
    
    weather_conditions = [
        {"condition": "☀️ Ensoleillé", "impact": "Aucun", "traffic_modifier": 1.0},
        {"condition": "🌧️ Pluie légère", "impact": "Faible", "traffic_modifier": 1.1},
        {"condition": "⛈️ Orage", "impact": "Modéré", "traffic_modifier": 1.3},
        {"condition": "🌫️ Brouillard", "impact": "Élevé", "traffic_modifier": 1.4}
    ]
    
    current_weather = random.choice(weather_conditions)
    
    st.markdown("#### 🌤️ Conditions Météo")
    st.write(f"**Actuel:** {current_weather['condition']}")
    st.write(f"**Impact trafic:** {current_weather['impact']}")
    
    if current_weather['traffic_modifier'] > 1.0:
        st.warning(f"⚠️ Temps de trajet majoré de {(current_weather['traffic_modifier'] - 1)*100:.0f}%")

def show_fuel_cost_estimator():
    """Estimateur coût carburant"""
    
    st.markdown("#### ⛽ Estimateur Carburant")
    
    fuel_col1, fuel_col2 = st.columns(2)
    
    with fuel_col1:
        distance = st.number_input("Distance (km)", min_value=1, max_value=100, value=10, key="fuel_distance")
        consumption = st.selectbox("Consommation", ["6L/100km", "8L/100km", "10L/100km", "12L/100km"], key="fuel_consumption")
    
    with fuel_col2:
        fuel_price = st.number_input("Prix carburant (DH/L)", min_value=10.0, max_value=20.0, value=14.5, key="fuel_price")
        
        # Calcul
        consumption_rate = float(consumption.replace('L/100km', '')) / 100
        fuel_needed = distance * consumption_rate
        total_cost = fuel_needed * fuel_price
        
        st.metric("💰 Coût estimé", f"{total_cost:.1f} DH")
        st.write(f"⛽ Carburant: {fuel_needed:.1f}L")

# Initialisation au chargement du module
initialize_dashboard_session()
check_dashboard_updates()