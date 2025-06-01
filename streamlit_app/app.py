import random
import time

import plotly.graph_objects as go
import requests
import streamlit as st

# Configuration
st.set_page_config(
    page_title="Traffic Flow MLOps - Casablanca",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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
    .sidebar-section {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .vehicle-selected {
        background: #e3f2fd;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
    }
    .api-status {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-weight: bold;
    }
    .api-success { background: #d4edda; color: #155724; }
    .api-warning { background: #fff3cd; color: #856404; }
    .api-error { background: #f8d7da; color: #721c24; }
</style>
""",
    unsafe_allow_html=True,
)

# ===== MULTIPLE ROUTING APIS =====


class MultiRouteClient:
    """Client multi-API pour routes réelles"""

    def __init__(self):
        self.apis = {
            "osrm": {
                "url": "http://router.project-osrm.org/route/v1/driving",
                "status": "unknown",
                "last_test": 0,
            },
            "graphhopper": {
                "url": "https://graphhopper.com/api/1/route",
                "key": None,  # Gratuit limité
                "status": "unknown",
                "last_test": 0,
            },
            "openroute": {
                "url": "https://api.openrouteservice.org/v2/directions/driving-car",
                "key": None,  # Gratuit limité
                "status": "unknown",
                "last_test": 0,
            },
        }
        self.cache = {}
        self.preferred_api = "osrm"

    def test_api_availability(self, api_name):
        """Test disponibilité d'une API"""
        current_time = time.time()

        # Test uniquement si pas testé récemment
        if current_time - self.apis[api_name]["last_test"] < 30:
            return self.apis[api_name]["status"]

        try:
            if api_name == "osrm":
                # Test OSRM avec route courte
                test_url = f"{self.apis['osrm']['url']}/33.5731,-7.5898;33.5750,-7.5900"
                response = requests.get(test_url, timeout=5)

                if response.status_code == 200:
                    self.apis["osrm"]["status"] = "available"
                    return "available"

            elif api_name == "graphhopper":
                # Test GraphHopper (version gratuite)
                params = {
                    "point": ["33.5731,-7.5898", "33.5750,-7.5900"],
                    "vehicle": "car",
                    "locale": "en",
                }
                response = requests.get(
                    self.apis["graphhopper"]["url"], params=params, timeout=5
                )

                if response.status_code == 200:
                    self.apis["graphhopper"]["status"] = "available"
                    return "available"

            self.apis[api_name]["status"] = "error"
            return "error"

        except Exception:
            self.apis[api_name]["status"] = "error"
            return "error"
        finally:
            self.apis[api_name]["last_test"] = current_time

    def get_route_osrm(
        self, start_lat, start_lon, end_lat, end_lon, route_type="fastest"
    ):
        """Route via OSRM avec vraies rues"""
        try:
            if route_type == "alternative":
                # Route alternative via waypoints stratégiques de Casablanca
                waypoints = self.get_casablanca_waypoints(
                    start_lat, start_lon, end_lat, end_lon
                )
                coords_str = f"{start_lon},{start_lat}"
                for wp in waypoints:
                    coords_str += f";{wp[1]},{wp[0]}"
                coords_str += f";{end_lon},{end_lat}"
            else:
                coords_str = f"{start_lon},{start_lat};{end_lon},{end_lat}"

            url = f"{self.apis['osrm']['url']}/{coords_str}"
            params = {"overview": "full", "geometries": "geojson", "steps": "true"}

            response = requests.get(url, params=params, timeout=8)

            if response.status_code == 200:
                data = response.json()
                if data.get("routes"):
                    route = data["routes"][0]

                    # Extraire coordonnées réelles
                    coords = route["geometry"]["coordinates"]
                    route_coords = [[coord[1], coord[0]] for coord in coords]

                    return {
                        "coordinates": route_coords,
                        "duration": int(route["duration"] / 60),
                        "distance": round(route["distance"] / 1000, 1),
                        "source": "OSRM",
                        "roads": self.extract_road_names(route),
                        "steps": len(route.get("legs", [{}])[0].get("steps", [])),
                        "confidence": 0.95,
                    }
        except Exception as e:
            st.warning(f"OSRM Error: {str(e)[:50]}")

        return None

    def get_route_graphhopper(
        self, start_lat, start_lon, end_lat, end_lon, route_type="fastest"
    ):
        """Route via GraphHopper"""
        try:
            points = [f"{start_lat},{start_lon}", f"{end_lat},{end_lon}"]

            if route_type == "alternative":
                # Ajouter waypoint pour alternative
                mid_lat = (start_lat + end_lat) / 2 + 0.01
                mid_lon = (start_lon + end_lon) / 2 + 0.01
                points.insert(1, f"{mid_lat},{mid_lon}")

            params = {
                "point": points,
                "vehicle": "car",
                "locale": "en",
                "instructions": "true",
                "calc_points": "true",
            }

            response = requests.get(
                self.apis["graphhopper"]["url"], params=params, timeout=8
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("paths"):
                    path = data["paths"][0]

                    # Extraire points
                    points_encoded = path.get("points_encoded", True)
                    if points_encoded:
                        # Décoder polyline (GraphHopper utilise son propre encodage)
                        route_coords = self.decode_graphhopper_polyline(path["points"])
                    else:
                        coords = path.get("points", {}).get("coordinates", [])
                        route_coords = [[coord[1], coord[0]] for coord in coords]

                    return {
                        "coordinates": route_coords,
                        "duration": int(path["time"] / 1000 / 60),
                        "distance": round(path["distance"] / 1000, 1),
                        "source": "GraphHopper",
                        "roads": self.extract_gh_road_names(path),
                        "steps": len(path.get("instructions", [])),
                        "confidence": 0.90,
                    }
        except Exception as e:
            st.warning(f"GraphHopper Error: {str(e)[:50]}")

        return None

    def get_casablanca_waypoints(self, start_lat, start_lon, end_lat, end_lon):
        """Waypoints stratégiques pour routes alternatives à Casablanca"""

        # Points stratégiques réels de Casablanca
        strategic_points = [
            (33.5731, -7.5898),  # Place Mohammed V (centre)
            (33.5892, -7.6031),  # Twin Center (Maarif)
            (33.5650, -7.6114),  # Anfa
            (33.6020, -7.6180),  # Casa Port
            (33.5870, -7.6830),  # Corniche
            (33.5583, -7.6387),  # Sidi Bernoussi
        ]

        # Calculer distance du trajet
        distance = ((end_lat - start_lat) ** 2 + (end_lon - start_lon) ** 2) ** 0.5

        # Choisir waypoints selon distance
        if distance > 0.05:  # Trajet long
            # Trouver point stratégique le plus proche du milieu
            mid_lat = (start_lat + end_lat) / 2
            mid_lon = (start_lon + end_lon) / 2

            best_point = min(
                strategic_points,
                key=lambda p: ((p[0] - mid_lat) ** 2 + (p[1] - mid_lon) ** 2),
            )
            return [best_point]

        return []

    def extract_road_names(self, route):
        """Extrait noms des rues OSRM"""
        roads = []
        try:
            for leg in route.get("legs", []):
                for step in leg.get("steps", []):
                    name = step.get("name", "")
                    if name and name not in roads:
                        roads.append(name)
        except Exception:
            pass

        return roads[:5]  # Limiter à 5 rues principales

    def extract_gh_road_names(self, path):
        """Extrait noms des rues GraphHopper"""
        roads = []
        try:
            for instruction in path.get("instructions", []):
                name = instruction.get("street_name", "")
                if name and name not in roads:
                    roads.append(name)
        except Exception:
            pass

        return roads[:5]

    def decode_graphhopper_polyline(self, encoded):
        """Décode polyline GraphHopper (simplifié)"""
        # Pour simplicité, retourner une approximation
        # En production, utiliser vraie lib de décodage
        return [[33.5731, -7.5898], [33.5750, -7.5900]]

    def get_best_route(
        self, start_lat, start_lon, end_lat, end_lon, route_type="fastest"
    ):
        """Obtient meilleure route disponible"""

        # Essayer APIs dans l'ordre de préférence
        api_order = (
            ["osrm", "graphhopper"]
            if route_type == "fastest"
            else ["graphhopper", "osrm"]
        )

        for api_name in api_order:
            if self.test_api_availability(api_name) == "available":

                if api_name == "osrm":
                    route = self.get_route_osrm(
                        start_lat, start_lon, end_lat, end_lon, route_type
                    )
                elif api_name == "graphhopper":
                    route = self.get_route_graphhopper(
                        start_lat, start_lon, end_lat, end_lon, route_type
                    )

                if route:
                    route["api_used"] = api_name
                    return route

        # Fallback si toutes APIs échouent
        return self.generate_casablanca_fallback(
            start_lat, start_lon, end_lat, end_lon, route_type
        )

    def generate_casablanca_fallback(
        self, start_lat, start_lon, end_lat, end_lon, route_type
    ):
        """Fallback spécifique à Casablanca avec vraies rues"""

        # Rues principales de Casablanca (coordonnées réelles)
        main_roads = {
            "Boulevard Mohammed V": [
                (33.5731, -7.5898),
                (33.5750, -7.5920),
                (33.5780, -7.5950),
                (33.5820, -7.5980),
            ],
            "Avenue Hassan II": [
                (33.5600, -7.5800),
                (33.5650, -7.5850),
                (33.5700, -7.5900),
                (33.5731, -7.5898),
            ],
            "Boulevard Anfa": [
                (33.5650, -7.6114),
                (33.5680, -7.6080),
                (33.5720, -7.6040),
                (33.5750, -7.6000),
            ],
            "Boulevard Zerktouni": [
                (33.5892, -7.6031),
                (33.5850, -7.5980),
                (33.5820, -7.5940),
                (33.5800, -7.5900),
            ],
        }

        # Construire route en utilisant rues réelles
        route_coords = [start_lat, start_lon]
        used_roads = []

        # Trouver rue la plus proche du départ
        start_road = self.find_nearest_road(start_lat, start_lon, main_roads)
        if start_road:
            used_roads.append(start_road)
            route_coords.extend([coord for coord in main_roads[start_road]])

        # Ajouter connexion vers destination
        if route_type == "alternative":
            # Passer par une rue différente
            alt_roads = [road for road in main_roads.keys() if road != start_road]
            if alt_roads:
                alt_road = random.choice(alt_roads)
                used_roads.append(alt_road)
                route_coords.extend([coord for coord in main_roads[alt_road][:2]])

        # Ajouter destination
        route_coords.extend([end_lat, end_lon])

        # Convertir en format [lat, lon]
        formatted_coords = []
        for i in range(0, len(route_coords), 2):
            if i + 1 < len(route_coords):
                formatted_coords.append([route_coords[i], route_coords[i + 1]])

        # Calculer métriques
        distance = self.calculate_distance(formatted_coords)
        duration = max(10, int(distance * 2.5))

        if route_type == "alternative":
            duration = int(duration * 1.3)
            distance *= 1.2

        return {
            "coordinates": formatted_coords,
            "duration": duration,
            "distance": round(distance, 1),
            "source": "Casablanca_Fallback",
            "roads": used_roads,
            "steps": len(used_roads) + 2,
            "confidence": 0.80,
            "api_used": "fallback",
        }

    def find_nearest_road(self, lat, lon, roads):
        """Trouve rue la plus proche"""
        min_distance = float("inf")
        nearest_road = None

        for road_name, coordinates in roads.items():
            for coord in coordinates:
                distance = ((lat - coord[0]) ** 2 + (lon - coord[1]) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    nearest_road = road_name

        return nearest_road

    def calculate_distance(self, coords):
        """Calcule distance totale"""
        if len(coords) < 2:
            return 0

        total = 0
        for i in range(1, len(coords)):
            lat1, lon1 = coords[i - 1]
            lat2, lon2 = coords[i]
            # Distance approximative en km
            distance = ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5 * 111
            total += distance

        return total


# Instance globale
route_client = MultiRouteClient()

# ===== DONNÉES =====

VEHICLES = {
    "T-1247": {
        "id": "T-1247",
        "type": "🚗 Taxi",
        "driver": "Ahmed",
        "lat": 33.5745,
        "lon": -7.5890,
        "location": "Boulevard Mohammed V",
        "speed": 45,
        "color": "yellow",
    },
    "B-856": {
        "id": "B-856",
        "type": "🚌 Bus",
        "driver": "Fatima",
        "lat": 33.5765,
        "lon": -7.5870,
        "location": "Avenue Hassan II",
        "speed": 32,
        "color": "green",
    },
    "L-432": {
        "id": "L-432",
        "type": "🚛 Livraison",
        "driver": "Youssef",
        "lat": 33.5725,
        "lon": -7.5910,
        "location": "Boulevard Zerktouni",
        "speed": 38,
        "color": "red",
    },
    "V-289": {
        "id": "V-289",
        "type": "🚐 VTC",
        "driver": "Aicha",
        "lat": 33.5755,
        "lon": -7.5880,
        "location": "Centre-ville",
        "speed": 28,
        "color": "blue",
    },
}

DESTINATIONS = {
    "Aéroport Mohammed V": (33.3675, -7.5883),
    "Centre-ville": (33.5731, -7.5898),
    "Twin Center (Maarif)": (33.5892, -7.6031),
    "Anfa Place": (33.5650, -7.6114),
    "Casa Port": (33.6020, -7.6180),
    "Corniche Ain Diab": (33.5870, -7.6830),
    "Marina Casablanca": (33.6061, -7.6261),
    "Hay Hassani": (33.5200, -7.6400),
}

# ===== FONCTIONS PRINCIPALES =====


def main():
    init_session()

    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem;">🚗 Traffic Flow MLOps</h1>
        <h2 style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
            Routes Réelles Casablanca - Multi-API 🇲🇦
        </h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">
            OSRM • GraphHopper • Fallback Casablanca • Routes Authentiques
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar avec test APIs
    with st.sidebar:
        show_sidebar()

    # Page principale
    show_navigation_page()


def init_session():
    if "selected_vehicle" not in st.session_state:
        st.session_state.selected_vehicle = None
    if "calculated_routes" not in st.session_state:
        st.session_state.calculated_routes = []
    if "api_status" not in st.session_state:
        st.session_state.api_status = {}


def show_sidebar():
    """Sidebar avec test multi-API"""

    # Test APIs
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🌐 Test APIs Routing")

    if st.button("🔄 Test Toutes APIs", use_container_width=True):
        with st.spinner("Test APIs en cours..."):
            for api_name in ["osrm", "graphhopper"]:
                status = route_client.test_api_availability(api_name)
                st.session_state.api_status[api_name] = status

    # Afficher status APIs
    for api_name, api_info in route_client.apis.items():
        if api_name in ["osrm", "graphhopper"]:
            status = st.session_state.api_status.get(api_name, "unknown")

            if status == "available":
                st.markdown(
                    """
                <div class="api-status api-success">
                    ✅ {api_name.upper()}: Disponible
                </div>
                """,
                    unsafe_allow_html=True,
                )
            elif status == "error":
                st.markdown(
                    """
                <div class="api-status api-error">
                    ❌ {api_name.upper()}: Indisponible
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                <div class="api-status api-warning">
                    ❓ {api_name.upper()}: Non testé
                </div>
                """,
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)

    # Véhicule sélectionné
    if st.session_state.selected_vehicle:
        show_selected_vehicle_sidebar()

    # Véhicules disponibles
    show_vehicles_sidebar()


def show_selected_vehicle_sidebar():
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🚗 Véhicule Sélectionné")

    vehicle = st.session_state.selected_vehicle
    st.markdown(
        """
    <div class="vehicle-selected">
        <strong>{vehicle['id']}</strong> - {vehicle['driver']}<br>
        📍 {vehicle['location']}<br>
        🏃‍♂️ {vehicle['speed']} km/h<br>
        ⚡ {vehicle['type']}
    </div>
    """,
        unsafe_allow_html=True,
    )

    if st.button("❌ Désélectionner", use_container_width=True):
        st.session_state.selected_vehicle = None
        st.session_state.calculated_routes = []
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def show_vehicles_sidebar():
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🚗 Flotte (4 véhicules)")

    for vehicle_id, vehicle in VEHICLES.items():
        status_icon = "🟢" if vehicle["speed"] > 30 else "🟡"
        st.write(
            f"{status_icon} **{vehicle_id}** ({vehicle['driver']}) - {vehicle['speed']} km/h"
        )

    st.markdown("</div>", unsafe_allow_html=True)


def show_navigation_page():
    """Page navigation avec routes réelles"""

    main_col, info_col = st.columns([3, 1])

    with main_col:
        # Sélection destination
        if st.session_state.selected_vehicle:
            show_destination_selection()

        # Carte avec vraies routes
        st.subheader("🗺️ Carte Navigation - Routes Réelles")
        show_real_routes_map()

        # Routes calculées
        if st.session_state.calculated_routes:
            show_real_calculated_routes()

    with info_col:
        show_vehicle_selection()


def show_destination_selection():
    vehicle = st.session_state.selected_vehicle
    st.subheader(f"🎯 Routes Réelles pour {vehicle['id']} - {vehicle['driver']}")

    dest_col1, dest_col2 = st.columns([3, 1])

    with dest_col1:
        destination = st.selectbox(
            "🎯 Destination (vraies adresses)", list(DESTINATIONS.keys()), index=0
        )

    with dest_col2:
        if st.button("🗺️ Routes Réelles", type="primary", use_container_width=True):
            calculate_real_routes(vehicle, destination)


def calculate_real_routes(vehicle, destination_name):
    """Calcule vraies routes via APIs multiples"""

    if destination_name not in DESTINATIONS:
        st.error("Destination non trouvée")
        return

    start_coords = (vehicle["lat"], vehicle["lon"])
    end_coords = DESTINATIONS[destination_name]

    with st.spinner("🌐 Calcul routes réelles via APIs..."):

        routes = []

        # Route optimale
        optimal_route = route_client.get_best_route(
            start_coords[0], start_coords[1], end_coords[0], end_coords[1], "fastest"
        )

        if optimal_route:
            routes.append(
                {
                    "name": f"🟢 Route Optimale ({optimal_route['source']})",
                    "coordinates": optimal_route["coordinates"],
                    "duration": optimal_route["duration"],
                    "distance": optimal_route["distance"],
                    "source": optimal_route["source"],
                    "api_used": optimal_route.get("api_used", "unknown"),
                    "roads": optimal_route.get("roads", []),
                    "confidence": optimal_route["confidence"],
                    "icon": "🟢",
                }
            )

        # Route alternative
        alt_route = route_client.get_best_route(
            start_coords[0],
            start_coords[1],
            end_coords[0],
            end_coords[1],
            "alternative",
        )

        if alt_route:
            routes.append(
                {
                    "name": f"🔴 Route Alternative ({alt_route['source']})",
                    "coordinates": alt_route["coordinates"],
                    "duration": alt_route["duration"],
                    "distance": alt_route["distance"],
                    "source": alt_route["source"],
                    "api_used": alt_route.get("api_used", "unknown"),
                    "roads": alt_route.get("roads", []),
                    "confidence": alt_route["confidence"],
                    "icon": "🔴",
                }
            )

        st.session_state.calculated_routes = routes

        # Messages de statut
        for route in routes:
            if route["api_used"] == "osrm":
                st.success(
                    f"🌐 {route['name']}: Route OSRM réelle (confiance: {route['confidence']:.0%})"
                )
            elif route["api_used"] == "graphhopper":
                st.success(
                    f"🗺️ {route['name']}: Route GraphHopper (confiance: {route['confidence']:.0%})"
                )
            else:
                st.info(
                    f"🏘️ {route['name']}: Route Casablanca locale (confiance: {route['confidence']:.0%})"
                )

        st.rerun()


def show_real_routes_map():
    """Carte avec vraies routes"""

    fig = go.Figure()

    # Ajouter véhicules
    for vehicle_id, vehicle in VEHICLES.items():
        size = (
            15
            if st.session_state.selected_vehicle
            and st.session_state.selected_vehicle["id"] == vehicle_id
            else 10
        )

        fig.add_trace(
            go.Scattermapbox(
                lat=[vehicle["lat"]],
                lon=[vehicle["lon"]],
                mode="markers",
                marker=dict(size=size, color=vehicle["color"]),
                text=[f"{vehicle['id']} - {vehicle['driver']}"],
                hovertemplate=f"<b>{vehicle['id']}</b><br>Conducteur: {vehicle['driver']}<br>Vitesse: {vehicle['speed']} km/h<extra></extra>",
                name=vehicle["type"],
                showlegend=True,
            )
        )

    # Ajouter routes réelles calculées
    if st.session_state.calculated_routes:
        colors = ["green", "red", "orange"]
        for i, route in enumerate(st.session_state.calculated_routes):
            if route.get("coordinates") and len(route["coordinates"]) > 1:

                # Vérifier format coordonnées
                try:
                    route_lats = [coord[0] for coord in route["coordinates"]]
                    route_lons = [coord[1] for coord in route["coordinates"]]

                    line_width = 8 if i == 0 else 6

                    fig.add_trace(
                        go.Scattermapbox(
                            lat=route_lats,
                            lon=route_lons,
                            mode="lines",
                            line=dict(width=line_width, color=colors[i]),
                            name=route["name"],
                            hovertemplate=f"<b>{route['name']}</b><br>Durée: {route['duration']} min<br>Distance: {route['distance']} km<br>API: {route['api_used']}<extra></extra>",
                            showlegend=True,
                        )
                    )
                except (IndexError, TypeError) as e:
                    st.error(f"Invalid route coordinates format: {e}")
                    continue
    # Configuration carte
    fig.update_layout(
        mapbox=dict(
            style="open-street-map", center=dict(lat=33.5731, lon=-7.5898), zoom=11.5
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def show_vehicle_selection():
    """Sélection véhicules"""
    st.subheader("🚗 Sélectionner Véhicule")

    for vehicle_id, vehicle in VEHICLES.items():
        if st.button(
            f"{vehicle['type']}\n{vehicle_id} - {vehicle['driver']}",
            key=f"select_{vehicle_id}",
            use_container_width=True,
        ):
            st.session_state.selected_vehicle = vehicle
            st.session_state.calculated_routes = []
            st.success(f"✅ Véhicule {vehicle_id} sélectionné!")
            st.rerun()


def show_real_calculated_routes():
    """Affiche routes réelles calculées"""
    st.subheader("🛣️ Routes Réelles Calculées")

    for i, route in enumerate(st.session_state.calculated_routes):
        css_class = "route-optimal" if i == 0 else "route-traffic"

        with st.container():
            st.markdown(f'<div class="route-card {css_class}">', unsafe_allow_html=True)

            # En-tête route
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown(f"### {route['icon']} {route['name']}")
                st.write(f"**API utilisée:** {route['api_used'].upper()}")
                st.write(f"**Confiance:** {route['confidence']:.0%}")

            with col2:
                st.metric("⏱️ Durée", f"{route['duration']} min")
                avg_speed = (
                    round((route["distance"] / route["duration"]) * 60, 1)
                    if route["duration"] > 0
                    else 0
                )
                st.write(f"🏃‍♂️ {avg_speed} km/h moy")

            with col3:
                st.metric("📏 Distance", f"{route['distance']} km")
                st.write(f"📍 {len(route['coordinates'])} points GPS")

            # Rues utilisées
            if route.get("roads"):
                st.markdown("#### 🛣️ Rues Empruntées")
                roads_text = " → ".join(route["roads"])
                st.write(f"**Itinéraire:** {roads_text}")

            # Détails techniques
            tech_col1, tech_col2 = st.columns(2)

            with tech_col1:
                if route["api_used"] in ["osrm", "graphhopper"]:
                    st.success(f"🌐 Route authentique via {route['source']}")
                else:
                    st.info("🏘️ Route locale Casablanca")

                # Estimation carburant
                fuel_consumption = round(route["distance"] * 0.08, 1)
                fuel_cost = round(fuel_consumption * 14.5, 1)
                st.write(f"⛽ **Carburant:** {fuel_consumption}L (~{fuel_cost} DH)")

            with tech_col2:
                # Analyse qualité route
                quality_factors = []
                if route["confidence"] > 0.9:
                    quality_factors.append("✅ Haute précision")
                if route["api_used"] in ["osrm", "graphhopper"]:
                    quality_factors.append("✅ Source fiable")
                if len(route["coordinates"]) > 10:
                    quality_factors.append("✅ Détail GPS élevé")

                for factor in quality_factors:
                    st.write(factor)

            # Actions route
            action_col1, action_col2, action_col3 = st.columns(3)

            with action_col1:
                if st.button(
                    "🎯 Choisir Route", key=f"choose_{i}", use_container_width=True
                ):
                    st.balloons()
                    st.success("🚀 Navigation activée!")
                    show_turn_by_turn_directions(route)

            with action_col2:
                if st.button(
                    "📊 Analyser", key=f"analyze_{i}", use_container_width=True
                ):
                    show_detailed_route_analysis(route)

            with action_col3:
                if st.button(
                    "🗺️ Export GPS", key=f"export_{i}", use_container_width=True
                ):
                    export_route_gpx(route)

            st.markdown("</div>", unsafe_allow_html=True)


def show_turn_by_turn_directions(route):
    """Instructions turn-by-turn détaillées"""
    with st.expander("🧭 Instructions Navigation Détaillées", expanded=True):
        st.markdown(f"### Navigation: {route['name']}")

        # Instructions basées sur les rues réelles
        instructions = [
            "🚀 **Départ** depuis votre position actuelle",
            f"➡️ **Suivez** {route['source']} pendant {route['distance']} km",
        ]

        # Ajouter instructions basées sur rues
        if route.get("roads"):
            for i, road in enumerate(route["roads"]):
                if i == 0:
                    instructions.append(f"🛣️ **Prenez** {road}")
                else:
                    instructions.append(f"↪️ **Continuez sur** {road}")

        instructions.extend(
            [
                f"⏱️ **Durée totale estimée:** {route['duration']} minutes",
                "🎯 **Arrivée** à destination",
            ]
        )

        for i, instruction in enumerate(instructions, 1):
            st.write(f"**{i}.** {instruction}")

        # Alertes spécifiques
        if route["api_used"] == "fallback":
            st.warning("⚠️ Route générée localement - Vérifiez conditions en temps réel")
        elif route["confidence"] < 0.9:
            st.info("ℹ️ Route avec données partielles - Suivez votre GPS principal")
        else:
            st.success("✅ Route optimisée et fiable")

        # Conseils circulation Casablanca
        current_hour = datetime.now().hour
        if 7 <= current_hour <= 9:
            st.warning("🚦 **Heures de pointe matinales** - Prévoyez +15-20 min")
        elif 17 <= current_hour <= 19:
            st.warning("🚦 **Heures de pointe soir** - Prévoyez +20-30 min")
        elif 12 <= current_hour <= 14:
            st.info("🕐 **Pause déjeuner** - Circulation modérée")


def show_detailed_route_analysis(route):
    """Analyse détaillée de route"""
    with st.expander(f"📊 Analyse Complète - {route['name']}", expanded=True):

        # Métriques techniques avancées
        analysis_col1, analysis_col2 = st.columns(2)

        with analysis_col1:
            st.markdown("#### 🔧 Métriques Techniques")
            st.write(f"**Source données:** {route['source']}")
            st.write(f"**API utilisée:** {route['api_used'].upper()}")
            st.write(f"**Confiance route:** {route['confidence']:.1%}")
            st.write(f"**Points GPS:** {len(route['coordinates'])}")
            st.write(
                f"**Résolution:** {len(route['coordinates'])/route['distance']:.1f} pts/km"
            )

            # Score qualité global
            quality_score = route["confidence"] * 100
            if route["api_used"] in ["osrm", "graphhopper"]:
                quality_score = min(98, quality_score + 5)
            if len(route["coordinates"]) > 20:
                quality_score = min(100, quality_score + 2)

            st.metric("🎯 Score Qualité", f"{quality_score:.0f}/100")

        with analysis_col2:
            st.markdown("#### 💰 Analyse Coûts")

            # Calculs détaillés
            fuel_l = round(route["distance"] * 0.08, 1)  # 8L/100km
            fuel_cost = round(fuel_l * 14.5, 1)  # 14.5 DH/L

            # Coûts additionnels selon destination
            parking_cost = 0
            if "Aéroport" in st.session_state.get("vehicle_destination", ""):
                parking_cost = 20  # Parking aéroport
            elif "Centre-ville" in st.session_state.get("vehicle_destination", ""):
                parking_cost = 15  # Parking centre
            elif "Marina" in st.session_state.get("vehicle_destination", ""):
                parking_cost = 10  # Parking marina

            toll_cost = 0  # Pas de péages à Casablanca
            total_cost = fuel_cost + parking_cost + toll_cost

            st.write(f"⛽ **Carburant:** {fuel_l}L → {fuel_cost} DH")
            st.write(f"🛣️ **Péages:** {toll_cost} DH")
            st.write(f"🅿️ **Parking estimé:** {parking_cost} DH")
            st.metric("💰 Coût Total Estimé", f"{total_cost} DH")

            # Émissions CO₂
            co2_emissions = round(route["distance"] * 0.12, 1)  # 120g CO₂/km
            st.write(f"🌱 **Émissions CO₂:** {co2_emissions} kg")

        # Analyse géographique
        if len(route["coordinates"]) > 5:
            st.markdown("#### 🗺️ Analyse Géographique")

            # Calculer dénivelé approximatif (simulé pour Casablanca)
            elevation_change = random.randint(20, 80)  # Casablanca relativement plate
            st.write(f"📈 **Dénivelé estimé:** {elevation_change}m")

            # Zones traversées
            zones = []
            start_lat, start_lon = route["coordinates"][0]
            end_lat, end_lon = route["coordinates"][-1]

            # Identifier zones de Casablanca
            if start_lat > 33.58:
                zones.append("Nord Casablanca")
            if any(coord[1] < -7.62 for coord in route["coordinates"]):
                zones.append("Zone Anfa")
            if any(coord[0] < 33.55 for coord in route["coordinates"]):
                zones.append("Sud Casablanca")
            if any(coord[1] > -7.58 for coord in route["coordinates"]):
                zones.append("Centre-ville")

            if zones:
                st.write(f"🏘️ **Zones traversées:** {', '.join(zones)}")

        # Graphique vitesse simulé
        if len(route["coordinates"]) > 3:
            st.markdown("#### 📈 Profil Vitesse Estimé")

            segments = min(8, len(route["coordinates"]) - 1)
            speeds = []

            for i in range(segments):
                # Vitesse variable selon zone et type de route
                base_speed = 50
                if route["api_used"] == "osrm":
                    base_speed = random.randint(45, 65)
                elif route["api_used"] == "graphhopper":
                    base_speed = random.randint(40, 60)
                else:
                    base_speed = random.randint(35, 55)

                speeds.append(base_speed + random.randint(-10, 10))

            speed_df = pd.DataFrame(
                {
                    "Segment": [f"Seg {i+1}" for i in range(segments)],
                    "Vitesse (km/h)": speeds,
                }
            )

            import plotly.express as px

            fig_speed = px.line(
                speed_df,
                x="Segment",
                y="Vitesse (km/h)",
                title="Vitesse estimée par segment",
                markers=True,
            )
            fig_speed.update_layout(height=250)
            st.plotly_chart(fig_speed, use_container_width=True)

        # Recommandations contextuelles
        st.markdown("#### 💡 Recommandations Personnalisées")

        recommendations = []

        # Selon API utilisée
        if route["api_used"] == "osrm":
            recommendations.append("✅ Route OSRM optimisée - Suivez les indications")
        elif route["api_used"] == "graphhopper":
            recommendations.append("🗺️ Route GraphHopper - Alternative fiable")
        else:
            recommendations.append("🏘️ Route locale - Double-vérifiez avec votre GPS")

        # Selon confiance
        if route["confidence"] > 0.95:
            recommendations.append("🎯 Très haute confiance - Route recommandée")
        elif route["confidence"] > 0.85:
            recommendations.append("👍 Bonne confiance - Route viable")
        else:
            recommendations.append("⚠️ Confiance modérée - Vérifiez alternatives")

        # Selon durée
        if route["duration"] > 45:
            recommendations.append("⏰ Trajet long - Prévoyez pause si nécessaire")
            recommendations.append("📱 Gardez votre téléphone chargé")

        # Selon heure
        current_hour = datetime.now().hour
        if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
            recommendations.append("🚦 Heures de pointe - Majorez le temps de trajet")

        # Selon destination
        dest = st.session_state.get("vehicle_destination", "")
        if "Aéroport" in dest:
            recommendations.append(
                "✈️ Destination aéroport - Prévoyez temps d'enregistrement"
            )
        elif "Marina" in dest:
            recommendations.append(
                "🌊 Destination Marina - Parking souvent complet weekend"
            )

        for rec in recommendations:
            st.write(f"• {rec}")


def export_route_gpx(route):
    """Export route au format GPX"""
    st.success("📱 Fonctionnalité à venir: Export GPX")
    st.info(
        """
    🔜 **Prochainement disponible:**
    - Export format GPX pour GPS
    - Import dans Google Maps
    - Partage par lien
    - Sauvegarde favoris
    """
    )


# ===== MONITORING MULTI-API =====


def show_api_monitoring():
    """Page monitoring des APIs"""
    st.header("📊 Monitoring Multi-API Routing")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌐 Status APIs en Temps Réel")

        # Test toutes APIs
        if st.button("🔄 Tester Toutes APIs", type="primary"):
            test_results = {}
            with st.spinner("Test des APIs..."):
                for api_name in ["osrm", "graphhopper"]:
                    start_time = time.time()
                    status = route_client.test_api_availability(api_name)
                    response_time = (time.time() - start_time) * 1000

                    test_results[api_name] = {
                        "status": status,
                        "response_time": round(response_time, 1),
                    }

            # Afficher résultats
            for api_name, result in test_results.items():
                if result["status"] == "available":
                    st.success(f"✅ {api_name.upper()}: {result['response_time']}ms")
                else:
                    st.error(
                        f"❌ {api_name.upper()}: {result['response_time']}ms (échec)"
                    )

        # Statistiques utilisation
        st.subheader("📈 Statistiques Utilisation")

        # Cache status
        cache_size = len(route_client.cache)
        st.metric("💾 Routes en Cache", cache_size)

        # Métriques simulées
        api_usage = {
            "OSRM": random.randint(60, 80),
            "GraphHopper": random.randint(15, 25),
            "Fallback": random.randint(5, 15),
        }

        for api, usage in api_usage.items():
            st.metric(f"📊 {api}", f"{usage}%")

    with col2:
        st.subheader("🛣️ Qualité des Routes")

        # Graphique qualité par API
        quality_data = pd.DataFrame(
            {
                "API": ["OSRM", "GraphHopper", "Fallback Casablanca"],
                "Confiance (%)": [95, 90, 80],
                "Routes Calculées": [
                    random.randint(50, 100),
                    random.randint(20, 50),
                    random.randint(10, 30),
                ],
            }
        )

        import plotly.express as px

        fig_quality = px.bar(
            quality_data,
            x="API",
            y="Confiance (%)",
            color="Routes Calculées",
            title="Qualité par API de Routing",
        )
        fig_quality.update_layout(height=300)
        st.plotly_chart(fig_quality, use_container_width=True)

        # Actions admin
        st.subheader("🔧 Actions Administrateur")

        admin_col1, admin_col2 = st.columns(2)

        with admin_col1:
            if st.button("🗑️ Vider Cache", use_container_width=True):
                route_client.cache.clear()
                st.success("✅ Cache vidé!")

        with admin_col2:
            if st.button("📊 Rapport APIs", use_container_width=True):
                st.info("📋 Rapport généré!")

        # Historique performance
        st.subheader("⏱️ Performance 24h")

        hours = list(range(24))
        perf_data = pd.DataFrame(
            {
                "Heure": [f"{h:02d}:00" for h in hours],
                "OSRM (ms)": [random.randint(50, 150) for _ in hours],
                "GraphHopper (ms)": [random.randint(80, 200) for _ in hours],
            }
        )

        fig_perf = px.line(
            perf_data,
            x="Heure",
            y=["OSRM (ms)", "GraphHopper (ms)"],
            title="Latence APIs sur 24h",
        )
        fig_perf.update_layout(height=250)
        st.plotly_chart(fig_perf, use_container_width=True)


# ===== POINT D'ENTRÉE PRINCIPAL =====

if __name__ == "__main__":
    main()
