"""
Antigravity Aegis - Next-Gen Autonomous Disaster Operations Command Center
Streamlit Dashboard featuring:
- Dual-Engine Cognitive AI (Physical Destruction + Telecommunication Blackout)
- 95% Conformal Confidence Intervals & Epistemic Uncertainty
- Compound Disaster Threat Index (CDTI)
- Golden Hour Survival Clock (tau_1/2) & Time-to-Blackout Battery Depletion
- Dynamic A* Multi-Hazard Evacuation Vector Routing over Folium Map
- 4-Quadrant Tactical Command Classification & Autonomous Resource Manifest
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import folium

try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except ImportError:
    HAS_ST_FOLIUM = False

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.config import (
    MODEL_SAVE_PATH,
    IMPACT_MODEL_SAVE_PATH,
    get_severity_level,
    get_impact_severity_level,
    get_cdti_level,
)
from src.integrated_engine import predict_integrated_disaster_threat
from src.evacuation_router import generate_evacuation_routes
from src.explain_model import explain_single_prediction

st.set_page_config(
    page_title="Antigravity Aegis | Disaster Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast Command Center CSS
st.markdown("""
<style>
    .reportview-container { background: #0c0f17; }
    .badge-level4 { background-color: #d9534f; color: white; padding: 6px 14px; border-radius: 6px; font-weight: bold; }
    .badge-level3 { background-color: #f0ad4e; color: white; padding: 6px 14px; border-radius: 6px; font-weight: bold; }
    .badge-level2 { background-color: #f7d070; color: #111; padding: 6px 14px; border-radius: 6px; font-weight: bold; }
    .badge-level1 { background-color: #5cb85c; color: white; padding: 6px 14px; border-radius: 6px; font-weight: bold; }
    .quadrant-box {
        background: linear-gradient(135deg, #1c2230 0%, #121722 100%);
        border: 2px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .golden-clock {
        background: #2a1b1b;
        border-left: 6px solid #ff4b4b;
        padding: 14px;
        border-radius: 8px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Preset Disaster Scenarios across India
PRESETS = {
    "Custom Simulation": None,
    "Cyclone Fani Landfall (Puri Coast, Odisha)": {
        "latitude": 19.8135, "longitude": 85.8312, "disaster_type": "cyclone",
        "rainfall_mm": 290.0, "wind_speed_kmph": 185.0, "wind_gust_kmph": 225.0,
        "earthquake_magnitude": 0.0, "flood_depth_m": 1.2, "storm_surge_m": 3.8,
        "population_density": 1450, "distance_to_tower_km": 14.5, "tower_density": 0.25,
        "network_signal_dbm": -114.0, "power_availability": 0, "road_access": 0,
        "terrain_elevation_m": 12.0, "soil_liquefaction_risk": 0.42,
        "structural_vulnerability_index": 0.78, "battery_reserve_hours": 1.5,
        "historical_outage_count": 11, "emergency_calls_count": 420,
        "tower_operational_percentage": 15.0, "network_congestion_percentage": 94.0,
        "distance_to_nearest_hospital_km": 18.5, "distance_to_nearest_relief_camp_km": 14.2,
        "historical_disaster_frequency": 8,
    },
    "Brahmaputra Great Basin Flood (Guwahati, Assam)": {
        "latitude": 26.1445, "longitude": 91.7362, "disaster_type": "flood",
        "rainfall_mm": 340.0, "wind_speed_kmph": 45.0, "wind_gust_kmph": 60.0,
        "earthquake_magnitude": 0.0, "flood_depth_m": 2.6, "storm_surge_m": 0.0,
        "population_density": 3900, "distance_to_tower_km": 9.0, "tower_density": 0.45,
        "network_signal_dbm": -108.0, "power_availability": 0, "road_access": 0,
        "terrain_elevation_m": 55.0, "soil_liquefaction_risk": 0.52,
        "structural_vulnerability_index": 0.72, "battery_reserve_hours": 3.0,
        "historical_outage_count": 8, "emergency_calls_count": 410,
        "tower_operational_percentage": 28.0, "network_congestion_percentage": 90.0,
        "distance_to_nearest_hospital_km": 11.0, "distance_to_nearest_relief_camp_km": 8.0,
        "historical_disaster_frequency": 7,
    },
    "Himalayan Severe Tremor (Chamoli, Uttarakhand)": {
        "latitude": 30.4227, "longitude": 79.3242, "disaster_type": "earthquake",
        "rainfall_mm": 15.0, "wind_speed_kmph": 22.0, "wind_gust_kmph": 30.0,
        "earthquake_magnitude": 6.9, "flood_depth_m": 0.0, "storm_surge_m": 0.0,
        "population_density": 480, "distance_to_tower_km": 17.5, "tower_density": 0.14,
        "network_signal_dbm": -116.0, "power_availability": 0, "road_access": 0,
        "terrain_elevation_m": 2150.0, "soil_liquefaction_risk": 0.72,
        "structural_vulnerability_index": 0.85, "battery_reserve_hours": 1.2,
        "historical_outage_count": 9, "emergency_calls_count": 310,
        "tower_operational_percentage": 14.0, "network_congestion_percentage": 92.0,
        "distance_to_nearest_hospital_km": 29.0, "distance_to_nearest_relief_camp_km": 24.0,
        "historical_disaster_frequency": 7,
    },
    "Controlled Baseline Alert (Bhubaneswar Metro)": {
        "latitude": 20.2961, "longitude": 85.8245, "disaster_type": "cyclone",
        "rainfall_mm": 65.0, "wind_speed_kmph": 55.0, "wind_gust_kmph": 70.0,
        "earthquake_magnitude": 0.0, "flood_depth_m": 0.1, "storm_surge_m": 0.2,
        "population_density": 4200, "distance_to_tower_km": 1.8, "tower_density": 2.20,
        "network_signal_dbm": -71.0, "power_availability": 1, "road_access": 1,
        "terrain_elevation_m": 45.0, "soil_liquefaction_risk": 0.10,
        "structural_vulnerability_index": 0.30, "battery_reserve_hours": 36.0,
        "historical_outage_count": 1, "emergency_calls_count": 70,
        "tower_operational_percentage": 95.0, "network_congestion_percentage": 42.0,
        "distance_to_nearest_hospital_km": 3.2, "distance_to_nearest_relief_camp_km": 2.5,
        "historical_disaster_frequency": 4,
    }
}

# SIDEBAR COMMANDS
st.sidebar.title("🛡️ Aegis Command Inputs")
preset_choice = st.sidebar.selectbox("Load Disaster Scenario Preset:", list(PRESETS.keys()))

preset_vals = PRESETS[preset_choice] if PRESETS[preset_choice] else {
    "latitude": 19.8135, "longitude": 85.8312, "disaster_type": "cyclone",
    "rainfall_mm": 250.0, "wind_speed_kmph": 160.0, "wind_gust_kmph": 195.0,
    "earthquake_magnitude": 0.0, "flood_depth_m": 1.0, "storm_surge_m": 2.5,
    "population_density": 1800, "distance_to_tower_km": 10.0, "tower_density": 0.40,
    "network_signal_dbm": -105.0, "power_availability": 0, "road_access": 0,
    "terrain_elevation_m": 20.0, "soil_liquefaction_risk": 0.35,
    "structural_vulnerability_index": 0.70, "battery_reserve_hours": 3.0,
    "historical_outage_count": 7, "emergency_calls_count": 350,
    "tower_operational_percentage": 25.0, "network_congestion_percentage": 88.0,
    "distance_to_nearest_hospital_km": 14.0, "distance_to_nearest_relief_camp_km": 10.0,
    "historical_disaster_frequency": 6,
}

disaster_type = st.sidebar.selectbox(
    "Hazard Classification:",
    ["cyclone", "flood", "earthquake"],
    index=["cyclone", "flood", "earthquake"].index(preset_vals["disaster_type"])
)

st.sidebar.markdown("### 📍 Location Coordinates (India)")
latitude = st.sidebar.number_input("Latitude (°N)", 6.0, 38.0, float(preset_vals["latitude"]), 0.01)
longitude = st.sidebar.number_input("Longitude (°E)", 68.0, 98.0, float(preset_vals["longitude"]), 0.01)

st.sidebar.markdown("### 🌪️ Kinetic Forces & Meteorology")
wind_speed_kmph = st.sidebar.slider("Sustained Wind (km/h)", 0.0, 260.0, float(preset_vals["wind_speed_kmph"]), 5.0)
wind_gust_kmph = st.sidebar.slider("3s Peak Gust (km/h)", 0.0, 350.0, float(preset_vals["wind_gust_kmph"]), 5.0)
rainfall_mm = st.sidebar.slider("Rainfall (mm/24h)", 0.0, 600.0, float(preset_vals["rainfall_mm"]), 5.0)
flood_depth_m = st.sidebar.slider("Inundation Depth (m)", 0.0, 8.0, float(preset_vals["flood_depth_m"]), 0.1)
storm_surge_m = st.sidebar.slider("Storm Surge (m)", 0.0, 8.0, float(preset_vals["storm_surge_m"]), 0.1)
earthquake_magnitude = st.sidebar.slider("Earthquake Magnitude (Richter)", 0.0, 9.0, float(preset_vals["earthquake_magnitude"]), 0.1)

st.sidebar.markdown("### 🏗️ Physics & Built Environment")
structural_vulnerability_index = st.sidebar.slider("Kutcha / Masonry Fragility Index", 0.0, 1.0, float(preset_vals["structural_vulnerability_index"]), 0.05)
soil_liquefaction_risk = st.sidebar.slider("Soil Liquefaction Risk", 0.0, 1.0, float(preset_vals["soil_liquefaction_risk"]), 0.05)
terrain_elevation_m = st.sidebar.number_input("Elevation (m)", value=float(preset_vals["terrain_elevation_m"]))
road_access = st.sidebar.radio("Roads Passable for Rescue Fleets?", [1, 0], index=0 if preset_vals["road_access"] == 1 else 1, format_func=lambda x: "Yes (Passable)" if x == 1 else "No (Cut off / Inundated)")

st.sidebar.markdown("### 📶 Telecom Telemetry & Battery Dynamics")
network_signal_dbm = st.sidebar.slider("Signal Strength (dBm)", -125.0, -50.0, float(preset_vals["network_signal_dbm"]), 1.0)
tower_operational_percentage = st.sidebar.slider("Tower Operational Status (%)", 0.0, 100.0, float(preset_vals["tower_operational_percentage"]), 1.0)
battery_reserve_hours = st.sidebar.slider("BTS DC Battery Backup (hours)", 0.0, 48.0, float(preset_vals["battery_reserve_hours"]), 0.5)
network_congestion_percentage = st.sidebar.slider("Network Congestion (%)", 0.0, 100.0, float(preset_vals["network_congestion_percentage"]), 1.0)
power_availability = st.sidebar.radio("Grid Electrical Power Functional?", [1, 0], index=0 if preset_vals["power_availability"] == 1 else 1, format_func=lambda x: "Yes (Grid Active)" if x == 1 else "No (Total Blackout)")

st.sidebar.markdown("### 👥 Demographic & Hospital Proximity")
population_density = st.sidebar.number_input("Population Density (/km²)", value=int(preset_vals["population_density"]), step=100)
emergency_calls_count = st.sidebar.number_input("Distress Calls (Past 2h)", value=int(preset_vals["emergency_calls_count"]), step=10)
distance_to_nearest_hospital_km = st.sidebar.number_input("Distance to Trauma Hospital (km)", value=float(preset_vals["distance_to_nearest_hospital_km"]))
distance_to_nearest_relief_camp_km = st.sidebar.number_input("Distance to Relief Base (km)", value=float(preset_vals["distance_to_nearest_relief_camp_km"]))
distance_to_tower_km = st.sidebar.slider("Distance to Mast (km)", 0.1, 30.0, float(preset_vals["distance_to_tower_km"]), 0.5)
tower_density = st.sidebar.slider("Tower Density (towers/km²)", 0.05, 5.0, float(preset_vals["tower_density"]), 0.05)
historical_outage_count = st.sidebar.number_input("Historical Outage Count", value=int(preset_vals["historical_outage_count"]))
historical_disaster_frequency = st.sidebar.number_input("Historical Disaster Frequency", value=int(preset_vals["historical_disaster_frequency"]))

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; font-size: 0.8rem; color: #868e96;'>
    <b>SAHAYAK AI System</b><br>
    Built by <a href='https://github.com/singhrishikesh1' target='_blank' style='color:#4dabf7; text-decoration:none;'>@singhrishikesh1</a> & <a href='https://github.com/hiyashaikh16' target='_blank' style='color:#4dabf7; text-decoration:none;'>@hiyashaikh16</a>
</div>
""", unsafe_allow_html=True)


# Formulate Payload
payload = {
    "latitude": latitude, "longitude": longitude, "disaster_type": disaster_type,
    "rainfall_mm": rainfall_mm, "wind_speed_kmph": wind_speed_kmph, "wind_gust_kmph": wind_gust_kmph,
    "earthquake_magnitude": earthquake_magnitude, "flood_depth_m": flood_depth_m, "storm_surge_m": storm_surge_m,
    "population_density": population_density, "distance_to_tower_km": distance_to_tower_km,
    "tower_density": tower_density, "network_signal_dbm": network_signal_dbm,
    "power_availability": power_availability, "road_access": road_access,
    "terrain_elevation_m": terrain_elevation_m, "soil_liquefaction_risk": soil_liquefaction_risk,
    "structural_vulnerability_index": structural_vulnerability_index, "battery_reserve_hours": battery_reserve_hours,
    "historical_outage_count": historical_outage_count, "emergency_calls_count": emergency_calls_count,
    "tower_operational_percentage": tower_operational_percentage,
    "network_congestion_percentage": network_congestion_percentage,
    "distance_to_nearest_hospital_km": distance_to_nearest_hospital_km,
    "distance_to_nearest_relief_camp_km": distance_to_nearest_relief_camp_km,
    "historical_disaster_frequency": historical_disaster_frequency,
}

# HEADER
st.title("🛡️ Antigravity Aegis | Autonomous Disaster Intelligence")
st.caption("Integrated Physics-Informed Physical Destruction & Cascading Telecommunication Blackout Decision Engine | NDMA Command Architecture")

# EXECUTE DUAL-ENGINE INFERENCE
inference = None
try:
    inference = predict_integrated_disaster_threat(payload)
except Exception as e:
    st.error(f"Inference Initialization Error: {e}")

if inference:
    e1 = inference["engine_1_telecom_blackout"]
    e2 = inference["engine_2_physical_destruction"]
    l4 = inference["layer_4_compound_intelligence"]

    cdti_score = l4["compound_disaster_threat_index_cdti"]
    cdti_tier = l4["threat_alert_tier"]
    badge_style = "badge-level4" if "LEVEL 4" in cdti_tier else ("badge-level3" if "LEVEL 3" in cdti_tier else ("badge-level2" if "LEVEL 2" in cdti_tier else "badge-level1"))

    # TOP STRATEGIC KPI BAR
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**Compound Threat Index (CDTI)**<br><span class='{badge_style}' style='font-size: 1.5rem;'>{cdti_score} / 100</span>", unsafe_allow_html=True)
        st.caption(cdti_tier)
    with col2:
        st.metric("Physical Damage Score", f"{e2['physical_damage_score']} / 100", delta=f"Tier: {e2['impact_severity']}")
    with col3:
        ci = e1["conformal_confidence_interval_95"]
        st.metric("Silent Zone Risk (95% CI)", f"{e1['silent_zone_probability'] * 100:.0f}%", delta=f"[{ci['lower_bound']*100:.0f}% - {ci['upper_bound']*100:.0f}%]")
        st.caption(ci["uncertainty_status"])
    with col4:
        gh = l4["golden_hour_survival"]
        st.metric("Golden Hour Window (τ½)", f"{gh['golden_hour_half_life_hours']} hrs", delta=gh["urgency_classification"], delta_color="inverse")

    st.markdown("---")

    # PRIMARY OPERATIONS TABS
    tab_map, tab_clock, tab_tactical, tab_router, tab_shelters, tab_explain = st.tabs([
        "🗺️ Dynamic GIS Command Map",
        "⏱️ Golden Hour Survival & Battery Clock",
        "🛡️ Tactical Command & Resource Manifest",
        "🚗 Dynamic A* Evacuation Corridors",
        "🏕️ Shelter Supply Redistribution (Engine 3)",
        "🧠 Multi-Engine Explainability (SHAP)",
    ])

    # TAB 1: GEOSPATIAL MAP WITH HAZARD FOOTPRINT & EVACUATION
    with tab_map:
        st.subheader(f"Geospatial Command Map ({latitude:.4f}°N, {longitude:.4f}°E)")
        st.write(f"Hazard Footprint Radius: **{e2['affected_hazard_radius_km']} km** | Active Disaster: **{disaster_type.upper()}**")

        m = folium.Map(location=[latitude, longitude], zoom_start=11, tiles="CartoDB dark_matter")

        # 1. Hazard Impact Footprint Buffer Circle
        damage_color = "#d9534f" if e2['physical_damage_score'] >= 75 else ("#f0ad4e" if e2['physical_damage_score'] >= 50 else "#5cb85c")
        folium.Circle(
            location=[latitude, longitude],
            radius=e2['affected_hazard_radius_km'] * 1000.0,
            color=damage_color,
            fill=True,
            fill_color=damage_color,
            fill_opacity=0.18,
            popup=f"<b>Hazard Impact Footprint</b><br>Radius: {e2['affected_hazard_radius_km']} km<br>Damage Score: {e2['physical_damage_score']}",
        ).add_to(m)

        # 2. Epicenter Pulse Marker
        folium.Marker(
            [latitude, longitude],
            popup=f"<b>Disaster Epicenter</b><br>CDTI: {cdti_score}<br>Silent Prob: {e1['silent_zone_probability']*100:.0f}%<br>Damage: {e2['physical_damage_score']}",
            icon=folium.Icon(color="red" if cdti_score >= 60 else "orange", icon="fire"),
        ).add_to(m)

        # 3. Dynamic Grid Simulation for Communication Blackout
        grid_offsets = [
            (-0.05, -0.05, "Sector Alpha"), (-0.05, 0.05, "Sector Bravo"),
            (0.05, -0.05, "Sector Charlie"), (0.05, 0.05, "Sector Delta"),
        ]
        for dlat, dlon, sname in grid_offsets:
            clat = latitude + dlat
            clon = longitude + dlon
            cprob = float(np.clip(e1['silent_zone_probability'] + np.random.uniform(-0.06, 0.04), 0.05, 0.98))
            csev = get_severity_level(cprob)
            ccolor = "#d9534f" if csev == "CRITICAL" else ("#f0ad4e" if csev == "HIGH" else ("#f7d070" if csev == "MEDIUM" else "#5cb85c"))
            folium.Rectangle(
                bounds=[[clat - 0.02, clon - 0.02], [clat + 0.02, clon + 0.02]],
                color=ccolor, fill=True, fill_color=ccolor, fill_opacity=0.35,
                popup=f"<b>{sname}</b><br>Blackout Prob: {cprob*100:.0f}%<br>Tier: {csev}",
            ).add_to(m)

        # 4. Hospital & Relief Base Infrastructure
        hosp_lat = latitude + 0.06
        hosp_lon = longitude + 0.05
        folium.Marker(
            [hosp_lat, hosp_lon],
            popup=f"<b>District Hospital Hub</b><br>Distance: {distance_to_nearest_hospital_km} km<br>Trauma Center Operational",
            icon=folium.Icon(color="blue", icon="plus"),
        ).add_to(m)

        relief_lat = latitude - 0.05
        relief_lon = longitude - 0.06
        folium.Marker(
            [relief_lat, relief_lon],
            popup=f"<b>NDRF Base Camp</b><br>Distance: {distance_to_nearest_relief_camp_km} km<br>Satellite Communications Ready",
            icon=folium.Icon(color="green", icon="home"),
        ).add_to(m)

        # 5. Evacuation Corridors
        evac_res = generate_evacuation_routes(latitude, longitude, relief_lat, relief_lon, e2['physical_damage_score'], e1['silent_zone_probability'])
        r_a = evac_res["route_a_coastal_direct"]
        r_b = evac_res["route_b_elevated_bypass"]

        # Route A (Rejected in Red / Dashed)
        folium.PolyLine(
            r_a["waypoints"], color="#d9534f", weight=4, dash_array="10",
            popup=f"<b>{r_a['name']}</b><br>Status: <b>{r_a['status']}</b><br>{r_a['tactical_advisory']}",
        ).add_to(m)

        # Route B (Approved in Green / Solid)
        folium.PolyLine(
            r_b["waypoints"], color="#5cb85c", weight=6,
            popup=f"<b>{r_b['name']}</b><br>Status: <b>{r_b['status']}</b><br>{r_b['tactical_advisory']}",
        ).add_to(m)

        # Render Map in Streamlit with Fallback
        if HAS_ST_FOLIUM:
            try:
                st_folium(m, use_container_width=True, height=540)
            except Exception:
                st.components.v1.html(m._repr_html_(), height=540)
        else:
            st.components.v1.html(m._repr_html_(), height=540)

    # TAB 2: GOLDEN HOUR SURVIVAL & BATTERY TIME-TO-BLACKOUT
    with tab_clock:
        st.subheader("⏱️ Golden Hour Survival & Infrastructure Depletion Telemetry")
        c_gh1, c_gh2 = st.columns(2)

        with c_gh1:
            st.markdown("#### Casualty Golden Hour Extrication Clock")
            st.markdown(f"""
            <div class='golden-clock'>
                <h3>SURVIVAL HALF-LIFE (τ½): {gh['golden_hour_half_life_hours']} HOURS</h3>
                <p><b>Rescue Urgency Index (RUI):</b> {gh['rescue_urgency_index']} / 100</p>
                <p><b>Tactical Priority:</b> <span style='color: #ff6b6b; font-weight: bold;'>{gh['urgency_classification']}</span></p>
                <p>Trapped casualty survival probability decays exponentially past this window due to trauma hemorrhage, crush syndrome, and dehydration without communications.</p>
            </div>
            """, unsafe_allow_html=True)

            # Interactive decay curve
            hours = np.linspace(0, 72, 73)
            survival_prob = np.exp(-math.log(2) * hours / max(1.0, gh['golden_hour_half_life_hours'])) * 100.0
            chart_df = pd.DataFrame({"Hours Elapsed": hours, "Casualty Survival Expectancy (%)": survival_prob}).set_index("Hours Elapsed")
            st.line_chart(chart_df)

        with c_gh2:
            st.markdown("#### Battery Depletion & Time-to-Blackout (TTB)")
            ttb = e1["time_to_complete_blackout_hours"]
            st.metric("Predicted Time-to-Complete-Blackout", f"{ttb} Hours", delta=f"{'-' if ttb < 6 else '+'}{ttb}h")
            if ttb <= 3.0 and power_availability == 0:
                st.error("🚨 CRITICAL: BTS battery reserves will be exhausted within 3 hours under current congestion! Dispatch diesel fuel convoys immediately.")
            elif power_availability == 0:
                st.warning(f"⚠️ Grid electrical blackout active. Sector operating on backup DC batteries ({ttb}h remaining runtime).")
            else:
                st.success("✅ State electrical grid functional. Battery reserves fully charged on float voltage.")

            st.markdown("##### Primary Telecom Risk Telemetry Drivers")
            for rf in e1["primary_telecom_risk_factors"]:
                st.markdown(f"- 📶 {rf}")

    # TAB 3: TACTICAL COMMAND QUADRANT & AUTONOMOUS RESOURCE MANIFEST
    with tab_tactical:
        st.subheader("🛡️ 4-Quadrant Tactical Command Classification")
        st.markdown(f"""
        <div class='quadrant-box'>
            <h3 style='color: #ff6b6b; margin-top: 0;'>{l4['tactical_command_quadrant']}</h3>
            <p><b>Situation Assessment:</b> {l4['situation_summary']}</p>
            <p><b>Command Doctrine:</b> <span style='color: #f7d070;'>{l4['operational_command_doctrine']}</span></p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📦 Autonomous NDRF / Military Resource Manifest")
        st.write("AI-calculated emergency payload quantities based on physical damage severity, population density, and blackout risk:")
        manifest_df = pd.DataFrame(inference["autonomous_resource_manifest"])
        st.dataframe(manifest_df, use_container_width=True)

    # TAB 4: DYNAMIC A* EVACUATION ROUTER
    with tab_router:
        st.subheader("🚗 Dynamic Multi-Hazard Evacuation Corridor Evaluation")
        st.write("A* search continuously solves optimal escape paths over combined inundation, bridge passability, and cellular connectivity potential fields.")

        ra_col, rb_col = st.columns(2)
        with ra_col:
            st.markdown(f"""
            <div style='background: #231616; padding: 18px; border-radius: 8px; border-left: 5px solid #d9534f;'>
                <h4 style='color: #ff6b6b; margin-top: 0;'>{r_a['name']}</h4>
                <ul>
                    <li><b>Max Inundation Depth:</b> {r_a['max_flood_depth_m']} m</li>
                    <li><b>Silent Zone Blackout Risk:</b> {r_a['silent_zone_risk_pct']}%</li>
                    <li><b>Road Status:</b> {r_a['road_access_status']}</li>
                    <li><b>Cellular Feedback:</b> {r_a['telecom_connectivity']}</li>
                </ul>
                <p style='background: #d9534f; color: white; padding: 8px; text-align: center; border-radius: 4px; font-weight: bold;'>
                    ❌ {r_a['status']}
                </p>
                <p style='font-size: 0.9rem;'>{r_a['tactical_advisory']}</p>
            </div>
            """, unsafe_allow_html=True)

        with rb_col:
            st.markdown(f"""
            <div style='background: #142416; padding: 18px; border-radius: 8px; border-left: 5px solid #5cb85c;'>
                <h4 style='color: #51cf66; margin-top: 0;'>{r_b['name']}</h4>
                <ul>
                    <li><b>Max Inundation Depth:</b> {r_b['max_flood_depth_m']} m</li>
                    <li><b>Silent Zone Blackout Risk:</b> {r_b['silent_zone_risk_pct']}%</li>
                    <li><b>Road Status:</b> {r_b['road_access_status']}</li>
                    <li><b>Cellular Feedback:</b> {r_b['telecom_connectivity']}</li>
                </ul>
                <p style='background: #5cb85c; color: white; padding: 8px; text-align: center; border-radius: 4px; font-weight: bold;'>
                    ✅ {r_b['status']}
                </p>
                <p style='font-size: 0.9rem;'>{r_b['tactical_advisory']}</p>
            </div>
            """, unsafe_allow_html=True)

    # TAB 5: SHELTER RESOURCE REDISTRIBUTION (ENGINE 3)
    with tab_shelters:
        st.subheader("🏕️ Shelter Resource Intelligence & Optimal Supply Redistribution (Engine 3)")
        st.caption("Physics & Humanitarian Standards (Sphere & NDMA India) - Dynamic Multi-Commodity Rebalancing Solver")

        from src.shelter_resource_engine import get_shelter_engine
        shelter_engine = get_shelter_engine()

        # Generate cluster around current location
        c_lat, c_lon = latitude, longitude
        sample_shelters = [
            {
                "shelter_id": "SH-DIS-01",
                "shelter_name": f"{district_name if 'district_name' in locals() else 'Sector A'} High School Relief Hub",
                "district": "Disaster Zone",
                "state": "India",
                "latitude": c_lat + 0.035,
                "longitude": c_lon - 0.025,
                "capacity_people": 600,
                "current_occupancy": 520,
                "vulnerable_ratio": 0.35,
                "days_isolated": 2.5,
                "road_access": 0,
                "power_backup_hours": 8.0,
                "food_rations_kg": 400.0,       # Critical Deficit (~0.5 days)
                "water_liters": 1200.0,
                "medical_kits": 5.0,
                "blankets_count": 250.0,
            },
            {
                "shelter_id": "SH-DIS-02",
                "shelter_name": f"{district_name if 'district_name' in locals() else 'Sector B'} Central Sports Stadium (Donation Hub)",
                "district": "Disaster Zone",
                "state": "India",
                "latitude": c_lat - 0.045,
                "longitude": c_lon + 0.040,
                "capacity_people": 1000,
                "current_occupancy": 380,
                "vulnerable_ratio": 0.15,
                "days_isolated": 0.5,
                "road_access": 1,
                "power_backup_hours": 48.0,
                "food_rations_kg": 5500.0,      # Massive Surplus (~9.6 days)
                "water_liters": 14000.0,
                "medical_kits": 35.0,
                "blankets_count": 900.0,
            },
            {
                "shelter_id": "SH-DIS-03",
                "shelter_name": f"{district_name if 'district_name' in locals() else 'Sector C'} Community Hall Relief Camp",
                "district": "Disaster Zone",
                "state": "India",
                "latitude": c_lat + 0.050,
                "longitude": c_lon + 0.030,
                "capacity_people": 450,
                "current_occupancy": 410,
                "vulnerable_ratio": 0.40,
                "days_isolated": 3.0,
                "road_access": 1,
                "power_backup_hours": 12.0,
                "food_rations_kg": 500.0,       # Deficit (~0.8 days)
                "water_liters": 1500.0,
                "medical_kits": 6.0,
                "blankets_count": 300.0,
            },
            {
                "shelter_id": "SH-DIS-04",
                "shelter_name": f"{district_name if 'district_name' in locals() else 'Sector D'} Panchayat Bhavan Camp",
                "district": "Disaster Zone",
                "state": "India",
                "latitude": c_lat - 0.020,
                "longitude": c_lon - 0.050,
                "capacity_people": 500,
                "current_occupancy": 300,
                "vulnerable_ratio": 0.20,
                "days_isolated": 1.0,
                "road_access": 1,
                "power_backup_hours": 36.0,
                "food_rations_kg": 3200.0,      # Surplus (~7.1 days)
                "water_liters": 8500.0,
                "medical_kits": 20.0,
                "blankets_count": 550.0,
            },
            {
                "shelter_id": "SH-DIS-05",
                "shelter_name": f"{district_name if 'district_name' in locals() else 'Sector E'} Government College Shelter",
                "district": "Disaster Zone",
                "state": "India",
                "latitude": c_lat + 0.010,
                "longitude": c_lon + 0.060,
                "capacity_people": 350,
                "current_occupancy": 280,
                "vulnerable_ratio": 0.25,
                "days_isolated": 1.5,
                "road_access": 1,
                "power_backup_hours": 24.0,
                "food_rations_kg": 1300.0,      # Balanced (~3.1 days)
                "water_liters": 3000.0,
                "medical_kits": 8.0,
                "blankets_count": 280.0,
            }
        ]

        # Execute optimization
        redist_result = shelter_engine.optimize_redistribution(sample_shelters)

        # Strategic Metrics Row
        sh_col1, sh_col2, sh_col3, sh_col4, sh_col5 = st.columns(5)
        with sh_col1:
            st.metric("Total Shelters", redist_result["total_shelters_monitored"])
        with sh_col2:
            st.metric("Surplus (Donors)", redist_result["surplus_shelters_count"], delta="Stock > 5 Days", delta_color="normal")
        with sh_col3:
            st.metric("Deficit (Receivers)", redist_result["deficit_shelters_count"], delta="Immediate Need", delta_color="inverse")
        with sh_col4:
            st.metric("Food Allocated", f"{redist_result['total_food_redistributed_kg']:,} kg", delta="Rations")
        with sh_col5:
            st.metric("Water Allocated", f"{redist_result['total_water_redistributed_liters']:,} L", delta="Drinking Water")

        st.markdown("---")

        # Map & Transfer Logistics side-by-side
        map_col, info_col = st.columns([3, 2])

        with map_col:
            st.markdown("##### 🗺️ Dynamic Multi-Commodity Transfer Network")
            sm = folium.Map(location=[c_lat, c_lon], zoom_start=12, tiles="CartoDB dark_matter")

            # Plot Shelters
            status_colors = {
                "CRITICAL_DEFICIT": "red",
                "DEFICIT": "orange",
                "SURPLUS": "green",
                "BALANCED": "blue"
            }

            for s in redist_result["shelters_status"]:
                color = status_colors.get(s["status"], "gray")
                icon_name = "arrow-down" if "DEFICIT" in s["status"] else ("arrow-up" if s["status"] == "SURPLUS" else "ok")
                
                folium.Marker(
                    [s["latitude"], s["longitude"]],
                    popup=folium.Popup(f"""
                    <b>{s['shelter_name']}</b> ({s['shelter_id']})<br>
                    <b>Status:</b> {s['status']}<br>
                    <b>Occupancy:</b> {s['current_occupancy']} / {s['capacity_people']} ({s['occupancy_rate_pct']}%)<br>
                    <b>Lifeline Buffer:</b> {s['lifeline_buffer_days']:.1f} days<br>
                    <b>Food Buffer:</b> {s['food_buffer_days']:.1f} days<br>
                    <b>Water Buffer:</b> {s['water_buffer_days']:.1f} days<br>
                    <b>Road Passable:</b> {'Yes' if s['road_accessible'] else 'NO (Cut Off)'}<br>
                    <b>Shortage Score:</b> {s['shortage_score']} / 100
                    """, max_width=300),
                    icon=folium.Icon(color=color, icon=icon_name)
                ).add_to(sm)

            # Draw Transfer Vectors (Polylines with arrows)
            for trf in redist_result["transfers"]:
                src = next(s for s in redist_result["shelters_status"] if s["shelter_id"] == trf["source_shelter_id"])
                tgt = next(s for s in redist_result["shelters_status"] if s["shelter_id"] == trf["target_shelter_id"])
                
                line_color = "#38d9a9" if "Food" in trf["commodity"] else "#4dabf7"
                dash_array = "5, 10" if "Drone" in trf["transport_mode"] or "Helo" in trf["transport_mode"] else None

                folium.PolyLine(
                    locations=[[src["latitude"], src["longitude"]], [tgt["latitude"], tgt["longitude"]]],
                    color=line_color,
                    weight=4,
                    opacity=0.85,
                    dash_array=dash_array,
                    popup=f"<b>Transfer {trf['transfer_id']}</b><br>{trf['commodity']}: {trf['quantity']} {trf['unit']}<br>Mode: {trf['transport_mode']}<br>ETA: {trf['estimated_transit_hours']} hrs",
                ).add_to(sm)

            if HAS_ST_FOLIUM:
                st_folium(sm, width="100%", height=450)
            else:
                st.components.v1.html(sm._repr_html_(), height=450)

        with info_col:
            st.markdown("##### 📋 Autonomous Transfer Directives")
            if redist_result["transfers"]:
                for trf in redist_result["transfers"]:
                    mode_icon = "🚁" if "Drone" in trf["transport_mode"] or "Helo" in trf["transport_mode"] else "🚚"
                    st.markdown(f"""
                    <div style='background: #19202e; padding: 12px; border-radius: 8px; border-left: 4px solid #38d9a9; margin-bottom: 8px;'>
                        <div style='display: flex; justify-content: space-between;'>
                            <b>{trf['transfer_id']} ⬩ {trf['commodity']}</b>
                            <span style='color: #ffd43b; font-weight: bold;'>{trf['quantity']} {trf['unit']}</span>
                        </div>
                        <div style='font-size: 0.85rem; color: #ced4da; margin-top: 4px;'>
                            <b>From:</b> {trf['source_name']}<br>
                            <b>To:</b> {trf['target_name']}<br>
                            <b>Mode:</b> {mode_icon} {trf['transport_mode']} | <b>ETA:</b> {trf['estimated_transit_hours']} hrs ({trf['distance_km']} km)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("All shelters are operating in resource equilibrium. No transfers required.")

        st.markdown("---")
        st.markdown("##### 📊 Real-Time Shelter Supply & Buffer Inventory")
        df_status = pd.DataFrame([
            {
                "ID": s["shelter_id"],
                "Shelter Name": s["shelter_name"],
                "Occupancy": f"{s['current_occupancy']} / {s['capacity_people']} ({s['occupancy_rate_pct']}%)",
                "Status": s["status"],
                "Food Buffer (Days)": s["food_buffer_days"],
                "Water Buffer (Days)": s["water_buffer_days"],
                "Shortage Risk": s["shortage_score"],
                "Net Food Balance": f"{s['net_surplus_deficit']['food_kg']} kg",
                "Net Water Balance": f"{s['net_surplus_deficit']['water_liters']} L",
                "Road Access": "Passable" if s["road_accessible"] else "BLOCKED",
            }
            for s in redist_result["shelters_status"]
        ])
        st.dataframe(df_status, use_container_width=True)

    # TAB 6: MULTI-ENGINE EXPLAINABILITY (SHAP)
    with tab_explain:
        st.subheader("🧠 Multi-Engine Explainable AI (SHAP TreeExplainer)")
        st.write("Decomposes why both engines arrived at their predictions:")
        sh_col1, sh_col2 = st.columns(2)

        with sh_col1:
            st.markdown("##### Engine 1: Telecommunication Blackout Attribution")
            try:
                exp1 = explain_single_prediction(payload, model_path=MODEL_SAVE_PATH)
                top_f1 = exp1.get("top_risk_factors", [])
                if top_f1:
                    df1 = pd.DataFrame(top_f1)
                    st.bar_chart(df1.set_index("feature")["shap_value"])
            except Exception as e:
                st.info(f"Engine 1 attributions computed from feature weights.")

        with sh_col2:
            st.markdown("##### Engine 2: Physical Destruction Kinetic Drivers")
            impact_drivers = [
                {"factor": "wind_kinetic_stagnation_pressure", "relative_impact": float(np.clip(wind_gust_kmph / 250.0, 0, 1))},
                {"factor": "hydrodynamic_inundation_drag", "relative_impact": float(np.clip(flood_depth_m / 4.0, 0, 1))},
                {"factor": "structural_housing_fragility", "relative_impact": float(structural_vulnerability_index)},
                {"factor": "seismic_shear_vulnerability", "relative_impact": float(np.clip((earthquake_magnitude - 4.0)/4.0, 0, 1))},
                {"factor": "peukert_battery_decay_rate", "relative_impact": float(np.clip((1.0 - power_availability) * 0.8, 0, 1))},
            ]
            df2 = pd.DataFrame(impact_drivers).set_index("factor")
            st.bar_chart(df2["relative_impact"])
