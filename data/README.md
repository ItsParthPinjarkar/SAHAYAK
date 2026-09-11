# SAHAYAK — Dataset Documentation & Data Dictionary

This directory contains the datasets and data synthesis engines powering the **SAHAYAK Autonomous Disaster Intelligence Platform**.

---

## 📁 Directory Structure

```
data/
├── README.md                          <- This document
├── silent_zone_dataset.csv            <- Master disaster & telecom training dataset (10,000 records)
├── shelter_resource_dataset.csv       <- Multi-commodity shelter network dataset (5,000 records)
├── raw/                               <- Raw telemetry ingestion staging
└── processed/                         <- Processed, normalized & encoded feature matrices
```

---

## 1. Telecom Silent Zone & Disaster Impact Dataset (`silent_zone_dataset.csv`)

### Geographic Scope
- Geo-bounded strictly within Indian territory:
  - **Latitude**: $6.0^\circ\text{ N} - 38.0^\circ\text{ N}$
  - **Longitude**: $68.0^\circ\text{ E} - 98.0^\circ\text{ E}$
- Major disaster corridors: Odisha (Cyclone), Assam & Bihar (Floods), Uttarakhand & Gujarat (Earthquakes), Kerala (Landslides/Floods).

### Feature Dictionary

| Column Name | Type | Range / Units | Description |
| :--- | :--- | :--- | :--- |
| `latitude` | Float | $6.0 - 38.0^\circ\text{N}$ | Geographic latitude coordinate |
| `longitude` | Float | $68.0 - 98.0^\circ\text{E}$ | Geographic longitude coordinate |
| `disaster_type` | Categorical | `flood`, `cyclone`, `earthquake` | Primary active natural hazard |
| `rainfall_mm` | Float | $0 - 1000\text{ mm}$ | 24-hour accumulated precipitation |
| `wind_speed_kmph` | Float | $0 - 350\text{ km/h}$ | Sustained wind velocity |
| `wind_gust_kmph` | Float | $0 - 450\text{ km/h}$ | Peak 3-second kinetic wind gust |
| `earthquake_magnitude` | Float | $0.0 - 10.0$ | Moment magnitude scale ($M_w$) |
| `flood_depth_m` | Float | $0.0 - 15.0\text{ m}$ | Inundation water depth |
| `storm_surge_m` | Float | $0.0 - 12.0\text{ m}$ | Penetrative coastal storm surge height |
| `population_density` | Float | People / $\text{km}^2$ | Local demographic density |
| `distance_to_tower_km`| Float | $0 - 100\text{ km}$ | Distance to nearest cellular transceiver |
| `tower_density` | Float | Towers / $\text{km}^2$ | Cellular mast spatial density |
| `network_signal_dbm` | Float | $-140 \text{ to } -30\text{ dBm}$| Cellular Received Signal Strength (RSSI) |
| `power_availability` | Binary | `0` or `1` | Grid electricity status ($1=\text{Active}, 0=\text{Blackout}$) |
| `road_access` | Binary | `0` or `1` | Road corridor transit status ($1=\text{Passable}, 0=\text{Blocked}$) |
| `terrain_elevation_m` | Float | $-10 \text{ to } 8000\text{ m}$ | Digital Elevation Model (DEM) altitude |
| `soil_liquefaction_risk`| Float | $0.0 - 1.0$ | Geological soil liquefaction probability |
| `structural_vulnerability_index` | Float | $0.0 - 1.0$ | Housing & infrastructure fragility index |
| `battery_reserve_hours` | Float | $0 - 120\text{ hrs}$ | Tower backup generator / battery life |
| `historical_outage_count` | Integer | Count | Previous outage count in past 24 months |
| `emergency_calls_count` | Integer | Count | Surge emergency call volume per hour |
| `tower_operational_percentage` | Float | $0 - 100\%$ | Operational telemetry percentage |
| `network_congestion_percentage` | Float | $0 - 100\%$ | Radio access network buffer occupancy |
| **`is_silent_zone`** *(Target 1)* | Binary | `0` or `1` | **Engine 1 Target**: 1 = Silent Zone, 0 = Normal |
| **`physical_damage_score`** *(Target 2)*| Float | $0 - 100$ | **Engine 2 Target**: Physical destruction score |
| **`affected_radius_km`** *(Target 2)*| Float | $0.5 - 150\text{ km}$ | **Engine 2 Target**: Kinetic hazard impact radius |
| **`time_to_blackout_hours`** *(Target 2)*| Float | $0.1 - 72\text{ hrs}$ | **Engine 2 Target**: Time until full battery depletion |

---

## 2. Shelter Resource & Supply Redistribution Dataset (`shelter_resource_dataset.csv`)

### Humanitarian Standards (NDMA India & Sphere Project Handbook)
- **Food Rations**: $1.5\text{ kg/person/day}$ ($\sim 2,100\text{ kcal}$)
- **Drinking Water**: $3.5\text{ L/person/day}$ (Potable + essential sanitation)
- **Medical Trauma Kits**: $2.0\text{ kits per 100 persons}$ (weighted by demographic vulnerability ratio)
- **Blankets**: $1.0\text{ blanket per person}$

### Feature Dictionary

| Column Name | Type | Range / Units | Description |
| :--- | :--- | :--- | :--- |
| `shelter_id` | String | `SH-XXX-XXXX` | Unique shelter identifier code |
| `shelter_name` | String | Text | Shelter designation and sector |
| `district` | String | Text | Administrative district (e.g. Puri, Darbhanga) |
| `state` | String | Text | State in India |
| `hazard_type` | Categorical | `cyclone`, `flood`, `earthquake` | Active hazard in the district |
| `capacity_people` | Integer | $100 - 5,000$ | Maximum human occupancy capacity |
| `current_occupancy` | Integer | $10 - 6,000$ | Current displaced persons sheltered |
| `vulnerable_ratio` | Float | $0.05 - 0.70$ | Proportion of children, elderly, and injured |
| `days_isolated` | Float | $0 - 30\text{ days}$ | Days cut off from regional supply chains |
| `food_rations_kg` | Float | $\text{kg}$ | Available food stock |
| `water_liters` | Float | $\text{Liters}$ | Available potable drinking water |
| `medical_kits` | Float | Units | Available trauma & first-aid kits |
| `blankets_count` | Float | Units | Available thermal blankets |
| **`target_shortage_score`** | Float | $0 - 100$ | **Engine 3 Target**: Compound resource shortage score |
| **`target_urgency_class`** | Categorical | `CRITICAL_DEFICIT`, `DEFICIT`, `BALANCED`, `SURPLUS` | **Engine 3 Target**: Urgency tier |

---

## 3. Dataset Generation & Master Training

To regenerate or scale the datasets, run:

```bash
# Trains all 3 engines and generates synthetic datasets automatically:
python train_all.py
```
