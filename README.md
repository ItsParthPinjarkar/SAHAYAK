# SAHAYAK (सहायक): AI-Powered Autonomous Disaster Intelligence Platform

[![FastAPI](https://img.shields.io/badge/FastAPI-2.0+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-blue.svg)](https://github.com/slundberg/shap)
[![Physics-Informed](https://img.shields.io/badge/Physics--Informed-Dynamic%20Drag-blueviolet.svg)](#)
[![Conformal-Prediction](https://img.shields.io/badge/Uncertainty-95%25%20Conformal%20CI-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Problem Statement & Breakthrough Philosophy
During severe natural disasters in India (Cyclones on the Odisha/Andhra/Gujarat coasts, Riverine floods in Bihar/Assam/Kerala, Himalayan earthquakes in Uttarakhand), disaster incident commanders and rescue personnel face multi-faceted life-critical threats:
1. **Telecommunication Collapse ("Silent Zones")**: Cell towers destroyed, electrical grids severed, and backup generator fuel exhausted. Trapped casualties cannot dial **112**, while rescue teams have zero operational visibility.
2. **Physical Destruction & Structural Collapse**: Buildings, culverts, and bridges collapsing under kinetic forces.
3. **Severe Humanitarian Relief Imbalance**: Displaced person shelters rapidly exhaust food and medical rations while neighboring facilities retain surplus stock without an automated redistribution protocol.
4. **Hazardous Evacuation Corridors**: Conventional routing apps navigate evacuees into submerged roads or communication blackout dead-zones.

**SAHAYAK (सहायक)** introduces a unified, cognitive triple-engine AI platform combining:
* **Engine 1 (Silent Zone Blackout Predictor)**: Cell tower failure, signal loss, 95% Conformal Confidence bounds, and Time-to-Blackout (TTB) forecasting.
* **Engine 2 (Physical Destruction & Damage Severity Predictor)**: Aerodynamic wind stagnation pressure ($q = \frac{1}{2}\rho v^2$), hydrodynamic flood drag, seismic ground acceleration, and structural collapse risk.
* **Engine 3 (Shelter Resource Dynamics & Autonomous Supply Redistribution Solver)**: Multi-commodity optimization (food rations, drinking water, trauma kits, blankets) based on NDMA/Sphere standards, solving distance-weighted donor-to-receiver transfer logistics.
* **Layer 4 Compound Threat Synthesis**: Calculates the **Compound Disaster Threat Index (CDTI)**, **Rescue Urgency Index (RUI)**, and **Golden Hour Extrication Clock ($\tau_{1/2}$)**.
* **Dynamic A\* Evacuation Vector Router**: Continuous graph optimization over risk fields, autonomously rejecting trapped corridors.
* **Autonomous Resource Manifest Generator**: Synthesizes military, NDRF, and shelter logistics payload dispatches (COWs, SATCOM terminals, drones, amphibious rescue boats).

```
                               ┌────────────────────────────────────────────────────────┐
                               │           Disaster Early Warning / Sensor Data         │
                               │  (IMD Weather, ISRO/Bhuvan, Telecom Status, Crowdsource)│
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │              Data Preprocessing & Validation           │
                               │        (Missing data handling, India geo-boundary      │
                               │         range checks, Categorical One-Hot Encoding)    │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │       Domain-Specific Feature Engineering Pipeline      │
                               │      - Communication Risk Score                        │
                               │      - Infrastructure Failure Risk                     │
                               │      - Population Pressure                             │
                               │      - Disaster Intensity (Flood/Cyclone/Quake)        │
                               │      - Network Failure Risk & Accessibility Risk       │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │           Machine Learning Model (Random Forest)       │
                               │   - Stratified Split, Balanced Class Weights           │
                               │   - Hyperparameter Optimization (GridSearchCV)         │
                               │   - Recall-Targeted Optimization (Life-Safety Priority)│
                               └─────────────┬───────────────────────────┬──────────────┘
                                             │                           │
                                             ▼                           ▼
                     ┌───────────────────────────────┐   ┌───────────────────────────────┐
                     │     Explainable AI (SHAP)     │   │       Severity & Action       │
                     │  - Feature Importance Ranking │   │        Decision Engine        │
                     │  - Top Local Risk Drivers     │   │  - LOW / MED / HIGH / CRITICAL│
                     │  - Waterfall / Summary Plots  │   │  - Response Protocols (COW,   │
                     │                               │   │    SDR, HAM, Satellite, Drone)│
                     └───────────────┬───────────────┘   └───────────────┬───────────────┘
                                     │                                   │
                                     └─────────────────┬─────────────────┘
                                                       │
                                                       ▼
                       ┌───────────────────────────────────────────────────────────────┐
                       │                   Application & Service Layer                 │
                       ├───────────────────────────────┬───────────────────────────────┤
                       │       FastAPI REST Engine     │       Streamlit Dynamic Map   │
                       │   - /predict & /predict_batch │   - Interactive India Map     │
                       │   - /health & /model_info     │   - Real-Time Parameter Tweak │
                       │   - Pydantic validation       │   - Grid-Cell Silent Zone Map │
                       │                               │   - Evacuation Route Risk     │
                       │                               │   - Rescue Resource Priority  │
                       └───────────────────────────────┴───────────────────────────────┘
```

---

## 4. Project Structure

```
silent-zone-detection/
│
├── data/
│   ├── raw/                           # Raw telemetry archives and provenance
│   ├── processed/                     # Cleaned, imputed, and stratified splits
│   └── silent_zone_dataset.csv        # Comprehensive synthetic dataset (India scenarios)
│
├── models/
│   ├── silent_zone_model.pkl          # Serialized production pipeline artifact
│   ├── model_evaluation_metrics.png   # ROC, PR, and Confusion Matrix charts
│   ├── shap_summary.png               # Global SHAP beeswarm feature importance
│   └── shap_bar_importance.png        # Mean absolute SHAP impact
│
├── notebooks/
│   └── exploratory_analysis.ipynb     # Exploratory analysis & disaster correlations
│
├── src/
│   ├── config.py                      # Centralized parameters, thresholds & action mappings
│   ├── data_preprocessing.py          # Data ingestion, schema validation & encoders
│   ├── feature_engineering.py         # Domain risk indices (zero data leakage)
│   ├── train_model.py                 # Scikit-learn Pipeline training & GridSearchCV
│   ├── evaluate_model.py              # Performance metrics & life-critical recall audit
│   ├── predict.py                     # Inference script with risk drivers & recommendations
│   └── explain_model.py               # SHAP TreeExplainer & local attribution generator
│
├── api/
│   └── main.py                        # FastAPI REST API with Pydantic validation
│
├── dashboard/
│   └── app.py                         # Streamlit interactive operations dashboard
│
├── requirements.txt                   # Environment dependencies
└── README.md                          # Complete system documentation
```

---

## 5. Dataset Schema

The dataset captures 21 raw physical, environmental, infrastructural, and demographic indicators across India:

| Feature Name | Type | Unit / Range | Description |
| :--- | :--- | :--- | :--- |
| `latitude` | float | 6.0 – 38.0 °N | Decimal latitude coordinate within India |
| `longitude` | float | 68.0 – 98.0 °E | Decimal longitude coordinate within India |
| `disaster_type` | string | `flood`, `cyclone`, `earthquake` | Classification of the primary disaster hazard |
| `rainfall_mm` | float | 0.0 – 600.0 mm | 24-hour cumulative precipitation |
| `wind_speed_kmph` | float | 0.0 – 260.0 km/h | Sustained surface wind speed |
| `earthquake_magnitude`| float | 0.0 – 9.5 Richter | Moment magnitude of seismic event |
| `flood_depth_m` | float | 0.0 – 8.0 m | Inundation water depth above surface |
| `population_density` | float | 50 – 25,000 /km² | Human settlement density |
| `distance_to_tower_km`| float | 0.1 – 35.0 km | Radial distance to nearest cellular mast |
| `tower_density` | float | 0.05 – 5.0 /km² | Cell towers per square kilometer |
| `network_signal_dbm` | float | -125.0 to -50.0 dBm| Cellular signal strength (-120 dBm = dead) |
| `power_availability` | int | 0 or 1 | 1 = Grid electrical power active, 0 = Blackout |
| `road_access` | int | 0 or 1 | 1 = Passable road, 0 = Submerged/landslide cut-off |
| `terrain_elevation_m` | float | 1.0 – 4,000.0 m | Elevation above mean sea level |
| `historical_outage_count`| int | 0 – 25 | Historical telecom outage incidents recorded |
| `emergency_calls_count`| int | 0 – 600 calls | Distress calls logged in preceding 2 hours |
| `tower_operational_percentage` | float | 0.0 – 100.0 % | Percentage of masts functioning normally |
| `network_congestion_percentage`| float | 0.0 – 100.0 % | Channel traffic load / saturation level |
| `distance_to_nearest_hospital_km` | float | 0.5 – 50.0 km | Distance to nearest district hospital |
| `distance_to_nearest_relief_camp_km` | float | 0.5 – 40.0 km | Distance to nearest disaster shelter |
| `historical_disaster_frequency` | int | 0 – 20 events | Past decade disaster frequency |
| **`silent_zone`** *(Target)* | int | 0 or 1 | **0 = Normal Zone, 1 = Silent Zone (Blackout)** |

> **Data Provenance**: Sample dataset records are realistic **synthetic demonstration data** crafted to reflect physical vulnerability patterns in India. They do not constitute official classified government telemetry.

---

## 6. Installation & Quick Start

### 6.1 Clone & Setup Environment
```bash
git clone https://github.com/example/silent-zone-detection.git
cd silent-zone-detection

# Create virtual environment (Python 3.10+)
python -m venv venv

# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 6.2 Train the ML Model
```bash
python src/train_model.py
```
This executes dataset validation, creates domain features, conducts stratified cross-validation, executes `GridSearchCV` optimized for **Silent Zone Recall**, prints evaluation metrics, and saves `models/silent_zone_model.pkl`.

### 6.3 Run Model Evaluation
```bash
python src/evaluate_model.py
```
Generates confusion matrix, ROC-AUC curve, and Precision-Recall plots in `models/model_evaluation_metrics.png`.

### 6.4 Execute Single CLI Prediction
```bash
python src/predict.py
```

### 6.5 Launch FastAPI Backend
```bash
uvicorn api.main:app --reload --port 8000
```
Interactive Swagger API documentation will be available at: [http://localhost:8000/docs](http://localhost:8000/docs).

### 6.6 Launch Streamlit Interactive Operations Dashboard
```bash
streamlit run dashboard/app.py
```
Dashboard will open in your browser at: [http://localhost:8501](http://localhost:8501).

---

## 7. API Usage

### Endpoints
- `GET /health`: Health check and system operational state.
- `GET /model_info`: Model architecture, training metrics, and severity configuration.
- `POST /predict`: Single disaster location inference.
- `POST /predict_batch`: Multi-coordinate or grid array batch inference.
- `POST /explain`: SHAP feature contribution breakdown for a given location.

### Example Request (`POST /predict`)
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "latitude": 21.1458,
  "longitude": 79.0882,
  "disaster_type": "flood",
  "rainfall_mm": 280.0,
  "wind_speed_kmph": 45.0,
  "earthquake_magnitude": 0.0,
  "flood_depth_m": 2.5,
  "population_density": 4500,
  "distance_to_tower_km": 14.2,
  "tower_density": 0.25,
  "network_signal_dbm": -112.0,
  "power_availability": 0,
  "road_access": 0,
  "terrain_elevation_m": 310.0,
  "historical_outage_count": 9,
  "emergency_calls_count": 380,
  "tower_operational_percentage": 18.0,
  "network_congestion_percentage": 93.0,
  "distance_to_nearest_hospital_km": 12.0,
  "distance_to_nearest_relief_camp_km": 8.5,
  "historical_disaster_frequency": 6
}'
```

### Example Response
```json
{
  "latitude": 21.1458,
  "longitude": 79.0882,
  "disaster_type": "flood",
  "prediction": "SILENT_ZONE",
  "silent_zone_probability": 0.92,
  "severity": "CRITICAL",
  "risk_factors": [
    "Critically weak network signal (<= -105 dBm)",
    "Power unavailable",
    "Severe cellular tower collapse (<= 30% operational)",
    "High tower distance",
    "High flood depth",
    "Torrential rainfall",
    "Road access blocked preventing repair crews",
    "Extreme network congestion",
    "High historical outage frequency",
    "Sparse cellular tower infrastructure"
  ],
  "recommended_action": [
    "Deploy emergency communication unit (Cell on Wheels - COW)",
    "Deploy portable satellite communication terminals (SATCOM / BGAN)",
    "Dispatch specialized disaster communication & technical rescue team",
    "Activate amateur radio (HAM) emergency backup network",
    "Deploy communication-relay tethered drones for survivors",
    "Prioritize this area for immediate evacuation route guidance"
  ]
}
```

---

## 8. Severity Classification System

Probability thresholds are centrally configured in `src/config.py`:

| Probability Range | Severity Tier | Operational Meaning | Automated Action Protocol |
| :---: | :---: | :--- | :--- |
| **0.80 – 1.00** | **CRITICAL** | Total telecommunication collapse; catastrophic isolation | Deploy COWs, SATCOM terminals, HAM radio squads, and tethered relay drones |
| **0.60 – 0.79** | **HIGH** | Severe degradation; intermittent carrier dropouts | Reroute microwave links, dispatch technician diesel generators, standby COWs |
| **0.40 – 0.59** | **MEDIUM** | Heightened vulnerability; network congestion surges | Continuous telemetry surveillance, alert telecom service providers (TSPs) |
| **0.00 – 0.39** | **LOW** | Normal operating state; resilient backup power | Standard baseline monitoring |

---

## 9. Explainable AI (SHAP)
Using `shap.TreeExplainer`, the system eliminates the "black box" nature of tree ensembles, giving disaster managers clear justifications:
- **Global Explanations**: Produces SHAP beeswarm and feature impact plots (`models/shap_summary.png`).
- **Local Explanations**: For every detected Silent Zone, the API and dashboard extract the top-5 positive contributors (e.g., `network_signal_dbm`, `tower_operational_percentage`, `power_availability`).
- **Disaster Hazard Specifics**:
  - **Floods**: Driven primarily by `flood_depth_m`, `terrain_elevation_m`, and `road_access` (equipment submersion & stranded repair boats).
  - **Cyclones**: Driven by `wind_speed_kmph`, `power_availability`, and `tower_operational_percentage` (sheared antennas & toppled transmission pylons).
  - **Earthquakes**: Driven by `earthquake_magnitude` and severed backhaul conduits.

---

## 10. Evacuation Routing Integration

The Silent Zone system serves as an early-stage gatekeeper for multi-modal evacuation routing engines:
- **Route Feasibility Formula**:
  $$\text{Route Danger} = w_1 \times \text{Hazard Risk} + w_2 \times \text{Silent Zone Risk} + w_3 \times (1 - \text{Road Access})$$
- Even if a highway is physically unflooded, if it traverses a **CRITICAL Silent Zone**, incident commanders cannot track stranded buses or broadcast reroute orders. Such routes are classified as **NOT RECOMMENDED**.

---

## 11. Resource Prioritization Engine

To optimize scarce NDRF / SDRF assets, the system scores zones dynamically:
$$\text{Priority Score} = \frac{\text{Population Density} \times \text{Disaster Severity Weight} \times P(\text{Silent Zone})}{(\text{Road Access} + 0.1) \times \text{Distance to Relief Camp}}$$
This score ranks allocations for:
1. Mobile Cell on Wheels (COW) units
2. Portable INMARSAT / BGAN SATCOM terminals
3. High-clearance amphibious rescue boats
4. HAM Radio disaster communication teams
5. Water, ration, and trauma emergency kits

---

## 12. Authoritative Indian Data Sources vs. Demonstration Data

In an operational deployment, this prototype integrates with authoritative Indian government and enterprise APIs:
- **Meteorological Data**: India Meteorological Department (IMD) Automatic Weather Stations (AWS) & Doppler Weather Radar network.
- **Geospatial & Flood Inundation**: ISRO Bhuvan Geo-portal, National Remote Sensing Centre (NRSC) flood mapping.
- **Topography & Elevation**: CartoDEM / Bhuvan 30m Digital Elevation Models.
- **Disaster Incident Reporting**: National Disaster Management Authority (NDMA) & State Disaster Management Authorities (SDMA).
- **Telecom Infrastructure**: Department of Telecommunications (DoT) and Telecom Regulatory Authority of India (TRAI) Central Equipment Identity Register / TSP NOC telemetry.
- **Demographic Baselines**: Census of India & BharatMap.

> **Demonstration Boundary**: In this repository, synthetic scenarios represent these physical behaviors for research and hackathon demonstration without infringing on proprietary telecom data or claiming live government API feeds.

---

## 13. Limitations & Disclaimers
1. **Decision Support Only**: This system is a statistical and probabilistic decision-support tool. Predictions must never supersede tactical ground intelligence from local district administration and NDRF reconnaissance teams.
2. **Synthetic Training Baseline**: The default model is trained on curated synthetic distributions. Production deployment requires calibration on historical disaster telemetry (e.g., Cyclone Fani, 2018 Kerala Floods, 2001 Bhuj Earthquake).
3. **Distribution Shifts**: Extreme multi-hazard compounding events (e.g., simultaneous dam breach and cyclone landfall) may fall outside standard feature distributions.
4. **Telecom Data Privacy**: Live operationalization requires authorized DoT access to tower telemetry under lawful emergency disaster declarations.
