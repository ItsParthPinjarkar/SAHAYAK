# 🛡️ SAHAYAK — Comprehensive Project Documentation & Technical Architecture

**SAHAYAK (सहायक)** is an Autonomous Multi-Engine AI Platform for Real-Time Disaster Impact Assessment, Telecommunication Silent Zone Prediction, Golden-Hour Evacuation Vector Routing, and Dynamic Shelter Resource Redistribution during Natural Hazards in India.

---

## 1. System Overview & Problem Statement

During catastrophic floods, severe tropical cyclones, and high-magnitude earthquakes in India:
1. **Telecommunication Collapse**: Mobile towers suffer power grid failure, backhaul fiber disconnection, physical structural collapse, or extreme network traffic congestion — creating **Silent Zones** where trapped citizens cannot call for rescue and first responders cannot communicate.
2. **Delayed Impact Estimation**: Rescue agencies lack immediate granular estimates of physical destruction, hazard radius, and time-to-blackout battery limits.
3. **Imbalanced Humanitarian Relief**: While some relief shelters become overwhelmed and run out of rations within hours, neighboring donor shelters sit on days of surplus supplies without an automated redistribution protocol.
4. **Dangerous Evacuation Corridors**: Conventional routing apps navigate evacuees into submerged roads or communication blackout dead-zones.

**SAHAYAK** solves these critical challenges with a synchronized Triple-Engine Cognitive AI System.

---

## 2. Technical Architecture & Engine Breakdown

```
                             ┌───────────────────────────────────────┐
                             │       LIVE DISASTER TELEMETRY        │
                             │ (IMD, NDMA, CPCB, Telecom Masts, GIS)  │
                             └──────────────────┬────────────────────┘
                                                │
                                                ▼
                             ┌───────────────────────────────────────┐
                             │     SAHAYAK COGNITIVE DATA FUSION     │
                             │ (Geo-Bounded Feature Engineering)     │
                             └───────┬──────────┬───────────┬────────┘
                                     │          │           │
           ┌─────────────────────────┘          │           └──────────────────────────┐
           ▼                                    ▼                                      ▼
┌─────────────────────────┐        ┌─────────────────────────┐        ┌──────────────────────────────┐
│        ENGINE 1         │        │        ENGINE 2         │        │           ENGINE 3           │
│  Telecom Silent Zone    │        │ Physical Destruction &  │        │ Shelter Resource Dynamics &  │
│  Classifier (Ensemble)  │        │ Kinetic Damage Models   │        │ Multi-Commodity Rebalancing  │
└──────────┬──────────────┘        └────────────┬────────────┘        └──────────────┬───────────────┘
           │ (Blackout Prob, 95% CI)            │ (Damage Score, Radius, TTB)        │ (Surplus / Deficit Transfers)
           └─────────────────────────┬──────────┘                                    │
                                     ▼                                               │
                         ┌───────────────────────┐                                   │
                         │        LAYER 4        │                                   │
                         │ Compound Threat Index │                                   │
                         │ (CDTI) & Golden Hour  │                                   │
                         └───────────┬───────────┘                                   │
                                     │                                               │
                                     ▼                                               │
                         ┌───────────────────────┐                                   │
                         │        LAYER 5        │                                   │
                         │ Dynamic A* Multi-Risk │                                   │
                         │  Evacuation Router    │                                   │
                         └───────────┬───────────┘                                   │
                                     │                                               │
                                     ▼                                               ▼
                         ┌───────────────────────────────────────────────────────────┐
                         │             NDMA / MILITARY TACTICAL DISPATCH             │
                         │  FastAPI REST Services ⬩ Streamlit Command Center         │
                         └───────────────────────────────────────────────────────────┘
```

---

## 3. Core Engine Mathematical Formulations

### Engine 1: Telecommunication Silent Zone Ensemble
- **Classifier Architecture**: Weighted Voting Ensemble combining Random Forest, Gradient Boosting, Extra Trees, and XGBoost with Conformal Prediction Intervals (95% CI).
- **Disaster-Critical Metric**: Maximizes Recall for True Silent Zones ($\text{Recall} \ge 0.95$) with optimal decision threshold calibration $\tau^* \approx 0.35 - 0.40$.

### Engine 2: Physics-Informed Kinetic Destruction
- **Wind Stagnation Pressure**:
  $$q = \frac{1}{2} \rho_{\text{air}} v_{\text{gust}}^2$$
- **Hydrodynamic Flood Drag & Hydrostatic Head**:
  $$F_D = \frac{1}{2} C_D \rho_{\text{water}} A v_{\text{flood}}^2, \quad P_H = \rho_{\text{water}} g h_{\text{flood}}$$
- **Peukert Battery Decay & Time-to-Blackout ($T_{\text{blackout}}$)**:
  $$T_{\text{effective}} = T_{\text{rated}} \cdot \left( \frac{I_{\text{rated}}}{I_{\text{surge}}} \right)^{k - 1}$$

### Engine 3: Multi-Commodity Shelter Redistribution Solver
- Calculates Haversine great-circle distance $d(i, j)$ and road impedance factors $R_{ij}$.
- Solves constrained cost-optimal flow:
  $$\min \sum_{i \in \text{Surplus}} \sum_{j \in \text{Deficit}} \left[ d(i, j) \cdot R_{ij} \cdot \text{Urgency}(j) \right] \cdot x_{ij}^{(k)}$$
- Enforces donor safe reserve retention ($\ge 3.0\text{ days}$) and generates automated dispatch manifests with transport mode selection (Truck / Drone / Helo / Amphibious Boat).

---

## 4. API Endpoints Specification

| Method | Route | Description |
| :--- | :--- | :--- |
| `POST` | `/predict/integrated` | Joint dual-engine cognitive inference (CDTI, 95% CI, Golden Hour) |
| `POST` | `/predict/silent_zone` | Standalone telecommunication blackout classifier |
| `POST` | `/predict/impact` | Standalone physical destruction & hazard radius model |
| `POST` | `/evacuation/route` | Dynamic A* multi-hazard potential field vector routing |
| `POST` | `/shelters/analyze` | Single shelter burn rate, buffer days & urgency evaluation |
| `POST` | `/shelters/redistribute`| Dynamic multi-commodity shelter supply chain optimizer |
| `GET` | `/shelters/sample_cluster`| Pre-configured disaster cluster for instant simulation |
| `POST` | `/explain` | SHAP TreeExplainer feature attributions |

---

## 5. Deployment & Execution Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Master Triple-Engine Training
python train_all.py

# 3. Launch FastAPI Backend
uvicorn api.main:app --reload --port 8000

# 4. Launch Command Center Dashboard
streamlit run dashboard/app.py
```

---

## 6. Project Contributors

- **Rishikesh Singh** ([@singhrishikesh1](https://github.com/singhrishikesh1)) — Core Developer & Machine Learning Research
- **Hiya Shaikh** ([@hiyashaikh16](https://github.com/hiyashaikh16)) — Core Developer & System Architecture / Analytics

