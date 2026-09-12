"""
Antigravity Aegis - FastAPI Cognitive Backend API
Provides asynchronous REST endpoints for:
1. Joint Cognitive Dual-Engine Inference (/predict/integrated)
2. Standalone Physical Disaster Impact & Damage (/predict/impact)
3. Standalone Silent Zone Telecommunication Blackout (/predict/silent_zone & /predict)
4. Dynamic A* Multi-Hazard Evacuation Vector Routing (/evacuation/route)
"""

import os
import sys
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.config import (
    MODEL_SAVE_PATH,
    IMPACT_MODEL_SAVE_PATH,
    SEVERITY_THRESHOLDS,
    IMPACT_SEVERITY_THRESHOLDS,
    CDTI_THRESHOLDS,
)
from src.predict import predict_silent_zone, load_model
from src.integrated_engine import predict_integrated_disaster_threat, load_impact_model
from src.evacuation_router import generate_evacuation_routes
from src.explain_model import explain_single_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Antigravity Aegis: Autonomous Disaster Intelligence Platform",
    description="State-of-the-Art Dual-Engine AI predicting Physical Destruction, Telecommunication Blackout, Conformal Confidence Bounds, and Dynamic A* Evacuation Corridors.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ComprehensiveDisasterRequest(BaseModel):
    latitude: float = Field(..., ge=6.0, le=38.0, description="Latitude inside India territory (6.0 - 38.0 N)")
    longitude: float = Field(..., ge=68.0, le=98.0, description="Longitude inside India territory (68.0 - 98.0 E)")
    disaster_type: str = Field(..., description="Hazard: 'flood', 'cyclone', or 'earthquake'")
    rainfall_mm: float = Field(0.0, ge=0.0, le=1000.0, description="24-hour rainfall in mm")
    wind_speed_kmph: float = Field(0.0, ge=0.0, le=350.0, description="Sustained wind speed in km/h")
    wind_gust_kmph: float = Field(0.0, ge=0.0, le=450.0, description="3-second peak wind gust in km/h")
    earthquake_magnitude: float = Field(0.0, ge=0.0, le=10.0, description="Richter magnitude")
    flood_depth_m: float = Field(0.0, ge=0.0, le=15.0, description="Inundation water depth in meters")
    storm_surge_m: float = Field(0.0, ge=0.0, le=12.0, description="Penetrative coastal storm surge depth in meters")
    population_density: float = Field(1000.0, ge=0.0, description="People per square kilometer")
    distance_to_tower_km: float = Field(2.0, ge=0.0, le=100.0, description="Distance to closest cellular mast in km")
    tower_density: float = Field(1.0, ge=0.0, le=20.0, description="Masts per square kilometer")
    network_signal_dbm: float = Field(-80.0, ge=-140.0, le=-30.0, description="Cellular signal strength in dBm")
    power_availability: int = Field(1, ge=0, le=1, description="Grid electricity status (1 = Active, 0 = Blackout)")
    road_access: int = Field(1, ge=0, le=1, description="Road transit status (1 = Passable, 0 = Blocked/Cut off)")
    terrain_elevation_m: float = Field(50.0, ge=0.0, le=9000.0, description="Elevation above mean sea level in meters")
    soil_liquefaction_risk: float = Field(0.2, ge=0.0, le=1.0, description="Soil bearing capacity loss / liquefaction index")
    structural_vulnerability_index: float = Field(0.5, ge=0.0, le=1.0, description="Fraction of kutcha / unreinforced masonry housing")
    battery_reserve_hours: float = Field(12.0, ge=0.0, le=120.0, description="Remaining backup battery runtime in hours")
    historical_outage_count: int = Field(2, ge=0, le=100, description="Past telecom failure incidents")
    emergency_calls_count: int = Field(50, ge=0, description="Distress calls logged in past 2 hours")
    tower_operational_percentage: float = Field(100.0, ge=0.0, le=100.0, description="Operational cell tower percentage")
    network_congestion_percentage: float = Field(40.0, ge=0.0, le=100.0, description="Channel saturation percentage")
    distance_to_nearest_hospital_km: float = Field(5.0, ge=0.0, le=150.0, description="Distance to district hospital in km")
    distance_to_nearest_relief_camp_km: float = Field(4.0, ge=0.0, le=150.0, description="Distance to nearest disaster camp in km")
    historical_disaster_frequency: int = Field(3, ge=0, le=50, description="Disaster frequency over past decade")

    @field_validator("disaster_type")
    @classmethod
    def validate_disaster_type(cls, value: str) -> str:
        val = value.strip().lower()
        allowed = {"flood", "cyclone", "earthquake"}
        if val not in allowed:
            raise ValueError(f"disaster_type must be one of {allowed}, received '{value}'")
        return val


class RouteOptimizationRequest(BaseModel):
    origin_latitude: float
    origin_longitude: float
    destination_latitude: float
    destination_longitude: float
    epicenter_damage_score: float = 85.0
    epicenter_silent_probability: float = 0.90


@app.on_event("startup")
def startup_event():
    """Preload models on server boot."""
    try:
        load_model(MODEL_SAVE_PATH)
        load_impact_model(IMPACT_MODEL_SAVE_PATH)
        logger.info("Antigravity Aegis: Both intelligence models loaded successfully.")
    except Exception as e:
        logger.warning(f"Startup warm-up warning: {e}")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, Any]:
    return {
        "status": "online",
        "platform": "Antigravity Aegis Autonomous Disaster Intelligence",
        "engine_1_silent_zone_loaded": os.path.exists(MODEL_SAVE_PATH),
        "engine_2_impact_model_loaded": os.path.exists(IMPACT_MODEL_SAVE_PATH),
        "capabilities": [
            "Physics-Informed Dynamic Stagnation & Drag Modeling",
            "95% Conformal Confidence Uncertainty Quantification",
            "Golden Hour Survival Clock (tau_1/2)",
            "Dynamic A* Multi-Hazard Potential Field Router",
            "Autonomous NDRF / Military Resource Manifest Synthesizer"
        ],
    }


@app.get("/model_info", status_code=status.HTTP_200_OK)
def model_info() -> Dict[str, Any]:
    """Returns technical metadata for both models."""
    try:
        b_silent = load_model(MODEL_SAVE_PATH)
        b_impact = load_impact_model(IMPACT_MODEL_SAVE_PATH)
        return {
            "engine_1_blackout": {
                "model_type": b_silent.get("model_type", "RandomForest"),
                "metrics": b_silent.get("metrics", {}),
                "severity_tiers": {lbl: f"{l:.2f}-{h:.2f}" for l, h, lbl in SEVERITY_THRESHOLDS},
            },
            "engine_2_impact": {
                "model_type": b_impact.get("model_type", "MultiTarget_RandomForestRegressor"),
                "metrics": b_impact.get("metrics", {}),
                "impact_tiers": {lbl: f"{l:.1f}-{h:.1f}" for l, h, lbl in IMPACT_SEVERITY_THRESHOLDS},
            },
            "compound_threat_cdti_tiers": {lbl: f"{l:.1f}-{h:.1f}" for l, h, lbl in CDTI_THRESHOLDS},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model metadata error: {str(e)}")


@app.post("/predict/integrated")
def predict_integrated(request: ComprehensiveDisasterRequest):
    """
    Primary Flagship Endpoint: Joint Cognitive Dual-Engine Intelligence.
    Synthesizes Physical Destruction + Communication Blackout + Conformal Uncertainty +
    Golden Hour Survival Clock + Tactical Resource Manifest.
    """
    try:
        payload = request.model_dump()
        result = predict_integrated_disaster_threat(payload)
        return result
    except Exception as e:
        logger.error(f"Integrated prediction failure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Joint inference error: {str(e)}")


@app.post("/predict/silent_zone")
@app.post("/predict")
def predict_silent_zone_endpoint(request: ComprehensiveDisasterRequest):
    """Standalone Engine 1 (Silent Zone Telecommunication Blackout)."""
    try:
        payload = request.model_dump()
        result = predict_silent_zone(payload, model_path=MODEL_SAVE_PATH)
        return result
    except Exception as e:
        logger.error(f"Silent zone prediction failure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Engine 1 error: {str(e)}")


@app.post("/predict/impact")
def predict_impact_endpoint(request: ComprehensiveDisasterRequest):
    """Standalone Engine 2 (Physical Disaster Impact & Damage Score)."""
    try:
        payload = request.model_dump()
        result = predict_integrated_disaster_threat(payload)
        return {
            "physical_damage_score": result["engine_2_physical_destruction"]["physical_damage_score"],
            "impact_severity": result["engine_2_physical_destruction"]["impact_severity"],
            "affected_hazard_radius_km": result["engine_2_physical_destruction"]["affected_hazard_radius_km"],
            "structural_housing_collapse_risk_pct": result["engine_2_physical_destruction"]["structural_housing_collapse_risk_pct"],
        }
    except Exception as e:
        logger.error(f"Impact prediction failure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Engine 2 error: {str(e)}")


@app.post("/evacuation/route")
def route_optimization(request: RouteOptimizationRequest):
    """Dynamic A* Multi-Hazard Potential Field Route Optimization."""
    try:
        routes = generate_evacuation_routes(
            request.origin_latitude,
            request.origin_longitude,
            request.destination_latitude,
            request.destination_longitude,
            request.epicenter_damage_score,
            request.epicenter_silent_probability
        )
        return routes
    except Exception as e:
        logger.error(f"Routing optimization failure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"A* routing error: {str(e)}")


@app.post("/explain")
def explain(request: ComprehensiveDisasterRequest):
    """Generates SHAP feature contribution explanations."""
    try:
        payload = request.model_dump()
        explanation = explain_single_prediction(payload, model_path=MODEL_SAVE_PATH)
        return explanation
    except Exception as e:
        logger.error(f"Explainability failure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"SHAP error: {str(e)}")


class ShelterResourceItem(BaseModel):
    shelter_id: str = Field("SH-001", description="Unique Shelter Code")
    shelter_name: str = Field("Puri Sector 4 Relief Camp", description="Shelter Name")
    district: str = Field("Puri", description="District Name")
    state: Optional[str] = Field("Odisha", description="State Name")
    latitude: float = Field(..., ge=6.0, le=38.0)
    longitude: float = Field(..., ge=68.0, le=98.0)
    capacity_people: int = Field(500, ge=10, le=20000)
    current_occupancy: int = Field(350, ge=0, le=30000)
    vulnerable_ratio: float = Field(0.25, ge=0.0, le=1.0, description="Proportion of children/elderly/injured")
    days_isolated: float = Field(1.0, ge=0.0, le=60.0)
    road_access: int = Field(1, ge=0, le=1, description="1 if passable, 0 if cut off")
    power_backup_hours: float = Field(24.0, ge=0.0)
    food_rations_kg: float = Field(1200.0, ge=0.0)
    water_liters: float = Field(3500.0, ge=0.0)
    medical_kits: float = Field(15.0, ge=0.0)
    blankets_count: float = Field(400.0, ge=0.0)


class ShelterRedistributionRequest(BaseModel):
    shelters: List[ShelterResourceItem]


@app.post("/shelters/analyze")
def analyze_shelter(shelter: ShelterResourceItem):
    """Engine 3: Evaluates a single shelter's inventory burn rate, buffer days, and urgency tier."""
    try:
        from src.shelter_resource_engine import get_shelter_engine
        engine = get_shelter_engine()
        return engine.analyze_shelter(shelter.model_dump())
    except Exception as e:
        logger.error(f"Shelter analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Engine 3 Analysis Error: {str(e)}")


@app.post("/shelters/redistribute")
def redistribute_resources(request: ShelterRedistributionRequest):
    """Engine 3: Solves the dynamic multi-commodity redistribution problem across all shelters in a network."""
    try:
        from src.shelter_resource_engine import get_shelter_engine
        engine = get_shelter_engine()
        shelters_data = [s.model_dump() for s in request.shelters]
        return engine.optimize_redistribution(shelters_data)
    except Exception as e:
        logger.error(f"Shelter redistribution optimization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Engine 3 Optimization Error: {str(e)}")


@app.get("/shelters/sample_cluster")
def get_sample_cluster():
    """Returns a realistic pre-configured Indian cyclone shelter cluster for instant simulation."""
    return [
        {
            "shelter_id": "SH-PURI-01",
            "shelter_name": "Puri Coastal Multi-Hazard Cyclone Shelter",
            "district": "Puri",
            "state": "Odisha",
            "latitude": 19.8135,
            "longitude": 85.8312,
            "capacity_people": 800,
            "current_occupancy": 720,
            "vulnerable_ratio": 0.35,
            "days_isolated": 3.0,
            "road_access": 0,
            "power_backup_hours": 6.0,
            "food_rations_kg": 450.0,      # Only ~0.4 days buffer -> CRITICAL DEFICIT
            "water_liters": 1100.0,        # Only ~0.4 days buffer -> CRITICAL DEFICIT
            "medical_kits": 4.0,
            "blankets_count": 200.0
        },
        {
            "shelter_id": "SH-PURI-02",
            "shelter_name": "Gop Block High School Relief Camp",
            "district": "Puri",
            "state": "Odisha",
            "latitude": 19.9982,
            "longitude": 86.0125,
            "capacity_people": 500,
            "current_occupancy": 320,
            "vulnerable_ratio": 0.20,
            "days_isolated": 1.0,
            "road_access": 1,
            "power_backup_hours": 36.0,
            "food_rations_kg": 3800.0,     # ~7.9 days buffer -> MASSIVE SURPLUS (DONOR)
            "water_liters": 9500.0,        # ~8.4 days buffer -> MASSIVE SURPLUS (DONOR)
            "medical_kits": 25.0,
            "blankets_count": 600.0
        },
        {
            "shelter_id": "SH-PURI-03",
            "shelter_name": "Konark Sun Temple Sector Shelter",
            "district": "Puri",
            "state": "Odisha",
            "latitude": 19.8876,
            "longitude": 86.0945,
            "capacity_people": 600,
            "current_occupancy": 550,
            "vulnerable_ratio": 0.40,
            "days_isolated": 2.5,
            "road_access": 1,
            "power_backup_hours": 12.0,
            "food_rations_kg": 700.0,      # ~0.8 days buffer -> DEFICIT
            "water_liters": 1800.0,        # ~0.9 days buffer -> DEFICIT
            "medical_kits": 6.0,
            "blankets_count": 350.0
        },
        {
            "shelter_id": "SH-PURI-04",
            "shelter_name": "Pipili Central Logistics & Relief Hub",
            "district": "Puri",
            "state": "Odisha",
            "latitude": 20.1147,
            "longitude": 85.8341,
            "capacity_people": 1000,
            "current_occupancy": 450,
            "vulnerable_ratio": 0.15,
            "days_isolated": 0.5,
            "road_access": 1,
            "power_backup_hours": 72.0,
            "food_rations_kg": 6500.0,     # ~9.6 days buffer -> MASSIVE SURPLUS (DONOR)
            "water_liters": 15000.0,       # ~9.5 days buffer -> MASSIVE SURPLUS (DONOR)
            "medical_kits": 40.0,
            "blankets_count": 1200.0
        },
        {
            "shelter_id": "SH-PURI-05",
            "shelter_name": "Satyabadi Panchayat Relief Shelter",
            "district": "Puri",
            "state": "Odisha",
            "latitude": 19.9521,
            "longitude": 85.8239,
            "capacity_people": 400,
            "current_occupancy": 360,
            "vulnerable_ratio": 0.25,
            "days_isolated": 1.5,
            "road_access": 1,
            "power_backup_hours": 24.0,
            "food_rations_kg": 1650.0,     # ~3.0 days buffer -> BALANCED
            "water_liters": 3800.0,        # ~3.0 days buffer -> BALANCED
            "medical_kits": 10.0,
            "blankets_count": 360.0
        }
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
