"""
Silent Zone & Disaster Impact Intelligence - System Configuration
Centralized configuration parameters, severity thresholds, feature sets, and tactical response mappings.
"""

from typing import Dict, List, Tuple, Any
import os

# Base directory paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(DATA_DIR, "silent_zone_dataset.csv")

# Model serialization artifacts
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "silent_zone_model.pkl")
IMPACT_MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "disaster_impact_model.pkl")
SHELTER_MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "shelter_resource_model.pkl")
SHELTER_DATASET_PATH = os.path.join(DATA_DIR, "shelter_resource_dataset.csv")

# Humanitarian Standards per Person per Day (Sphere Handbook / NDMA India)
SHELTER_HUMANITARIAN_STANDARDS = {
    "food_rations_kg_per_person_day": 1.5,       # Approx 2,100 kcal
    "water_liters_per_person_day": 3.5,          # Drinking + basic sanitation
    "medical_kits_per_100_people": 2.0,          # First aid & emergency trauma
    "blankets_per_person": 1.0,
    "safe_buffer_days": 3.0,                     # Minimum inventory buffer before surplus declaration
    "critical_threshold_days": 1.5,              # Critical deficit if below this threshold
}

# Reproducibility Seed
RANDOM_SEED = 42

# India Geo-Boundary Bounding Box
GEO_BOUNDS = {
    "lat_min": 6.0,
    "lat_max": 38.0,
    "lon_min": 68.0,
    "lon_max": 98.0,
}

# Categorical Feature
CATEGORICAL_FEATURES = ["disaster_type"]

# Raw Baseline Telecom & Meteorological Features
NUMERICAL_RAW_FEATURES = [
    "latitude",
    "longitude",
    "rainfall_mm",
    "wind_speed_kmph",
    "wind_gust_kmph",
    "earthquake_magnitude",
    "flood_depth_m",
    "storm_surge_m",
    "population_density",
    "distance_to_tower_km",
    "tower_density",
    "network_signal_dbm",
    "power_availability",
    "road_access",
    "terrain_elevation_m",
    "soil_liquefaction_risk",
    "structural_vulnerability_index",
    "battery_reserve_hours",
    "historical_outage_count",
    "emergency_calls_count",
    "tower_operational_percentage",
    "network_congestion_percentage",
    "distance_to_nearest_hospital_km",
    "distance_to_nearest_relief_camp_km",
    "historical_disaster_frequency",
]

# Engineered Features for Silent Zone Detection (Engine 1)
ENGINEERED_SILENT_FEATURES = [
    "communication_risk_score",
    "infrastructure_failure_risk",
    "population_pressure",
    "disaster_intensity",
    "network_failure_risk",
    "accessibility_risk",
]

# Engineered Features for Physical Disaster Impact (Engine 2)
ENGINEERED_PHYSICS_FEATURES = [
    "wind_kinetic_stagnation_pressure",
    "hydrodynamic_inundation_drag",
    "seismic_shear_vulnerability",
    "peukert_battery_decay_rate",
    "built_environment_fragility",
]

ALL_RAW_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_RAW_FEATURES
TARGET_SILENT_ZONE = "silent_zone"

# Aliases for backward compatibility with train_model.py imports
ENGINEERED_FEATURES = ENGINEERED_SILENT_FEATURES   # Engine 1 alias
TARGET_COLUMN = TARGET_SILENT_ZONE                 # Engine 1 target alias
TARGET_PHYSICAL_DAMAGE = "physical_damage_score"
TARGET_IMPACT_SEVERITY = "impact_severity"
TARGET_TIME_TO_BLACKOUT = "time_to_blackout_hours"
TARGET_AFFECTED_RADIUS = "affected_radius_km"

# Model 1: Silent Zone Probability Severity Tiers
SEVERITY_THRESHOLDS: List[Tuple[float, float, str]] = [
    (0.00, 0.40, "LOW"),
    (0.40, 0.60, "MEDIUM"),
    (0.60, 0.80, "HIGH"),
    (0.80, 1.00, "CRITICAL"),
]

# Model 2: Physical Disaster Impact Tiers (0 - 100 Score)
IMPACT_SEVERITY_THRESHOLDS: List[Tuple[float, float, str]] = [
    (0.00, 30.00, "MINOR"),
    (30.00, 60.00, "MODERATE"),
    (60.00, 80.00, "SEVERE"),
    (80.00, 100.00, "CATASTROPHIC"),
]

# Layer 4: Compound Disaster Threat Index (CDTI) Tiers (0 - 100 Score)
CDTI_THRESHOLDS: List[Tuple[float, float, str]] = [
    (0.00, 40.00, "LEVEL 1: LOW RISK"),
    (40.00, 60.00, "LEVEL 2: MODERATE DANGER"),
    (60.00, 80.00, "LEVEL 3: HIGH SEVERITY"),
    (80.00, 100.00, "LEVEL 4: CRITICAL EMERGENCY"),
]


def get_severity_level(probability: float) -> str:
    """Maps predicted Silent Zone probability to a communication severity tier."""
    try:
        import numpy as np
        prob = float(probability)
        if np.isnan(prob):
            return "LOW"
    except Exception:
        prob = 0.0

    prob = max(0.0, min(1.0, prob))
    if prob < 0.40:
        return "LOW"
    elif prob < 0.60:
        return "MEDIUM"
    elif prob < 0.80:
        return "HIGH"
    else:
        return "CRITICAL"


def get_impact_severity_level(damage_score: float) -> str:
    """Maps physical damage score (0-100) to an impact severity tier."""
    try:
        import numpy as np
        score = float(damage_score)
        if np.isnan(score):
            return "MINOR"
    except Exception:
        score = 0.0

    score = max(0.0, min(100.0, score))
    if score < 30.0:
        return "MINOR"
    elif score < 60.0:
        return "MODERATE"
    elif score < 80.0:
        return "SEVERE"
    else:
        return "CATASTROPHIC"


def get_cdti_level(cdti_score: float) -> str:
    """Maps Compound Disaster Threat Index (0-100) to an emergency alert tier."""
    try:
        import numpy as np
        score = float(cdti_score)
        if np.isnan(score):
            return "LEVEL 1: LOW RISK"
    except Exception:
        score = 0.0

    score = max(0.0, min(100.0, score))
    if score < 40.0:
        return "LEVEL 1: LOW RISK"
    elif score < 60.0:
        return "LEVEL 2: MODERATE DANGER"
    elif score < 80.0:
        return "LEVEL 3: HIGH SEVERITY"
    else:
        return "LEVEL 4: CRITICAL EMERGENCY"


# Emergency Communication Response Protocols
EMERGENCY_ACTIONS: Dict[str, List[str]] = {
    "CRITICAL": [
        "Deploy emergency communication unit (Cell on Wheels - COW)",
        "Deploy portable satellite communication terminals (SATCOM / BGAN)",
        "Dispatch specialized disaster communication & technical rescue team",
        "Activate amateur radio (HAM) emergency backup network",
        "Deploy communication-relay tethered drones for survivors",
        "Prioritize this area for immediate evacuation route guidance"
    ],
    "HIGH": [
        "Deploy mobile emergency communication vehicle (Cell on Wheels)",
        "Reroute cellular traffic through secondary microwave backhaul",
        "Establish satellite-linked relief camp communication hotspot",
        "Send technician team with diesel generators for tower power restoration",
        "Broadcast emergency cell broadcast alerts on backup frequencies"
    ],
    "MEDIUM": [
        "Monitor tower operational telemetry and power fuel reserves",
        "Alert telecom service providers (TSPs) for rapid standby maintenance",
        "Place mobile diesel generators on standby near perimeter roads",
        "Advise public to conserve battery and prioritize SMS over voice calls"
    ],
    "LOW": [
        "Maintain routine continuous network telemetry monitoring",
        "Keep standard emergency protocol on baseline standby"
    ]
}


def get_recommended_actions(severity: str) -> List[str]:
    """Returns protocol recommendations based on severity level."""
    return EMERGENCY_ACTIONS.get(severity.upper(), EMERGENCY_ACTIONS["LOW"])
