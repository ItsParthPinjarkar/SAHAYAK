"""
Silent Zone Detection - Inference and Prediction Module
Predicts communication status, probability, severity level, risk drivers, and recommended response.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Union
import joblib
import pandas as pd
import numpy as np

# Ensure local modules can be found regardless of execution directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from config import MODEL_SAVE_PATH, get_severity_level, get_recommended_actions, CATEGORICAL_FEATURES, NUMERICAL_RAW_FEATURES
except ImportError:
    from src.config import MODEL_SAVE_PATH, get_severity_level, get_recommended_actions, CATEGORICAL_FEATURES, NUMERICAL_RAW_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# Global model cache for performance
_LOADED_MODEL_BUNDLE = None


def load_model(model_path: str = MODEL_SAVE_PATH):
    """Loads and caches the model pipeline. If not found, automatically trains and saves it."""
    global _LOADED_MODEL_BUNDLE
    if _LOADED_MODEL_BUNDLE is None:
        if not os.path.exists(model_path):
            logger.info(f"Model file not found at {model_path}. Auto-training production model from dataset...")
            try:
                from train_model import train_and_tune_model
            except ImportError:
                from src.train_model import train_and_tune_model
            _LOADED_MODEL_BUNDLE = train_and_tune_model(model_save_path=model_path)
        else:
            _LOADED_MODEL_BUNDLE = joblib.load(model_path)
            logger.info(f"Loaded model bundle from {model_path}")
    return _LOADED_MODEL_BUNDLE


def identify_risk_factors(row: Dict[str, Any]) -> List[str]:
    """
    Identifies primary physical and telecom drivers leading to Silent Zone risk.
    """
    risk_factors = []

    # Signal strength
    sig = float(row.get("network_signal_dbm", -80.0))
    if sig <= -105.0:
        risk_factors.append("Critically weak network signal (<= -105 dBm)")
    elif sig <= -95.0:
        risk_factors.append("Weak network signal")

    # Power grid status
    if int(row.get("power_availability", 1)) == 0:
        risk_factors.append("Power unavailable")

    # Tower operational status
    tower_op = float(row.get("tower_operational_percentage", 100.0))
    if tower_op <= 30.0:
        risk_factors.append("Severe cellular tower collapse (<= 30% operational)")
    elif tower_op <= 60.0:
        risk_factors.append("Degraded cellular tower capacity")

    # Distance to tower
    dist_tower = float(row.get("distance_to_tower_km", 2.0))
    if dist_tower >= 12.0:
        risk_factors.append("High tower distance")

    # Disaster specific physical hazards
    dtype = str(row.get("disaster_type", "")).lower()
    flood_depth = float(row.get("flood_depth_m", 0.0))
    wind = float(row.get("wind_speed_kmph", 0.0))
    mag = float(row.get("earthquake_magnitude", 0.0))
    rain = float(row.get("rainfall_mm", 0.0))

    if flood_depth >= 1.5 or (dtype == "flood" and flood_depth >= 1.0):
        risk_factors.append("High flood depth")

    if wind >= 120.0 or (dtype == "cyclone" and wind >= 100.0):
        risk_factors.append("Destructive cyclone winds")

    if rain >= 200.0:
        risk_factors.append("Torrential rainfall")

    if mag >= 6.0 or (dtype == "earthquake" and mag >= 5.5):
        risk_factors.append("High earthquake magnitude")

    # Road & Logistics
    if int(row.get("road_access", 1)) == 0:
        risk_factors.append("Road access blocked preventing repair crews")

    # Congestion & Call Load
    congestion = float(row.get("network_congestion_percentage", 0.0))
    if congestion >= 85.0:
        risk_factors.append("Extreme network congestion")

    # Historical failures
    outages = int(row.get("historical_outage_count", 0))
    if outages >= 6:
        risk_factors.append("High historical outage frequency")

    # Tower Density
    density = float(row.get("tower_density", 1.0))
    if density <= 0.3:
        risk_factors.append("Sparse cellular tower infrastructure")

    if not risk_factors:
        risk_factors.append("Standard operating telemetry within acceptable limits")

    return risk_factors


def predict_silent_zone(input_data: Union[Dict[str, Any], pd.DataFrame], model_path: str = MODEL_SAVE_PATH) -> Dict[str, Any]:
    """
    Executes end-to-end inference on input features.
    Compatible with both old (single pipeline) and new (voting ensemble) bundle formats.
    Returns prediction, probability, severity, risk factors, and recommended response.
    """
    bundle = load_model(model_path)

    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
        single_mode = True
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
        single_mode = False
    else:
        raise ValueError("input_data must be a dictionary or a pandas DataFrame")

    # ── NEW FORMAT: Voting ensemble with optimised threshold ──────────────
    if "all_pipelines" in bundle:
        all_pipes = bundle["all_pipelines"]
        thresh    = bundle.get("optimal_threshold", 0.5)
        xgb_avail = any(name == "xgb" for name, _ in all_pipes)
        w_map = {"rf": 0.35, "gb": 0.25, "et": 0.20}
        if xgb_avail:
            w_map["xgb"] = 0.20
        else:
            w_map["rf"] += 0.10; w_map["et"] += 0.10
        probs = np.zeros(len(df_input))
        for name, pipe in all_pipes:
            probs += w_map.get(name, 0.0) * pipe.predict_proba(df_input)[:, 1]
        preds = (probs >= thresh).astype(int)
    # ── OLD FORMAT: single pipeline ───────────────────────────────────────
    else:
        pipeline = bundle["pipeline"]
        probs    = pipeline.predict_proba(df_input)[:, 1]
        preds    = pipeline.predict(df_input)

    results = []
    for i, (idx, row) in enumerate(df_input.iterrows()):
        prob = float(np.round(probs[i], 4))
        pred_label = "SILENT_ZONE" if preds[i] == 1 else "NORMAL_ZONE"
        severity = get_severity_level(prob)
        risk_factors = identify_risk_factors(row.to_dict())
        actions = get_recommended_actions(severity)

        result = {
            "latitude":               float(row.get("latitude", 0.0)),
            "longitude":              float(row.get("longitude", 0.0)),
            "disaster_type":          str(row.get("disaster_type", "")).lower(),
            "prediction":             pred_label,
            "silent_zone_probability": prob,
            "severity":               severity,
            "risk_factors":           risk_factors,
            "recommended_action":     actions,
            "recommended_actions":    actions,
        }
        results.append(result)

    return results[0] if single_mode else results


if __name__ == "__main__":
    # Example test matching exact user prompt scenario
    sample_disaster_location = {
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
        "historical_disaster_frequency": 6,
    }

    try:
        prediction_output = predict_silent_zone(sample_disaster_location)
        print(json.dumps(prediction_output, indent=2))
    except Exception as e:
        print(f"Prediction error (ensure model is trained first): {e}")
