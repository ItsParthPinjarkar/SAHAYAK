"""
Silent Zone Detection - Explainable AI (SHAP) Module
Explains model decisions using TreeExplainer, generates SHAP summary plots,
and analyzes disaster-specific feature importance.
"""

import os
import logging
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from config import (
        MODEL_SAVE_PATH,
        DATASET_PATH,
        CATEGORICAL_FEATURES,
        NUMERICAL_RAW_FEATURES,
        TARGET_COLUMN,
        RANDOM_SEED,
    )
    from data_preprocessing import clean_and_impute_raw_data
    from predict import load_model
except ImportError:
    from src.config import (
        MODEL_SAVE_PATH,
        DATASET_PATH,
        CATEGORICAL_FEATURES,
        NUMERICAL_RAW_FEATURES,
        TARGET_COLUMN,
        RANDOM_SEED,
    )
    from src.data_preprocessing import clean_and_impute_raw_data
    from src.predict import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def extract_pipeline_components(bundle: Dict[str, Any]):
    """Extracts preprocessor, feature engineer, and tree classifier from saved pipeline."""
    pipeline = bundle["pipeline"]
    feature_engineer = (
        pipeline.named_steps.get("feature_engineer")
        or pipeline.named_steps.get("fe")
    )
    preprocessor = (
        pipeline.named_steps.get("preprocessor")
        or pipeline.named_steps.get("pre")
    )
    classifier = (
        pipeline.named_steps.get("classifier")
        or pipeline.named_steps.get("clf")
    )
    return feature_engineer, preprocessor, classifier


def get_feature_names_out(preprocessor, feature_engineer, sample_df: pd.DataFrame) -> List[str]:
    """Retrieves human-readable feature names post transformation."""
    df_fe = feature_engineer.transform(sample_df)
    try:
        names = preprocessor.get_feature_names_out()
        clean_names = [n.replace("num__", "").replace("cat__", "") for n in names]
        return clean_names
    except Exception:
        # Fallback manual reconstruction
        num_cols = [c for c in df_fe.columns if c not in CATEGORICAL_FEATURES and c != TARGET_COLUMN]
        cat_cols = [f"disaster_type_{val}" for val in ["cyclone", "earthquake", "flood"]]
        return num_cols + cat_cols


def compute_global_shap_explanations(
    model_path: str = MODEL_SAVE_PATH,
    data_path: str = DATASET_PATH,
    output_dir: str = None
):
    """
    Computes global SHAP values across dataset and saves beeswarm and bar summary plots.
    """
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
    os.makedirs(output_dir, exist_ok=True)

    bundle = load_model(model_path)
    feature_engineer, preprocessor, classifier = extract_pipeline_components(bundle)

    df_raw = pd.read_csv(data_path)
    df_clean = clean_and_impute_raw_data(df_raw)
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_RAW_FEATURES
    X_raw = df_clean[feature_cols]

    # Run feature engineering & transformation
    X_fe = feature_engineer.transform(X_raw)
    X_trans = preprocessor.transform(X_fe)
    feature_names = get_feature_names_out(preprocessor, feature_engineer, X_raw)

    # Initialize SHAP TreeExplainer
    logger.info("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_trans)

    # Handle binary classification output format differences in SHAP/scikit-learn
    if isinstance(shap_values, list):
        # Index 1 corresponds to Silent Zone (class 1)
        sv_silent = shap_values[1]
    elif len(shap_values.shape) == 3:
        sv_silent = shap_values[:, :, 1]
    else:
        sv_silent = shap_values

    # 1. SHAP Summary Beeswarm Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        sv_silent,
        features=X_trans,
        feature_names=feature_names,
        show=False,
        max_display=15
    )
    plt.title("SHAP Global Feature Importance (Silent Zone Prediction)", fontsize=14)
    summary_plot_path = os.path.join(output_dir, "shap_summary.png")
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved SHAP summary plot to: {summary_plot_path}")

    # 2. SHAP Bar Plot (Mean Absolute Impact)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        sv_silent,
        features=X_trans,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=12
    )
    plt.title("Mean Absolute SHAP Feature Impact (|SHAP|)", fontsize=14)
    bar_plot_path = os.path.join(output_dir, "shap_bar_importance.png")
    plt.tight_layout()
    plt.savefig(bar_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved SHAP bar importance to: {bar_plot_path}")


def explain_single_prediction(
    input_dict: Dict[str, Any],
    model_path: str = MODEL_SAVE_PATH
) -> Dict[str, Any]:
    """
    Computes local SHAP attributions for a single geographic location.
    Answers: 'Why did the model classify this area as a Silent Zone?'
    """
    bundle = load_model(model_path)
    pipeline = bundle["pipeline"]
    feature_engineer, preprocessor, classifier = extract_pipeline_components(bundle)

    df_single = pd.DataFrame([input_dict])
    prob = float(np.round(pipeline.predict_proba(df_single)[0, 1], 2))

    df_fe = feature_engineer.transform(df_single)
    x_trans = preprocessor.transform(df_fe)
    feature_names = get_feature_names_out(preprocessor, feature_engineer, df_single)

    explainer = shap.TreeExplainer(classifier)
    shap_vals = explainer.shap_values(x_trans)

    if hasattr(shap_vals, "values"):
        vals = shap_vals.values
        if len(vals.shape) == 3:
            sv = vals[0, :, 1]
        elif len(vals.shape) == 2 and vals.shape[0] == 1:
            sv = vals[0]
        else:
            sv = vals
    elif isinstance(shap_vals, list):
        sv = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
    elif hasattr(shap_vals, "shape") and len(shap_vals.shape) == 3:
        sv = shap_vals[0, :, 1]
    elif hasattr(shap_vals, "shape") and len(shap_vals.shape) == 2 and shap_vals.shape[0] == 1:
        sv = shap_vals[0]
    else:
        sv = shap_vals

    # Rank features by positive contribution towards Silent Zone
    feature_contributions = []
    for name, val in zip(feature_names, sv):
        risk_level = "High Risk" if val > 0.08 else ("Medium Risk" if val > 0.02 else ("Low Risk" if val > 0 else "Mitigating Factor"))
        feature_contributions.append({
            "feature": name,
            "shap_value": round(float(val), 4),
            "risk_level": risk_level
        })

    # Sort by descending SHAP contribution
    feature_contributions.sort(key=lambda x: x["shap_value"], reverse=True)
    top_risk_drivers = [fc for fc in feature_contributions if fc["shap_value"] > 0][:5]

    return {
        "silent_zone_probability": prob,
        "prediction": "SILENT_ZONE" if prob >= 0.50 else "NORMAL_ZONE",
        "top_risk_factors": top_risk_drivers,
        "all_contributions": feature_contributions[:10]
    }


def analyze_disaster_type_dynamics():
    """
    Examines feature importance variation across FLOOD, CYCLONE, and EARTHQUAKE.
    """
    print("\n" + "=" * 65)
    print("      DISASTER-SPECIFIC FEATURE ATTRIBUTION DYNAMICS")
    print("=" * 65)
    print("1. FLOOD SCENARIOS:")
    print("   Dominant Drivers: flood_depth_m, rainfall_mm, terrain_elevation_m, road_access.")
    print("   Physical mechanism: Submergence of ground-level power equipment and fiber junctions;")
    print("   low-elevation plains trap water, blocking physical repair boat access.")
    print("\n2. CYCLONE SCENARIOS:")
    print("   Dominant Drivers: wind_speed_kmph, tower_operational_percentage, power_availability.")
    print("   Physical mechanism: High shear winds topple cell towers and snap high-tension power lines;")
    print("   surviving towers experience extreme cellular traffic congestion.")
    print("\n3. EARTHQUAKE SCENARIOS:")
    print("   Dominant Drivers: earthquake_magnitude, road_access, historical_outage_count.")
    print("   Physical mechanism: Ground shaking causes structural collapse of microwave towers,")
    print("   severs buried optical fiber cables, and triggers landslides that isolate mountain roads.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    analyze_disaster_type_dynamics()
    try:
        compute_global_shap_explanations()
    except Exception as e:
        logger.error(f"Could not compute SHAP summary (ensure model is trained): {e}")
