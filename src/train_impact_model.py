"""
Silent Zone & Disaster Impact Intelligence - Model 2 Training Pipeline
Trains the Physical Destruction & Impact Severity Model.
Predicts continuous Damage Score (0-100), Affected Geographic Radius (km),
Time-to-Blackout (hours), and 4-tier Impact Severity.
Saves model bundle to models/disaster_impact_model.pkl.
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import (
    DATASET_PATH,
    IMPACT_MODEL_SAVE_PATH,
    RANDOM_SEED,
    CATEGORICAL_FEATURES,
    NUMERICAL_RAW_FEATURES,
    ENGINEERED_PHYSICS_FEATURES,
    TARGET_PHYSICAL_DAMAGE,
    TARGET_AFFECTED_RADIUS,
    TARGET_TIME_TO_BLACKOUT,
    get_impact_severity_level,
)
from feature_engineering_physics import PhysicsInformedImpactEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def build_impact_pipeline(regressor) -> Pipeline:
    """Constructs Scikit-learn Pipeline with Physics Feature Engineering."""
    all_numerical = NUMERICAL_RAW_FEATURES + ENGINEERED_PHYSICS_FEATURES

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, all_numerical),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    full_pipeline = Pipeline([
        ("physics_engineer", PhysicsInformedImpactEngineer()),
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])

    return full_pipeline


def train_disaster_impact_model(
    data_path: str = DATASET_PATH,
    model_save_path: str = IMPACT_MODEL_SAVE_PATH
) -> Dict[str, Any]:
    """
    End-to-end training routine for Disaster Impact Engine:
    1. Loads dataset
    2. Trains multi-target Random Forest Regressor for damage, radius, and time-to-blackout
    3. Evaluates MAE, R2, and derived tier accuracy
    4. Serializes bundle to models/disaster_impact_model.pkl
    """
    logger.info("=== STARTING DISASTER IMPACT & DAMAGE MODEL TRAINING ===")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_RAW_FEATURES

    X = df[feature_cols].copy()
    # Multi-target regression: [physical_damage_score, affected_radius_km, time_to_blackout_hours]
    y = df[[TARGET_PHYSICAL_DAMAGE, TARGET_AFFECTED_RADIUS, TARGET_TIME_TO_BLACKOUT]].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )

    regressor = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    pipeline = build_impact_pipeline(regressor)

    logger.info("Fitting multi-target regression pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # Evaluate Physical Damage Score
    r2_damage = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
    mae_damage = mean_absolute_error(y_test.iloc[:, 0], y_pred[:, 0])

    # Evaluate Affected Radius
    r2_radius = r2_score(y_test.iloc[:, 1], y_pred[:, 1])
    mae_radius = mean_absolute_error(y_test.iloc[:, 1], y_pred[:, 1])

    # Evaluate Time-to-Blackout
    r2_ttb = r2_score(y_test.iloc[:, 2], y_pred[:, 2])
    mae_ttb = mean_absolute_error(y_test.iloc[:, 2], y_pred[:, 2])

    # Evaluate Derived Tier Classification Accuracy
    true_tiers = [get_impact_severity_level(val) for val in y_test.iloc[:, 0]]
    pred_tiers = [get_impact_severity_level(val) for val in y_pred[:, 0]]
    tier_accuracy = accuracy_score(true_tiers, pred_tiers)

    logger.info("\n" + "=" * 55)
    logger.info("MODEL 2 (DISASTER IMPACT & DAMAGE) TEST RESULTS:")
    logger.info(f"Physical Damage Score R2:     {r2_damage:.4f} (MAE: {mae_damage:.2f} pts)")
    logger.info(f"Affected Radius R2:            {r2_radius:.4f} (MAE: {mae_radius:.2f} km)")
    logger.info(f"Time-to-Blackout R2:           {r2_ttb:.4f} (MAE: {mae_ttb:.2f} hrs)")
    logger.info(f"Severity Tier Class Accuracy:  {tier_accuracy * 100:.1f}%")
    logger.info("=" * 55)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "model_type": "MultiTarget_RandomForestRegressor",
        "targets": [TARGET_PHYSICAL_DAMAGE, TARGET_AFFECTED_RADIUS, TARGET_TIME_TO_BLACKOUT],
        "metrics": {
            "r2_damage": float(r2_damage),
            "mae_damage": float(mae_damage),
            "r2_radius": float(r2_radius),
            "mae_radius": float(mae_radius),
            "r2_ttb": float(r2_ttb),
            "mae_ttb": float(mae_ttb),
            "tier_accuracy": float(tier_accuracy),
        },
        "feature_names": feature_cols,
    }

    joblib.dump(bundle, model_save_path)
    logger.info(f"Saved Disaster Impact Model to: {model_save_path}")

    return bundle


if __name__ == "__main__":
    train_disaster_impact_model()
