"""
Silent Zone Detection - Data Preprocessing Module
Handles data loading, validation, cleaning, imputation, encoding, and train-test splitting.
"""

import os
import logging
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from config import GEO_BOUNDS, CATEGORICAL_FEATURES, NUMERICAL_RAW_FEATURES, TARGET_COLUMN
    INDIA_BOUNDS = GEO_BOUNDS
    NUMERICAL_FEATURES = NUMERICAL_RAW_FEATURES
except ImportError:
    from src.config import GEO_BOUNDS, CATEGORICAL_FEATURES, NUMERICAL_RAW_FEATURES, TARGET_COLUMN
    INDIA_BOUNDS = GEO_BOUNDS
    NUMERICAL_FEATURES = NUMERICAL_RAW_FEATURES

NUMERICAL_FEATURES = [
    "latitude",
    "longitude",
    "rainfall_mm",
    "wind_speed_kmph",
    "earthquake_magnitude",
    "flood_depth_m",
    "population_density",
    "distance_to_tower_km",
    "tower_density",
    "network_signal_dbm",
    "power_availability",
    "road_access",
    "terrain_elevation_m",
    "historical_outage_count",
    "emergency_calls_count",
    "tower_operational_percentage",
    "network_congestion_percentage",
    "distance_to_nearest_hospital_km",
    "distance_to_nearest_relief_camp_km",
    "historical_disaster_frequency",
]

ALL_FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET_COLUMN = "silent_zone"


def validate_dataset_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates whether the dataframe contains all required columns and valid value domains.
    Returns (is_valid, error_messages).
    """
    errors = []
    missing_cols = [col for col in ALL_FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if errors:
        return False, errors

    # Check Indian Geographic Bounds
    out_of_bounds_lat = df[(df["latitude"] < INDIA_BOUNDS["lat_min"]) | (df["latitude"] > INDIA_BOUNDS["lat_max"])]
    if not out_of_bounds_lat.empty:
        errors.append(f"Found {len(out_of_bounds_lat)} records with latitude outside India boundaries (6.0 - 38.0 N)")

    out_of_bounds_lon = df[(df["longitude"] < INDIA_BOUNDS["lon_min"]) | (df["longitude"] > INDIA_BOUNDS["lon_max"])]
    if not out_of_bounds_lon.empty:
        errors.append(f"Found {len(out_of_bounds_lon)} records with longitude outside India boundaries (68.0 - 98.0 E)")

    # Logical Range Validations
    if (df["rainfall_mm"] < 0).any():
        errors.append("Negative rainfall_mm values detected.")
    if (df["wind_speed_kmph"] < 0).any():
        errors.append("Negative wind_speed_kmph values detected.")
    if ((df["earthquake_magnitude"] < 0) | (df["earthquake_magnitude"] > 10)).any():
        errors.append("Earthquake magnitude outside plausible 0.0 - 10.0 Richter range.")
    if (df["flood_depth_m"] < 0).any():
        errors.append("Negative flood_depth_m values detected.")
    if ((df["tower_operational_percentage"] < 0) | (df["tower_operational_percentage"] > 100)).any():
        errors.append("tower_operational_percentage must be between 0.0 and 100.0%.")
    if ((df["network_congestion_percentage"] < 0) | (df["network_congestion_percentage"] > 100)).any():
        errors.append("network_congestion_percentage must be between 0.0 and 100.0%.")
    if (df["network_signal_dbm"] > 0).any():
        errors.append("network_signal_dbm must be negative values in dBm (e.g. -50 to -120 dBm).")

    is_valid = len(errors) == 0
    return is_valid, errors


def clean_and_impute_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw data by removing duplicates and imputing missing values.
    """
    df_clean = df.copy()

    # Drop duplicate records if any
    init_count = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    diff = init_count - len(df_clean)
    if diff > 0:
        logger.info(f"Removed {diff} duplicate records.")

    # Impute missing categorical values
    for col in CATEGORICAL_FEATURES:
        if col in df_clean.columns and df_clean[col].isnull().any():
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "flood"
            df_clean[col] = df_clean[col].fillna(mode_val)

    # Impute missing numerical values with median
    for col in NUMERICAL_FEATURES:
        if col in df_clean.columns and df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    # Standardize string values
    if "disaster_type" in df_clean.columns:
        df_clean["disaster_type"] = df_clean["disaster_type"].astype(str).str.strip().str.lower()

    return df_clean


def build_preprocessor_pipeline() -> ColumnTransformer:
    """
    Builds a Scikit-learn ColumnTransformer for standard preprocessing:
    - Numerical: Median imputation + StandardScaler
    - Categorical: Most frequent imputation + OneHotEncoder
    """
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
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def load_and_preprocess_dataset(
    csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads, validates, cleans, and splits the dataset into stratified train and test sets.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    logger.info(f"Loading raw dataset from {csv_path}")
    df = pd.read_csv(csv_path)

    # Validate Schema
    is_valid, errors = validate_dataset_schema(df)
    if not is_valid:
        logger.warning(f"Dataset validation encountered issues: {errors}")

    # Clean & Impute
    df_cleaned = clean_and_impute_raw_data(df)

    if TARGET_COLUMN not in df_cleaned.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataset.")

    X = df_cleaned[ALL_FEATURE_COLUMNS].copy()
    y = df_cleaned[TARGET_COLUMN].astype(int)

    logger.info(f"Class distribution in target: {dict(y.value_counts())}")

    # Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(f"Train set: {len(X_train)} records, Test set: {len(X_test)} records.")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Self-test execution
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_path = os.path.join(base_dir, "data", "silent_zone_dataset.csv")

    X_train, X_test, y_train, y_test = load_and_preprocess_dataset(dataset_path)

    # Save processed splits for reproducibility
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_path = os.path.join(processed_dir, "train_data.csv")
    test_path = os.path.join(processed_dir, "test_data.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved processed splits to:\n- {train_path}\n- {test_path}")
