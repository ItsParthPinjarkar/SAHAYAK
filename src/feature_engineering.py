"""
Silent Zone Detection - Feature Engineering Module
Constructs domain-specific disaster communication risk features without data leakage.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DisasterFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that constructs risk indices:
    1. communication_risk_score
    2. infrastructure_failure_risk
    3. population_pressure
    4. disaster_intensity
    5. network_failure_risk
    6. accessibility_risk

    All operations are row-wise, mathematically bounded to [0, 1], and completely
    isolated from external sample statistics to guarantee zero data leakage.
    """

    def __init__(self):
        super().__init__()
        self.engineered_feature_names = [
            "communication_risk_score",
            "infrastructure_failure_risk",
            "population_pressure",
            "disaster_intensity",
            "network_failure_risk",
            "accessibility_risk",
        ]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit method (stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input DataFrame or array by appending engineered risk features.
        Safely handles missing columns and data type inconsistencies.
        """
        if not isinstance(X, pd.DataFrame):
            try:
                from config import CATEGORICAL_FEATURES, NUMERICAL_RAW_FEATURES
            except ImportError:
                from src.config import CATEGORICAL_FEATURES, NUMERICAL_RAW_FEATURES
            cols = CATEGORICAL_FEATURES + NUMERICAL_RAW_FEATURES
            if hasattr(X, "shape") and X.shape[1] == len(cols):
                df = pd.DataFrame(X, columns=cols)
            else:
                df = pd.DataFrame(X)
        else:
            df = X.copy()

        def safe_col(col_name: str, default: float) -> pd.Series:
            if col_name in df.columns:
                return pd.to_numeric(df[col_name], errors="coerce").fillna(default)
            return pd.Series(default, index=df.index)

        # 1. Signal attenuation normalized risk (0.0 at -50 dBm or better, 1.0 at -120 dBm or worse)
        sig_raw = safe_col("network_signal_dbm", -85.0)
        signal_risk = np.clip((-50.0 - sig_raw) / 70.0, 0.0, 1.0)

        # Distance to tower risk (0 km -> 0.0 risk, >= 20 km -> 1.0 risk)
        dist_tower_risk = np.clip(safe_col("distance_to_tower_km", 2.0) / 20.0, 0.0, 1.0)

        # Tower density deficit (3.0+ towers/km^2 -> 0.0 risk, 0 towers -> 1.0 risk)
        tower_density_deficit = 1.0 - np.clip(safe_col("tower_density", 1.0) / 3.0, 0.0, 1.0)

        # Power loss risk (1 = available -> 0 risk; 0 = unavailable -> 1 risk)
        power_risk = 1.0 - safe_col("power_availability", 1.0)

        # Network congestion normalized
        congestion_norm = np.clip(safe_col("network_congestion_percentage", 40.0) / 100.0, 0.0, 1.0)

        # Outage history risk (>= 15 past outages -> 1.0 risk)
        outage_risk = np.clip(safe_col("historical_outage_count", 2.0) / 15.0, 0.0, 1.0)

        # Tower operational deficit (100% -> 0 risk, 0% -> 1 risk)
        tower_op_deficit = 1.0 - np.clip(safe_col("tower_operational_percentage", 100.0) / 100.0, 0.0, 1.0)

        # Road blockage risk (1 = open -> 0 risk, 0 = blocked -> 1 risk)
        road_block_risk = 1.0 - safe_col("road_access", 1.0)

        # --- FEATURE 1: communication_risk_score ---
        df["communication_risk_score"] = (
            0.25 * signal_risk
            + 0.20 * dist_tower_risk
            + 0.15 * tower_density_deficit
            + 0.20 * power_risk
            + 0.10 * congestion_norm
            + 0.10 * outage_risk
        )

        # --- FEATURE 2: infrastructure_failure_risk ---
        df["infrastructure_failure_risk"] = (
            0.45 * tower_op_deficit
            + 0.35 * power_risk
            + 0.20 * road_block_risk
        )

        # --- FEATURE 3: population_pressure ---
        call_intensity = np.clip(safe_col("emergency_calls_count", 50.0) / 500.0, 0.0, 1.0)
        pop_density_norm = np.clip(safe_col("population_density", 1000.0) / 15000.0, 0.0, 1.0)
        effective_tower_capacity = np.maximum(
            0.05,
            (safe_col("tower_density", 1.0) * (safe_col("tower_operational_percentage", 100.0) / 100.0))
        )
        df["population_pressure"] = np.clip(
            (0.55 * call_intensity + 0.45 * pop_density_norm) / (effective_tower_capacity + 0.25),
            0.0,
            1.0
        )

        # --- FEATURE 4: disaster_intensity ---
        rain_norm = np.clip(safe_col("rainfall_mm", 0.0) / 350.0, 0.0, 1.0)
        wind_norm = np.clip(safe_col("wind_speed_kmph", 0.0) / 200.0, 0.0, 1.0)
        flood_norm = np.clip(safe_col("flood_depth_m", 0.0) / 4.0, 0.0, 1.0)
        quake_norm = np.clip((safe_col("earthquake_magnitude", 0.0) - 4.0) / 4.0, 0.0, 1.0)

        if "disaster_type" in df.columns:
            disaster_types = df["disaster_type"].astype(str).str.lower()
        else:
            disaster_types = pd.Series("flood", index=df.index)

        intensity_series = pd.Series(0.0, index=df.index)

        flood_mask = disaster_types == "flood"
        cyclone_mask = disaster_types == "cyclone"
        quake_mask = disaster_types == "earthquake"

        intensity_series[flood_mask] = (0.50 * rain_norm[flood_mask] + 0.50 * flood_norm[flood_mask])
        intensity_series[cyclone_mask] = (0.65 * wind_norm[cyclone_mask] + 0.35 * rain_norm[cyclone_mask])
        intensity_series[quake_mask] = quake_norm[quake_mask]

        other_mask = ~(flood_mask | cyclone_mask | quake_mask)
        if other_mask.any():
            intensity_series[other_mask] = np.clip(
                (rain_norm[other_mask] + wind_norm[other_mask] + quake_norm[other_mask]) / 3.0,
                0.0,
                1.0
            )

        df["disaster_intensity"] = intensity_series

        # --- FEATURE 5: network_failure_risk ---
        # Mathematical logic: Telecommunication survivability factoring signal blackout,
        # channel saturation, and cell tower loss.
        df["network_failure_risk"] = (
            0.40 * signal_risk
            + 0.35 * tower_op_deficit
            + 0.25 * congestion_norm
        )

        # --- FEATURE 6: accessibility_risk ---
        # Mathematical logic: Physical barrier to emergency deployment (blocked transit,
        # hospital isolation, and relief depot isolation).
        hosp_dist_norm = np.clip(df["distance_to_nearest_hospital_km"] / 30.0, 0.0, 1.0)
        relief_dist_norm = np.clip(df["distance_to_nearest_relief_camp_km"] / 25.0, 0.0, 1.0)
        df["accessibility_risk"] = (
            0.50 * road_block_risk
            + 0.25 * hosp_dist_norm
            + 0.25 * relief_dist_norm
        )

        return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to apply feature engineering on any input DataFrame.
    """
    engineer = DisasterFeatureEngineer()
    return engineer.transform(df)


if __name__ == "__main__":
    import os
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_path = os.path.join(base_dir, "data", "silent_zone_dataset.csv")
    if os.path.exists(dataset_path):
        sample_df = pd.read_csv(dataset_path).head(5)
        fe_df = engineer_features(sample_df)
        print("Feature Engineering Test - Newly Created Columns:")
        for col in [
            "communication_risk_score",
            "infrastructure_failure_risk",
            "population_pressure",
            "disaster_intensity",
            "network_failure_risk",
            "accessibility_risk",
        ]:
            print(f"- {col}: {fe_df[col].values[:3]}")
