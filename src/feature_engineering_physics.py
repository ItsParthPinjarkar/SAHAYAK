"""
Silent Zone & Disaster Impact Intelligence - Physics-Informed Feature Engineering
Encodes aerodynamic wind shear, hydrodynamic drag, seismic shear, and battery decay dynamics.
Zero data leakage, row-wise transformations with safe numeric fallbacks.
"""

import sys
import os
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class PhysicsInformedImpactEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms raw environmental & structural features into physics-derived forces:
    1. wind_kinetic_stagnation_pressure
    2. hydrodynamic_inundation_drag
    3. seismic_shear_vulnerability
    4. peukert_battery_decay_rate
    5. built_environment_fragility
    """

    def __init__(self):
        super().__init__()
        self.engineered_physics_features = [
            "wind_kinetic_stagnation_pressure",
            "hydrodynamic_inundation_drag",
            "seismic_shear_vulnerability",
            "peukert_battery_decay_rate",
            "built_environment_fragility",
        ]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        def safe_col(col_name: str, default: float) -> pd.Series:
            if col_name in df.columns:
                return pd.to_numeric(df[col_name], errors="coerce").fillna(default)
            return pd.Series(default, index=df.index)

        # 1. Aerodynamic Wind Stagnation Pressure: q = 0.5 * rho * v^2
        # rho_air = 1.225 kg/m3, wind speed converted from km/h to m/s
        wind_gust = safe_col("wind_gust_kmph", safe_col("wind_speed_kmph", 0.0) * 1.25)
        wind_ms = wind_gust * 0.27778
        q_pressure = 0.5 * 1.225 * (wind_ms ** 2)
        # Normalized by design collapse pressure (~2000 Pa for unreinforced structures)
        df["wind_kinetic_stagnation_pressure"] = np.clip(q_pressure / 2000.0, 0.0, 1.0)

        # 2. Hydrodynamic Flood Drag & Hydrostatic Inundation Head Pressure
        depth = safe_col("flood_depth_m", 0.0)
        surge = safe_col("storm_surge_m", 0.0)
        effective_water_depth = depth + surge * 0.5
        elev = safe_col("terrain_elevation_m", 50.0)
        rain = safe_col("rainfall_mm", 0.0)
        elev_attenuation = np.clip(1.0 - (elev / 1500.0), 0.2, 1.0)
        df["hydrodynamic_inundation_drag"] = np.clip(
            (effective_water_depth / 3.5) * (1.0 + (rain / 400.0)) * elev_attenuation,
            0.0,
            1.0
        )

        # 3. Seismic Shear Ground Acceleration Vulnerability
        mag = safe_col("earthquake_magnitude", 0.0)
        liquefaction = safe_col("soil_liquefaction_risk", 0.2)
        # Richter log-scale progression above baseline M4.0
        seismic_intensity = np.clip((mag - 4.0) / 4.0, 0.0, 1.0) ** 1.5
        df["seismic_shear_vulnerability"] = np.clip(
            seismic_intensity * (1.0 + 0.6 * liquefaction),
            0.0,
            1.0
        )

        # 4. Peukert Battery Decay & Time-to-Blackout Degradation
        battery_hours = safe_col("battery_reserve_hours", 12.0)
        power_avail = safe_col("power_availability", 1.0)
        congestion = safe_col("network_congestion_percentage", 40.0)
        # Discharge multiplier under emergency traffic spike (1.0x to 4.0x)
        discharge_multiplier = 1.0 + 3.0 * (congestion / 100.0)
        effective_runtime = battery_hours / discharge_multiplier
        # High decay rate when battery runtime is under 4 hours and power is dead
        power_loss = 1.0 - power_avail
        df["peukert_battery_decay_rate"] = np.clip(
            power_loss * (1.0 - np.clip(effective_runtime / 16.0, 0.0, 1.0)),
            0.0,
            1.0
        )

        # 5. Built-Environment Structural Fragility
        structural_vuln = safe_col("structural_vulnerability_index", 0.5)
        pop_density = safe_col("population_density", 1000.0)
        pop_norm = np.clip(pop_density / 15000.0, 0.0, 1.0)
        df["built_environment_fragility"] = np.clip(
            0.65 * structural_vuln + 0.35 * pop_norm,
            0.0,
            1.0
        )

        return df


def engineer_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience helper to apply physics-informed feature transformations."""
    engineer = PhysicsInformedImpactEngineer()
    return engineer.transform(df)


if __name__ == "__main__":
    from config import DATASET_PATH
    if os.path.exists(DATASET_PATH):
        sample_df = pd.read_csv(DATASET_PATH).head(5)
        res = engineer_physics_features(sample_df)
        print("Physics-Informed Features Computed:")
        for col in [
            "wind_kinetic_stagnation_pressure",
            "hydrodynamic_inundation_drag",
            "seismic_shear_vulnerability",
            "peukert_battery_decay_rate",
            "built_environment_fragility",
        ]:
            print(f" - {col}: {res[col].values[:3]}")
