"""
Antigravity Aegis - Synthetic Dataset Generator
Generates 5,000 production-quality records spanning floods, cyclones, and earthquakes across India.
Physics-consistent correlations ensure model signals are meaningful and not random.
"""

import os
import sys
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

RNG = np.random.default_rng(42)

N_TOTAL = 5000

# India disaster-prone region bounding boxes (lat_min, lat_max, lon_min, lon_max, weight)
REGIONS = [
    # Coastal Odisha / Andhra (cyclone belt)
    (13.0, 22.0, 79.0, 87.0, 0.30),
    # Gujarat coast (cyclone + earthquake)
    (20.0, 24.5, 68.5, 74.0, 0.18),
    # Assam / Bihar (flood belt)
    (24.0, 28.0, 87.0, 94.0, 0.20),
    # Himachal / Uttarakhand foothills (flood + seismic)
    (28.0, 32.5, 76.0, 81.0, 0.12),
    # Tamil Nadu / Kerala coast (cyclone + flood)
    (8.0, 13.5, 76.5, 80.5, 0.10),
    # Maharashtra (flood + seismic)
    (16.0, 21.0, 73.0, 78.0, 0.10),
]
DISASTER_TYPES = ["cyclone", "flood", "earthquake"]


def sample_lat_lon(n: int) -> tuple:
    weights = np.array([r[4] for r in REGIONS])
    weights /= weights.sum()
    indices = RNG.choice(len(REGIONS), size=n, p=weights)
    lats, lons = [], []
    for idx in indices:
        lat_min, lat_max, lon_min, lon_max, _ = REGIONS[idx]
        lats.append(RNG.uniform(lat_min, lat_max))
        lons.append(RNG.uniform(lon_min, lon_max))
    return np.array(lats), np.array(lons)


def generate_disaster_params(disaster_type: str, n: int, rng: np.random.Generator) -> dict:
    """Generate realistic disaster parameters per type with continuous severity gradients."""
    if disaster_type == "cyclone":
        wind_speed = rng.weibull(2.5, n) * 80 + rng.uniform(60, 180, n)
        wind_speed = np.clip(wind_speed, 40, 280)
        wind_gust = wind_speed * rng.uniform(1.15, 1.35, n)
        rainfall = wind_speed * rng.uniform(0.8, 2.2, n) + rng.exponential(40, n)
        flood_depth = np.clip((rainfall - 100) / 80 + rng.normal(0, 0.3, n), 0.0, 5.5)
        storm_surge = np.clip(wind_speed / 60 * rng.uniform(0.8, 2.0, n), 0.0, 6.0)
        earthquake_magnitude = rng.uniform(0.0, 1.5, n)

    elif disaster_type == "flood":
        rainfall = rng.exponential(120, n) + rng.uniform(80, 400, n)
        rainfall = np.clip(rainfall, 30, 700)
        wind_speed = rng.uniform(5, 55, n)
        wind_gust = wind_speed * rng.uniform(1.1, 1.3, n)
        flood_depth = np.clip(rainfall / 90 + rng.normal(0.5, 0.5, n), 0.0, 7.0)
        storm_surge = rng.uniform(0.0, 0.5, n)
        earthquake_magnitude = rng.uniform(0.0, 1.5, n)

    else:  # earthquake
        earthquake_magnitude = rng.uniform(3.0, 8.5, n)
        rainfall = rng.uniform(0, 120, n)
        wind_speed = rng.uniform(0, 30, n)
        wind_gust = wind_speed * rng.uniform(1.05, 1.2, n)
        flood_depth = np.where(
            earthquake_magnitude > 6.5,
            rng.uniform(0.1, 1.5, n),
            rng.uniform(0.0, 0.3, n),
        )
        storm_surge = rng.uniform(0.0, 0.2, n)

    return {
        "rainfall_mm": rainfall,
        "wind_speed_kmph": wind_speed,
        "wind_gust_kmph": wind_gust,
        "earthquake_magnitude": earthquake_magnitude,
        "flood_depth_m": flood_depth,
        "storm_surge_m": storm_surge,
    }


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Derive target labels using physics-consistent rules."""
    cyclone_mask = (df["disaster_type"] == "cyclone").astype(float)
    flood_mask = (df["disaster_type"] == "flood").astype(float)
    quake_mask = (df["disaster_type"] == "earthquake").astype(float)

    wind_factor = np.clip((df["wind_speed_kmph"] - 60) / 180, 0, 1) ** 1.4
    flood_factor = np.clip(df["flood_depth_m"] / 6.0, 0, 1) ** 1.2
    surge_factor = np.clip(df["storm_surge_m"] / 5.0, 0, 1)
    quake_factor = np.clip((df["earthquake_magnitude"] - 4.0) / 4.0, 0, 1) ** 1.5
    vuln_factor = df["structural_vulnerability_index"]
    liquefaction = df["soil_liquefaction_risk"]
    pop_dense = np.clip(df["population_density"] / 15000, 0, 1)

    raw_damage = (
        cyclone_mask * (55 * wind_factor + 20 * flood_factor + 15 * surge_factor + 10 * vuln_factor)
        + flood_mask * (60 * flood_factor + 20 * vuln_factor + 15 * pop_dense + 5 * wind_factor)
        + quake_mask * (65 * quake_factor + 20 * vuln_factor + 15 * liquefaction)
    ) + RNG.normal(0, 3, len(df))

    df["physical_damage_score"] = np.clip(raw_damage, 0, 100).round(2)

    def tier(score):
        if score < 30:
            return "MINOR"
        elif score < 60:
            return "MODERATE"
        elif score < 80:
            return "SEVERE"
        return "CATASTROPHIC"

    df["impact_severity"] = df["physical_damage_score"].apply(tier)

    raw_radius = (
        cyclone_mask * (40 * wind_factor + 20 * flood_factor + 15 * surge_factor)
        + flood_mask * (30 * flood_factor + 10 * pop_dense + 5 * wind_factor)
        + quake_mask * (50 * quake_factor + 20 * liquefaction)
    ) + RNG.uniform(2, 15, len(df))

    df["affected_radius_km"] = np.clip(raw_radius, 2, 100).round(2)

    battery = df["battery_reserve_hours"]
    power_ok = df["power_availability"]
    congestion = df["network_congestion_percentage"] / 100
    tower_op = df["tower_operational_percentage"] / 100
    battery_decay = battery / (1 + 3 * congestion)

    ttb_raw = np.where(
        power_ok < 0.5,
        battery_decay * 0.5 + RNG.uniform(0, 2, len(df)),
        battery_decay * tower_op + RNG.uniform(0, 6, len(df)),
    )
    ttb_damaged = np.where(
        df["physical_damage_score"] > 70,
        ttb_raw * 0.3,
        ttb_raw,
    )
    df["time_to_blackout_hours"] = np.clip(ttb_damaged, 0.2, 72.0).round(2)

    comm_failure_prob = (
        0.25 * (1 - tower_op)
        + 0.20 * (1 - power_ok)
        + 0.15 * congestion
        + 0.15 * np.clip((df["network_signal_dbm"].abs() - 60) / 60, 0, 1)
        + 0.10 * np.clip(df["distance_to_tower_km"] / 20, 0, 1)
        + 0.08 * (df["physical_damage_score"] / 100)
        + 0.07 * np.clip((df["historical_outage_count"]) / 12, 0, 1)
    )
    comm_failure_prob += RNG.normal(0, 0.05, len(df))
    comm_failure_prob = np.clip(comm_failure_prob, 0, 1)
    df["silent_zone"] = (comm_failure_prob >= 0.5).astype(int)

    return df


def generate_dataset(n: int = N_TOTAL) -> pd.DataFrame:
    lats, lons = sample_lat_lon(n)
    d_types = RNG.choice(DISASTER_TYPES, size=n, p=[0.35, 0.40, 0.25])

    rows = []
    for dtype in DISASTER_TYPES:
        mask = d_types == dtype
        n_d = mask.sum()
        if n_d == 0:
            continue
        params = generate_disaster_params(dtype, n_d, RNG)
        sub = {
            "latitude": lats[mask],
            "longitude": lons[mask],
            "disaster_type": dtype,
            **params,
        }
        sub["population_density"] = RNG.integers(200, 18000, n_d).astype(float)
        sub["distance_to_tower_km"] = np.clip(RNG.exponential(6, n_d), 0.3, 35.0).round(2)
        sub["tower_density"] = np.clip(RNG.normal(1.2, 0.8, n_d), 0.05, 5.0).round(3)
        sub["network_signal_dbm"] = RNG.uniform(-130, -50, n_d).round(1)
        sub["power_availability"] = RNG.choice([0, 1], size=n_d, p=[0.42, 0.58]).astype(float)
        sub["road_access"] = RNG.choice([0, 1], size=n_d, p=[0.35, 0.65]).astype(float)
        sub["terrain_elevation_m"] = RNG.exponential(80, n_d).round(1)
        sub["soil_liquefaction_risk"] = RNG.uniform(0.0, 1.0, n_d).round(3)
        sub["structural_vulnerability_index"] = RNG.uniform(0.1, 1.0, n_d).round(3)
        sub["battery_reserve_hours"] = np.clip(RNG.normal(8, 5, n_d), 0.5, 48.0).round(2)
        sub["historical_outage_count"] = RNG.integers(0, 15, n_d).astype(float)
        sub["emergency_calls_count"] = RNG.integers(0, 600, n_d).astype(float)
        sub["tower_operational_percentage"] = np.clip(RNG.normal(60, 25, n_d), 0, 100).round(1)
        sub["network_congestion_percentage"] = np.clip(RNG.normal(55, 25, n_d), 0, 100).round(1)
        sub["distance_to_nearest_hospital_km"] = np.clip(RNG.exponential(12, n_d), 1, 80).round(2)
        sub["distance_to_nearest_relief_camp_km"] = np.clip(RNG.exponential(10, n_d), 1, 60).round(2)
        sub["historical_disaster_frequency"] = RNG.integers(0, 12, n_d).astype(float)

        rows.append(pd.DataFrame(sub))

    df = pd.concat(rows, ignore_index=True)
    df = compute_targets(df)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Generating 5,000-row Antigravity Aegis dataset...")
    df = generate_dataset(N_TOTAL)
    out_path = os.path.join(parent_dir, "data", "silent_zone_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} records to {out_path}")
    print(f"Silent Zone rate:  {df['silent_zone'].mean()*100:.1f}%")
    print(f"Impact severity distribution:\n{df['impact_severity'].value_counts()}")
    print(f"Disaster types:\n{df['disaster_type'].value_counts()}")
    print(f"Columns: {list(df.columns)}")
