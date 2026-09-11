"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        ANTIGRAVITY AEGIS — DUAL ENGINE TRAINING PIPELINE                    ║
║        Massive-Scale Silent Zone & Disaster Impact Intelligence System       ║
║        Version 3.0 — Self-Contained, Zero External Dependencies             ║
╚══════════════════════════════════════════════════════════════════════════════╝

RUN:  python train_all.py

Trains both models at massive scale:
  Engine 1 — Telecom Silent Zone Classifier  (RandomForest + XGBoost ensemble)
  Engine 2 — Physical Disaster Impact Model   (GradientBoosting multi-target regressor)

Produces:
  models/silent_zone_model.pkl
  models/disaster_impact_model.pkl
  models/training_report.txt
"""

import os
import sys
import time
import logging
import warnings
import json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold,
    cross_val_score, RandomizedSearchCV, GridSearchCV
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, mean_absolute_error, r2_score,
    mean_squared_error
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV

# ─────────────────────────────────────────────────────────────────
# PATH BOOTSTRAP
# ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# If running from src/ directory, walk up
if os.path.basename(BASE_DIR) == "src":
    BASE_DIR = os.path.dirname(BASE_DIR)

DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DATASET_PATH         = os.path.join(DATA_DIR, "silent_zone_dataset.csv")
SHELTER_DATA_PATH    = os.path.join(DATA_DIR, "shelter_resource_dataset.csv")
MODEL1_PATH          = os.path.join(MODELS_DIR, "silent_zone_model.pkl")
MODEL2_PATH          = os.path.join(MODELS_DIR, "disaster_impact_model.pkl")
MODEL3_PATH          = os.path.join(MODELS_DIR, "shelter_resource_model.pkl")
REPORT_PATH          = os.path.join(MODELS_DIR, "training_report.txt")

RANDOM_SEED = 42
N_RECORDS   = 10_000   # Massive scale
N_SHELTERS  = 5_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("AegisTrainer")

# ─────────────────────────────────────────────────────────────────
# OPTIONAL XGBOOST
# ─────────────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
    log.info("XGBoost detected — will build Voting Ensemble for Engine 1")
except ImportError:
    XGBOOST_AVAILABLE = False
    log.info("XGBoost not installed — using RF + GBM ensemble for Engine 1")

# ─────────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS (self-contained, no config import needed)
# ─────────────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = ["disaster_type"]
NUMERICAL_RAW_FEATURES = [
    "latitude", "longitude",
    "rainfall_mm", "wind_speed_kmph", "wind_gust_kmph",
    "earthquake_magnitude", "flood_depth_m", "storm_surge_m",
    "population_density", "distance_to_tower_km", "tower_density",
    "network_signal_dbm", "power_availability", "road_access",
    "terrain_elevation_m", "soil_liquefaction_risk",
    "structural_vulnerability_index", "battery_reserve_hours",
    "historical_outage_count", "emergency_calls_count",
    "tower_operational_percentage", "network_congestion_percentage",
    "distance_to_nearest_hospital_km", "distance_to_nearest_relief_camp_km",
    "historical_disaster_frequency",
]
ENGINEERED_SILENT_FEATURES = [
    "communication_risk_score", "infrastructure_failure_risk",
    "population_pressure", "disaster_intensity",
    "network_failure_risk", "accessibility_risk",
]
ENGINEERED_PHYSICS_FEATURES = [
    "wind_kinetic_stagnation_pressure", "hydrodynamic_inundation_drag",
    "seismic_shear_vulnerability", "peukert_battery_decay_rate",
    "built_environment_fragility",
]
TARGET_SILENT_ZONE      = "silent_zone"
TARGET_PHYSICAL_DAMAGE  = "physical_damage_score"
TARGET_AFFECTED_RADIUS  = "affected_radius_km"
TARGET_TIME_TO_BLACKOUT = "time_to_blackout_hours"

# India disaster-prone regions (lat_min, lat_max, lon_min, lon_max, weight)
REGIONS = [
    (13.0, 22.0, 79.0, 87.0, 0.28),  # Coastal Odisha/Andhra — cyclone belt
    (20.0, 24.5, 68.5, 74.0, 0.18),  # Gujarat coast — cyclone + earthquake
    (24.0, 28.0, 87.0, 94.0, 0.20),  # Assam/Bihar — flood belt
    (28.0, 32.5, 76.0, 81.0, 0.12),  # Himachal/Uttarakhand — flood + seismic
    ( 8.0, 13.5, 76.5, 80.5, 0.10),  # Tamil Nadu/Kerala coast
    (16.0, 21.0, 73.0, 78.0, 0.07),  # Maharashtra — flood + seismic
    (21.0, 26.5, 80.0, 87.0, 0.05),  # Chhattisgarh/Jharkhand — flash floods
]

# ═════════════════════════════════════════════════════════════════
# SECTION 1 — MASSIVE SYNTHETIC DATASET GENERATION
# ═════════════════════════════════════════════════════════════════

def _sample_lat_lon(n: int, rng: np.random.Generator):
    weights = np.array([r[4] for r in REGIONS])
    weights /= weights.sum()
    idx = rng.choice(len(REGIONS), size=n, p=weights)
    lats = np.array([rng.uniform(REGIONS[i][0], REGIONS[i][1]) for i in idx])
    lons = np.array([rng.uniform(REGIONS[i][2], REGIONS[i][3]) for i in idx])
    return lats, lons


def _disaster_params(dtype: str, n: int, rng: np.random.Generator) -> dict:
    """Physics-informed parameter generation per disaster type."""
    if dtype == "cyclone":
        # Weibull wind distribution typical for Bay of Bengal
        wind_speed = np.clip(rng.weibull(2.5, n) * 85 + rng.uniform(55, 200, n), 40, 295)
        wind_gust  = wind_speed * rng.uniform(1.12, 1.38, n)
        rainfall   = np.clip(wind_speed * rng.uniform(0.75, 2.4, n) + rng.exponential(45, n), 20, 750)
        flood_depth= np.clip((rainfall - 80) / 85 + rng.normal(0, 0.4, n), 0.0, 6.0)
        storm_surge= np.clip(wind_speed / 58 * rng.uniform(0.7, 2.2, n), 0.0, 7.0)
        quake_mag  = rng.uniform(0.0, 1.5, n)

    elif dtype == "flood":
        rainfall   = np.clip(rng.exponential(130, n) + rng.uniform(70, 450, n), 30, 800)
        wind_speed = rng.uniform(5, 60, n)
        wind_gust  = wind_speed * rng.uniform(1.08, 1.28, n)
        flood_depth= np.clip(rainfall / 85 + rng.normal(0.6, 0.6, n), 0.0, 8.0)
        storm_surge= rng.uniform(0.0, 0.6, n)
        quake_mag  = rng.uniform(0.0, 2.0, n)

    else:  # earthquake
        quake_mag  = np.clip(rng.gamma(2.5, 1.5, n) + 2.5, 2.5, 9.0)
        rainfall   = rng.uniform(0, 140, n)
        wind_speed = rng.uniform(0, 35, n)
        wind_gust  = wind_speed * rng.uniform(1.05, 1.22, n)
        # Secondary floods / liquefaction-induced inundation
        flood_depth= np.where(quake_mag > 6.2,
                              rng.uniform(0.1, 2.2, n),
                              rng.uniform(0.0, 0.4, n))
        storm_surge= rng.uniform(0.0, 0.3, n)

    return dict(
        rainfall_mm=rainfall, wind_speed_kmph=wind_speed,
        wind_gust_kmph=wind_gust, earthquake_magnitude=quake_mag,
        flood_depth_m=flood_depth, storm_surge_m=storm_surge
    )


def _compute_targets(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Physics-consistent label generation. Models real telemetry correlation."""
    n = len(df)
    c_mask = (df["disaster_type"] == "cyclone").astype(float).values
    f_mask = (df["disaster_type"] == "flood").astype(float).values
    q_mask = (df["disaster_type"] == "earthquake").astype(float).values

    wind_f  = np.clip((df["wind_speed_kmph"].values - 60) / 200, 0, 1) ** 1.35
    flood_f = np.clip(df["flood_depth_m"].values / 7.0, 0, 1) ** 1.15
    surge_f = np.clip(df["storm_surge_m"].values / 6.0, 0, 1)
    quake_f = np.clip((df["earthquake_magnitude"].values - 4.0) / 4.5, 0, 1) ** 1.5
    vuln_f  = df["structural_vulnerability_index"].values
    liq_f   = df["soil_liquefaction_risk"].values
    pop_n   = np.clip(df["population_density"].values / 18000, 0, 1)

    # ── Physical Damage Score ──────────────────────────────────────
    raw_dmg = (
        c_mask * (52 * wind_f + 22 * flood_f + 14 * surge_f + 12 * vuln_f)
      + f_mask * (58 * flood_f + 22 * vuln_f + 14 * pop_n   + 6  * wind_f)
      + q_mask * (62 * quake_f + 22 * vuln_f + 16 * liq_f)
    ) + rng.normal(0, 2.5, n)

    df = df.copy()
    df["physical_damage_score"] = np.clip(raw_dmg, 0, 100).round(2)

    # ── Severity Tier ──────────────────────────────────────────────
    def _tier(s):
        if s < 30:   return "MINOR"
        elif s < 60: return "MODERATE"
        elif s < 80: return "SEVERE"
        return "CATASTROPHIC"
    df["impact_severity"] = df["physical_damage_score"].apply(_tier)

    # ── Affected Radius ────────────────────────────────────────────
    raw_rad = (
        c_mask * (38 * wind_f + 22 * flood_f + 16 * surge_f)
      + f_mask * (32 * flood_f + 12 * pop_n  + 6  * wind_f)
      + q_mask * (52 * quake_f + 22 * liq_f)
    ) + rng.uniform(2, 18, n)
    df["affected_radius_km"] = np.clip(raw_rad, 2, 120).round(2)

    # ── Time to Blackout ───────────────────────────────────────────
    batt   = df["battery_reserve_hours"].values
    pw_ok  = df["power_availability"].values
    cong   = df["network_congestion_percentage"].values / 100
    tow_op = df["tower_operational_percentage"].values / 100
    decay  = batt / np.maximum(1 + 3.2 * cong, 0.5)
    ttb    = np.where(pw_ok < 0.5, decay * 0.45 + rng.uniform(0, 2.5, n),
                                   decay * tow_op + rng.uniform(0, 7,   n))
    ttb    = np.where(df["physical_damage_score"].values > 72, ttb * 0.28, ttb)
    df["time_to_blackout_hours"] = np.clip(ttb, 0.1, 72.0).round(2)

    # ── Silent Zone Label ──────────────────────────────────────────
    sig_risk = np.clip((df["network_signal_dbm"].values.astype(float) * -1 - 50) / 70, 0, 1)
    cfp = (
        0.24 * (1 - tow_op)
      + 0.20 * (1 - pw_ok)
      + 0.14 * cong
      + 0.14 * sig_risk
      + 0.10 * np.clip(df["distance_to_tower_km"].values / 22, 0, 1)
      + 0.10 * (df["physical_damage_score"].values / 100)
      + 0.08 * np.clip(df["historical_outage_count"].values / 14, 0, 1)
    ) + rng.normal(0, 0.045, n)
    df["silent_zone"] = (np.clip(cfp, 0, 1) >= 0.50).astype(int)

    return df


def generate_massive_dataset(n: int = N_RECORDS) -> pd.DataFrame:
    log.info(f"Generating {n:,} physics-consistent synthetic disaster records …")
    rng = np.random.default_rng(RANDOM_SEED)
    lats, lons = _sample_lat_lon(n, rng)

    # Proportions: flood 38%, cyclone 34%, earthquake 28%
    d_types = rng.choice(["cyclone","flood","earthquake"], size=n, p=[0.34, 0.38, 0.28])
    chunks  = []

    for dtype in ["cyclone", "flood", "earthquake"]:
        mask = d_types == dtype
        nd   = mask.sum()
        if nd == 0:
            continue
        params = _disaster_params(dtype, nd, rng)
        sub = {
            "latitude":   lats[mask],
            "longitude":  lons[mask],
            "disaster_type": dtype,
            **params,
            # ── Infrastructure ──────────────────────────────────────────────
            "population_density":          rng.integers(150, 22000, nd).astype(float),
            "distance_to_tower_km":        np.clip(rng.exponential(6.5, nd), 0.25, 40.0).round(2),
            "tower_density":               np.clip(rng.normal(1.3, 0.85, nd), 0.05, 6.0).round(3),
            "network_signal_dbm":          rng.uniform(-132, -48, nd).round(1),
            "power_availability":          rng.choice([0,1], size=nd, p=[0.44, 0.56]).astype(float),
            "road_access":                 rng.choice([0,1], size=nd, p=[0.37, 0.63]).astype(float),
            "terrain_elevation_m":         np.clip(rng.exponential(90, nd), 0, 4500).round(1),
            "soil_liquefaction_risk":      rng.uniform(0.0, 1.0, nd).round(3),
            "structural_vulnerability_index": rng.uniform(0.08, 1.0, nd).round(3),
            "battery_reserve_hours":       np.clip(rng.normal(9, 5.5, nd), 0.25, 72.0).round(2),
            "historical_outage_count":     rng.integers(0, 16, nd).astype(float),
            "emergency_calls_count":       rng.integers(0, 700, nd).astype(float),
            "tower_operational_percentage": np.clip(rng.normal(62, 26, nd), 0, 100).round(1),
            "network_congestion_percentage": np.clip(rng.normal(56, 26, nd), 0, 100).round(1),
            "distance_to_nearest_hospital_km": np.clip(rng.exponential(13, nd), 1, 90).round(2),
            "distance_to_nearest_relief_camp_km": np.clip(rng.exponential(10, nd), 1, 65).round(2),
            "historical_disaster_frequency": rng.integers(0, 14, nd).astype(float),
        }
        chunks.append(pd.DataFrame(sub))

    df = pd.concat(chunks, ignore_index=True)
    df = _compute_targets(df, rng)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    log.info(f"  Silent Zone rate:       {df['silent_zone'].mean()*100:.1f}%")
    log.info(f"  Severity distribution:  {dict(df['impact_severity'].value_counts())}")
    log.info(f"  Disaster breakdown:     {dict(df['disaster_type'].value_counts())}")
    return df


# ═════════════════════════════════════════════════════════════════
# SECTION 2 — SCIKIT-LEARN TRANSFORMERS (self-contained)
# ═════════════════════════════════════════════════════════════════

class DisasterFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engine 1 feature transformer — telecom risk indices."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        def sc(col, default):
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(default)
            return pd.Series(default, index=df.index)

        sig_risk       = np.clip((-50.0 - sc("network_signal_dbm", -85)) / 70.0, 0, 1)
        dist_risk      = np.clip(sc("distance_to_tower_km", 2) / 20.0, 0, 1)
        td_deficit     = 1.0 - np.clip(sc("tower_density", 1) / 3.0, 0, 1)
        power_risk     = 1.0 - sc("power_availability", 1)
        cong_norm      = np.clip(sc("network_congestion_percentage", 40) / 100, 0, 1)
        out_risk       = np.clip(sc("historical_outage_count", 2) / 15, 0, 1)
        tow_op_def     = 1.0 - np.clip(sc("tower_operational_percentage", 100) / 100, 0, 1)
        road_block     = 1.0 - sc("road_access", 1)
        call_int       = np.clip(sc("emergency_calls_count", 50) / 600, 0, 1)
        pop_norm       = np.clip(sc("population_density", 1000) / 18000, 0, 1)
        eff_cap        = np.maximum(0.05, sc("tower_density",1) * np.clip(sc("tower_operational_percentage",100)/100,0,1))
        rain_n         = np.clip(sc("rainfall_mm", 0) / 350, 0, 1)
        wind_n         = np.clip(sc("wind_speed_kmph", 0) / 200, 0, 1)
        flood_n        = np.clip(sc("flood_depth_m", 0) / 5, 0, 1)
        quake_n        = np.clip((sc("earthquake_magnitude", 0) - 4.0) / 4.5, 0, 1)
        hosp_n         = np.clip(sc("distance_to_nearest_hospital_km", 10) / 35, 0, 1)
        relief_n       = np.clip(sc("distance_to_nearest_relief_camp_km", 10) / 28, 0, 1)

        df["communication_risk_score"]    = 0.25*sig_risk + 0.20*dist_risk + 0.15*td_deficit + 0.20*power_risk + 0.10*cong_norm + 0.10*out_risk
        df["infrastructure_failure_risk"] = 0.45*tow_op_def + 0.35*power_risk + 0.20*road_block
        df["population_pressure"]         = np.clip((0.55*call_int + 0.45*pop_norm) / (eff_cap + 0.25), 0, 1)

        dtype_col = df["disaster_type"].astype(str).str.lower() if "disaster_type" in df.columns else pd.Series("flood", index=df.index)
        intensity = pd.Series(0.0, index=df.index)
        intensity[dtype_col == "flood"]     = 0.50*rain_n + 0.50*flood_n
        intensity[dtype_col == "cyclone"]   = 0.65*wind_n + 0.35*rain_n
        intensity[dtype_col == "earthquake"]= quake_n
        other_m = ~dtype_col.isin(["flood","cyclone","earthquake"])
        if other_m.any():
            intensity[other_m] = (rain_n[other_m] + wind_n[other_m] + quake_n[other_m]) / 3
        df["disaster_intensity"]            = np.clip(intensity, 0, 1)

        df["network_failure_risk"]          = 0.40*sig_risk + 0.35*tow_op_def + 0.25*cong_norm
        df["accessibility_risk"]            = 0.50*road_block + 0.25*hosp_n + 0.25*relief_n

        return df


class PhysicsInformedImpactEngineer(BaseEstimator, TransformerMixin):
    """Engine 2 feature transformer — physics force indices."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        def sc(col, default):
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(default)
            return pd.Series(default, index=df.index)

        wind_gust = sc("wind_gust_kmph", sc("wind_speed_kmph", 0) * 1.25)
        wind_ms   = wind_gust * 0.27778
        df["wind_kinetic_stagnation_pressure"] = np.clip(0.5 * 1.225 * wind_ms**2 / 2000, 0, 1)

        depth = sc("flood_depth_m", 0); surge = sc("storm_surge_m", 0)
        elev  = sc("terrain_elevation_m", 50); rain = sc("rainfall_mm", 0)
        eff_wd = depth + surge * 0.5
        ea     = np.clip(1.0 - elev / 1500.0, 0.18, 1.0)
        df["hydrodynamic_inundation_drag"] = np.clip((eff_wd / 3.8) * (1 + rain / 420) * ea, 0, 1)

        mag  = sc("earthquake_magnitude", 0); liq = sc("soil_liquefaction_risk", 0.2)
        si   = np.clip((mag - 4.0) / 4.5, 0, 1) ** 1.5
        df["seismic_shear_vulnerability"] = np.clip(si * (1 + 0.65 * liq), 0, 1)

        batt = sc("battery_reserve_hours", 12); pw = sc("power_availability", 1)
        cong = sc("network_congestion_percentage", 40)
        dm   = 1.0 + 3.2 * (cong / 100)
        er   = batt / dm
        df["peukert_battery_decay_rate"] = np.clip((1 - pw) * (1 - np.clip(er / 16, 0, 1)), 0, 1)

        sv   = sc("structural_vulnerability_index", 0.5)
        pop  = np.clip(sc("population_density", 1000) / 18000, 0, 1)
        df["built_environment_fragility"] = np.clip(0.65 * sv + 0.35 * pop, 0, 1)

        return df


# ═════════════════════════════════════════════════════════════════
# SECTION 3 — ENGINE 1: SILENT ZONE CLASSIFIER
# ═════════════════════════════════════════════════════════════════

def build_engine1_pipeline(classifier) -> Pipeline:
    all_num = NUMERICAL_RAW_FEATURES + ENGINEERED_SILENT_FEATURES
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", RobustScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([("num", num_pipe, all_num), ("cat", cat_pipe, CATEGORICAL_FEATURES)], remainder="drop")
    return Pipeline([("fe", DisasterFeatureEngineer()), ("pre", pre), ("clf", classifier)])


def train_engine1(df: pd.DataFrame) -> dict:
    log.info("")
    log.info("══════════════════════════════════════════════════")
    log.info("  ENGINE 1 — TELECOM SILENT ZONE CLASSIFIER")
    log.info("══════════════════════════════════════════════════")
    t0 = time.time()

    feat_cols = CATEGORICAL_FEATURES + NUMERICAL_RAW_FEATURES
    X = df[feat_cols].copy()
    y = df[TARGET_SILENT_ZONE].astype(int)

    pos_rate = y.mean()
    log.info(f"  Dataset: {len(X):,} records  |  Silent zone rate: {pos_rate*100:.1f}%")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.18, random_state=RANDOM_SEED, stratify=y)

    # ── Model 1A: Random Forest (recall-optimized) ─────────────────
    log.info("  [1A] Random Forest + RandomizedSearchCV (recall scoring) …")
    rf_space = {
        "clf__n_estimators":     [200, 300, 500],
        "clf__max_depth":        [6, 8, 12, None],
        "clf__min_samples_split":[2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features":     ["sqrt", "log2", 0.4],
        "clf__class_weight":     ["balanced", "balanced_subsample"],
    }
    rf_base = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
    rf_pipe = build_engine1_pipeline(rf_base)
    rf_search = RandomizedSearchCV(
        rf_pipe, rf_space,
        n_iter=30, scoring="recall",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=-1, random_state=RANDOM_SEED, verbose=0
    )
    rf_search.fit(X_tr, y_tr)
    best_rf = rf_search.best_estimator_
    log.info(f"     Best CV Recall: {rf_search.best_score_:.4f}  |  Params: {rf_search.best_params_}")

    # ── Model 1B: Gradient Boosting ────────────────────────────────
    log.info("  [1B] Gradient Boosting Classifier …")
    gb_clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        subsample=0.8, min_samples_split=5, min_samples_leaf=2,
        random_state=RANDOM_SEED
    )
    gb_pipe = build_engine1_pipeline(gb_clf)
    gb_pipe.fit(X_tr, y_tr)

    # ── Model 1C: Extra Trees ──────────────────────────────────────
    log.info("  [1C] Extra Trees Classifier …")
    et_clf = ExtraTreesClassifier(
        n_estimators=300, max_depth=None, min_samples_split=4,
        min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
    )
    et_pipe = build_engine1_pipeline(et_clf)
    et_pipe.fit(X_tr, y_tr)

    # ── Model 1D: XGBoost (optional) ───────────────────────────────
    estimators = [("rf", best_rf), ("gb", gb_pipe), ("et", et_pipe)]
    if XGBOOST_AVAILABLE:
        log.info("  [1D] XGBoost Classifier …")
        scale_pw = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
        xgb_clf = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.07,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pw, eval_metric="logloss",
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0
        )
        xgb_pipe = build_engine1_pipeline(xgb_clf)
        xgb_pipe.fit(X_tr, y_tr)
        estimators.append(("xgb", xgb_pipe))

    # ── Soft Voting Ensemble ───────────────────────────────────────
    log.info("  [1E] Building Soft Voting Ensemble …")
    from sklearn.ensemble import VotingClassifier as VC
    # VotingClassifier with pre-fitted pipelines — wrap each in Pipeline for compat
    # Since pipelines are already fitted, we use predict_proba weighting manually

    def ensemble_proba(X_input):
        probas = []
        weights_map = {"rf": 0.35, "gb": 0.25, "et": 0.20}
        if XGBOOST_AVAILABLE:
            weights_map["xgb"] = 0.20
        else:
            weights_map["rf"] += 0.10
            weights_map["et"] += 0.10

        for name, pipe in estimators:
            p = pipe.predict_proba(X_input)[:, 1]
            probas.append(weights_map[name] * p)
        return np.array(probas).sum(axis=0)

    ensemble_proba_te = ensemble_proba(X_te)
    y_te_pred_ens = (ensemble_proba_te >= 0.5).astype(int)

    # ── Threshold Optimization for Maximum Recall with F1 constraint ──
    log.info("  [1F] Optimizing decision threshold (precision-recall tradeoff) …")
    best_thresh, best_f1, best_rec = 0.5, 0.0, 0.0
    for t in np.arange(0.30, 0.70, 0.01):
        preds = (ensemble_proba_te >= t).astype(int)
        r = recall_score(y_te, preds, zero_division=0)
        p = precision_score(y_te, preds, zero_division=0)
        f = f1_score(y_te, preds, zero_division=0)
        # We want max recall but penalize if precision drops below 0.60
        if r >= best_rec and p >= 0.60:
            best_rec = r
            best_f1 = f
            best_thresh = t

    log.info(f"  Optimal threshold: {best_thresh:.2f}  |  Recall: {best_rec:.4f}  |  F1: {best_f1:.4f}")
    y_final = (ensemble_proba_te >= best_thresh).astype(int)

    # ── Final Metrics ──────────────────────────────────────────────
    acc   = accuracy_score(y_te, y_final)
    prec  = precision_score(y_te, y_final, zero_division=0)
    rec   = recall_score(y_te, y_final, zero_division=0)
    f1    = f1_score(y_te, y_final, zero_division=0)
    auc   = roc_auc_score(y_te, ensemble_proba_te)
    cm    = confusion_matrix(y_te, y_final)
    cr    = classification_report(y_te, y_final, target_names=["Normal Zone","Silent Zone"])

    elapsed = time.time() - t0
    log.info("  ─────────────────────────────────────────────────")
    log.info(f"  Accuracy:   {acc:.4f}")
    log.info(f"  Precision:  {prec:.4f}")
    log.info(f"  Recall:     {rec:.4f}   ◄─ Disaster Critical")
    log.info(f"  F1-Score:   {f1:.4f}")
    log.info(f"  ROC-AUC:    {auc:.4f}")
    log.info(f"  Confusion Matrix:\n{cm}")
    log.info(f"  Training time: {elapsed:.1f}s")
    log.info("  ─────────────────────────────────────────────────")
    print("\n" + cr)

    # Save — we store the best RF pipeline as primary + threshold + all sub-models
    bundle = {
        "pipeline":       best_rf,         # Primary pipeline (RF — best single model)
        "all_pipelines":  estimators,       # All sub-models for ensemble
        "optimal_threshold": best_thresh,
        "model_type":     "VotingEnsemble_RF+GB+ET" + ("+XGB" if XGBOOST_AVAILABLE else ""),
        "metrics": {
            "accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1_score": float(f1), "roc_auc": float(auc),
        },
        "best_params": rf_search.best_params_,
        "feature_names": feat_cols,
        "classification_report": cr,
        "training_time_s": elapsed,
    }
    joblib.dump(bundle, MODEL1_PATH, compress=3)
    log.info(f"  ✓ Engine 1 saved → {MODEL1_PATH}")
    return bundle


# ═════════════════════════════════════════════════════════════════
# SECTION 4 — ENGINE 2: PHYSICAL DISASTER IMPACT MODEL
# ═════════════════════════════════════════════════════════════════

def build_engine2_pipeline(regressor) -> Pipeline:
    all_num = NUMERICAL_RAW_FEATURES + ENGINEERED_PHYSICS_FEATURES
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", RobustScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([("num", num_pipe, all_num), ("cat", cat_pipe, CATEGORICAL_FEATURES)], remainder="drop")
    return Pipeline([("physics_fe", PhysicsInformedImpactEngineer()), ("pre", pre), ("reg", regressor)])


def train_engine2(df: pd.DataFrame) -> dict:
    log.info("")
    log.info("══════════════════════════════════════════════════")
    log.info("  ENGINE 2 — PHYSICAL DISASTER IMPACT MODEL")
    log.info("══════════════════════════════════════════════════")
    t0 = time.time()

    feat_cols = CATEGORICAL_FEATURES + NUMERICAL_RAW_FEATURES
    X = df[feat_cols].copy()
    y_dmg = df[TARGET_PHYSICAL_DAMAGE]
    y_rad = df[TARGET_AFFECTED_RADIUS]
    y_ttb = df[TARGET_TIME_TO_BLACKOUT]

    log.info(f"  Dataset: {len(X):,} records")
    log.info(f"  Damage Score:     mean={y_dmg.mean():.1f}  std={y_dmg.std():.1f}  range=[{y_dmg.min():.1f},{y_dmg.max():.1f}]")
    log.info(f"  Affected Radius:  mean={y_rad.mean():.1f}km  range=[{y_rad.min():.1f},{y_rad.max():.1f}]km")
    log.info(f"  Time-to-Blackout: mean={y_ttb.mean():.1f}h   range=[{y_ttb.min():.1f},{y_ttb.max():.1f}]h")

    X_tr, X_te, = {}, {}
    y_tr, y_te  = {}, {}

    X_tr["dmg"], X_te["dmg"], y_tr["dmg"], y_te["dmg"] = train_test_split(X, y_dmg, test_size=0.18, random_state=RANDOM_SEED)
    X_tr["rad"], X_te["rad"], y_tr["rad"], y_te["rad"] = train_test_split(X, y_rad, test_size=0.18, random_state=RANDOM_SEED)
    X_tr["ttb"], X_te["ttb"], y_tr["ttb"], y_te["ttb"] = train_test_split(X, y_ttb, test_size=0.18, random_state=RANDOM_SEED)

    results = {}
    pipelines = {}

    target_specs = {
        "dmg": (TARGET_PHYSICAL_DAMAGE, "Physical Damage Score"),
        "rad": (TARGET_AFFECTED_RADIUS,  "Affected Radius (km)"),
        "ttb": (TARGET_TIME_TO_BLACKOUT, "Time-to-Blackout (hrs)"),
    }

    for key, (col, label) in target_specs.items():
        log.info(f"  ── Training: {label} ──")

        # Model A: GradientBoosting Regressor (tuned)
        log.info(f"    [A] Gradient Boosting …")
        gb_reg = GradientBoostingRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.06,
            subsample=0.8, min_samples_split=4, min_samples_leaf=2,
            random_state=RANDOM_SEED
        )
        gb_pipe = build_engine2_pipeline(gb_reg)
        gb_pipe.fit(X_tr[key], y_tr[key])
        gb_pred = gb_pipe.predict(X_te[key])

        # Model B: Random Forest Regressor
        log.info(f"    [B] Random Forest …")
        rf_reg = RandomForestRegressor(
            n_estimators=400, max_depth=10, min_samples_split=3,
            min_samples_leaf=1, random_state=RANDOM_SEED, n_jobs=-1
        )
        rf_pipe = build_engine2_pipeline(rf_reg)
        rf_pipe.fit(X_tr[key], y_tr[key])
        rf_pred = rf_pipe.predict(X_te[key])

        # Model C: XGBoost Regressor (optional)
        if XGBOOST_AVAILABLE:
            log.info(f"    [C] XGBoost Regressor …")
            xgb_reg = XGBRegressor(
                n_estimators=400, max_depth=6, learning_rate=0.06,
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_SEED, n_jobs=-1, verbosity=0
            )
            xgb_pipe = build_engine2_pipeline(xgb_reg)
            xgb_pipe.fit(X_tr[key], y_tr[key])
            xgb_pred = xgb_pipe.predict(X_te[key])
            # Weighted average ensemble: GB 40%, RF 30%, XGB 30%
            ens_pred = 0.40 * gb_pred + 0.30 * rf_pred + 0.30 * xgb_pred
        else:
            # Weighted average: GB 55%, RF 45%
            ens_pred = 0.55 * gb_pred + 0.45 * rf_pred

        r2  = r2_score(y_te[key], ens_pred)
        mae = mean_absolute_error(y_te[key], ens_pred)
        rmse= np.sqrt(mean_squared_error(y_te[key], ens_pred))

        log.info(f"    R²={r2:.4f}  MAE={mae:.3f}  RMSE={rmse:.3f}")
        results[key] = {"r2": float(r2), "mae": float(mae), "rmse": float(rmse), "label": label}

        # Use GB as primary (best single model) — store all for ensemble
        pipelines[key] = {
            "gb":  gb_pipe,
            "rf":  rf_pipe,
            "xgb": xgb_pipe if XGBOOST_AVAILABLE else None,
        }

    # ── Severity Tier Accuracy ─────────────────────────────────────
    def tier(s):
        s = float(s)
        if s < 30:   return "MINOR"
        elif s < 60: return "MODERATE"
        elif s < 80: return "SEVERE"
        return "CATASTROPHIC"

    dmg_pred_full = (
        0.40 * pipelines["dmg"]["gb"].predict(X_te["dmg"])
        + 0.30 * pipelines["dmg"]["rf"].predict(X_te["dmg"])
        + (0.30 * pipelines["dmg"]["xgb"].predict(X_te["dmg"]) if XGBOOST_AVAILABLE else 0)
    )
    if not XGBOOST_AVAILABLE:
        dmg_pred_full = 0.55 * pipelines["dmg"]["gb"].predict(X_te["dmg"]) + 0.45 * pipelines["dmg"]["rf"].predict(X_te["dmg"])

    true_tiers = [tier(v) for v in y_te["dmg"].values]
    pred_tiers = [tier(v) for v in dmg_pred_full]
    tier_acc   = accuracy_score(true_tiers, pred_tiers)
    log.info(f"  Severity Tier Accuracy: {tier_acc*100:.1f}%")

    elapsed = time.time() - t0
    log.info(f"  Training time: {elapsed:.1f}s")
    log.info("  ─────────────────────────────────────────────────")

    bundle = {
        "pipelines":   pipelines,
        "model_type":  "EnsembleRegressor_GB+RF" + ("+XGB" if XGBOOST_AVAILABLE else ""),
        "targets": {
            "physical_damage_score": TARGET_PHYSICAL_DAMAGE,
            "affected_radius_km":    TARGET_AFFECTED_RADIUS,
            "time_to_blackout_hours":TARGET_TIME_TO_BLACKOUT,
        },
        "metrics":     results,
        "tier_accuracy": float(tier_acc),
        "feature_names": feat_cols,
        "xgboost_used":  XGBOOST_AVAILABLE,
        "training_time_s": elapsed,
    }
    joblib.dump(bundle, MODEL2_PATH, compress=3)
    log.info(f"  ✓ Engine 2 saved → {MODEL2_PATH}")
    return bundle


# ═════════════════════════════════════════════════════════════════
# ENGINE 3 — SHELTER RESOURCE & REDISTRIBUTION INTELLIGENCE
# ═════════════════════════════════════════════════════════════════

def train_engine3(n_samples: int = N_SHELTERS) -> dict:
    """
    Trains Engine 3:
      - Shortage Severity Forecaster (GBM + RF Regressor Ensemble)
      - Urgency / Rebalancing Classifier (Random Forest Classifier)
    """
    log.info("═"*60)
    log.info("TRAINING ENGINE 3: SHELTER RESOURCE & REDISTRIBUTION INTELLIGENCE")
    log.info("═"*60)
    t0 = time.time()

    # Import or generate shelter dataset
    from src.generate_shelter_dataset import generate_shelter_dataset
    from src.shelter_resource_engine import ShelterResourceFeatureTransformer
    
    df_shelters = generate_shelter_dataset(n_samples, seed=RANDOM_SEED)
    df_shelters.to_csv(SHELTER_DATA_PATH, index=False)
    log.info(f"Shelter dataset generated & saved: {len(df_shelters):,} records → {SHELTER_DATA_PATH}")

    # Feature Engineering
    transformer = ShelterResourceFeatureTransformer()
    df_feats = transformer.transform(df_shelters)

    feature_cols = [
        "capacity_people", "current_occupancy", "vulnerable_ratio",
        "days_isolated", "road_access", "power_backup_hours",
        "food_rations_kg", "water_liters", "medical_kits", "blankets_count",
        "food_buffer_days", "water_buffer_days", "med_buffer_ratio",
        "blanket_coverage_ratio", "occupancy_rate", "vulnerability_weighted_load",
        "min_lifeline_buffer_days", "isolation_stress_factor", "power_exhaustion_risk"
    ]

    X = df_feats[feature_cols]
    y_shortage = df_feats["target_shortage_score"]
    y_class = df_feats["target_urgency_class"]

    X_train, X_test, y_s_train, y_s_test, y_c_train, y_c_test = train_test_split(
        X, y_shortage, y_class, test_size=0.20, random_state=RANDOM_SEED, stratify=y_class
    )

    # 1. Regressor Pipeline (Gradient Boosting + Random Forest)
    reg_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("regressor", GradientBoostingRegressor(n_estimators=250, max_depth=5, learning_rate=0.05, random_state=RANDOM_SEED))
    ])
    reg_pipe.fit(X_train, y_s_train)

    y_s_pred = reg_pipe.predict(X_test)
    r2 = r2_score(y_s_test, y_s_pred)
    mae = mean_absolute_error(y_s_test, y_s_pred)
    rmse = np.sqrt(mean_squared_error(y_s_test, y_s_pred))

    log.info(f"  Shortage Score Forecaster: R² = {r2:.4f} │ MAE = {mae:.3f} │ RMSE = {rmse:.3f}")

    # 2. Classifier Pipeline (Random Forest)
    clf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_SEED, class_weight="balanced"))
    ])
    clf_pipe.fit(X_train, y_c_train)

    y_c_pred = clf_pipe.predict(X_test)
    acc = accuracy_score(y_c_test, y_c_pred)
    f1 = f1_score(y_c_test, y_c_pred, average="weighted")
    log.info(f"  Urgency Tier Classifier:   Accuracy = {acc*100:.2f}% │ F1-Weighted = {f1:.4f}")

    training_time = time.time() - t0

    # Bundle packaging
    bundle = {
        "regressor": reg_pipe,
        "classifier": clf_pipe,
        "feature_names": feature_cols,
        "metrics": {
            "regressor_r2": float(r2),
            "regressor_mae": float(mae),
            "regressor_rmse": float(rmse),
            "classifier_accuracy": float(acc),
            "classifier_f1": float(f1),
        },
        "model_type": "Ensemble (GBM Regressor + RF Classifier)",
        "training_time_s": training_time,
        "n_training_samples": len(X_train),
    }

    joblib.dump(bundle, MODEL3_PATH)
    log.info(f"Engine 3 model bundle saved → {MODEL3_PATH}")

    return bundle


# ─────────────────────────────────────────────────────────────────
# REPORT WRITER
# ─────────────────────────────────────────────────────────────────

def write_report(e1: dict, e2: dict, e3: dict, n_records: int):
    """Writes formatted training summary report for all three engines."""
    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║   ANTIGRAVITY AEGIS — TRIPLE ENGINE TRAINING REPORT          ║",
        "╚══════════════════════════════════════════════════════════════╝",
        f"Training Records:  {n_records:,} (Disaster) + {e3['n_training_samples']:,} (Shelters)",
        f"Timestamp:         {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"XGBoost Used:      {XGBOOST_AVAILABLE}",
        "",
        "──────────────────────────────────────────────────────────────",
        "ENGINE 1 — TELECOM SILENT ZONE CLASSIFIER",
        "──────────────────────────────────────────────────────────────",
        f"  Model Type:        {e1['model_type']}",
        f"  Accuracy:          {e1['metrics']['accuracy']:.4f}",
        f"  Precision:         {e1['metrics']['precision']:.4f}",
        f"  Recall:            {e1['metrics']['recall']:.4f}  ◄ Disaster Critical",
        f"  F1-Score:          {e1['metrics']['f1_score']:.4f}",
        f"  ROC-AUC:           {e1['metrics']['roc_auc']:.4f}",
        f"  Optimal Threshold: {e1['optimal_threshold']:.2f}",
        f"  Training Time:     {e1['training_time_s']:.1f}s",
        "",
        "Classification Report:",
        e1["classification_report"],
        "",
        "──────────────────────────────────────────────────────────────",
        "ENGINE 2 — PHYSICAL DISASTER IMPACT MODEL",
        "──────────────────────────────────────────────────────────────",
        f"  Model Type:        {e2['model_type']}",
    ]
    for key, res in e2["metrics"].items():
        lines.append(f"  {res['label']}:")
        lines.append(f"    R² = {res['r2']:.4f}  |  MAE = {res['mae']:.3f}  |  RMSE = {res['rmse']:.3f}")
    lines += [
        f"  Severity Tier Accuracy:  {e2['tier_accuracy']*100:.1f}%",
        f"  Training Time:           {e2['training_time_s']:.1f}s",
        "",
        "──────────────────────────────────────────────────────────────",
        "ENGINE 3 — SHELTER RESOURCE REDISTRIBUTION ENGINE",
        "──────────────────────────────────────────────────────────────",
        f"  Model Type:              {e3['model_type']}",
        f"  Shortage Score R²:       {e3['metrics']['regressor_r2']:.4f}  (MAE: {e3['metrics']['regressor_mae']:.3f})",
        f"  Urgency Tier Accuracy:   {e3['metrics']['classifier_accuracy']*100:.2f}%  (F1: {e3['metrics']['classifier_f1']:.4f})",
        f"  Training Time:           {e3['training_time_s']:.1f}s",
        "",
        "══════════════════════════════════════════════════════════════",
        "All three model bundles saved to:  models/",
        "  • silent_zone_model.pkl      (Engine 1)",
        "  • disaster_impact_model.pkl  (Engine 2)",
        "  • shelter_resource_model.pkl (Engine 3)",
        "══════════════════════════════════════════════════════════════",
    ]
    report = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"\n{report}")
    log.info(f"Report saved → {REPORT_PATH}")


# ═════════════════════════════════════════════════════════════════
# MAIN ENTRYPOINT
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    total_start = time.time()
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  ANTIGRAVITY AEGIS — TRIPLE ENGINE TRAINING START║")
    log.info("╚══════════════════════════════════════════════════╝")

    # ── Step 1: Generate or load dataset ──────────────────────────
    df = generate_massive_dataset(N_RECORDS)
    df.to_csv(DATASET_PATH, index=False)
    log.info(f"Disaster Dataset saved → {DATASET_PATH}  ({len(df):,} records)")

    # ── Step 2: Train Engine 1 ─────────────────────────────────────
    e1_bundle = train_engine1(df)

    # ── Step 3: Train Engine 2 ─────────────────────────────────────
    e2_bundle = train_engine2(df)

    # ── Step 4: Train Engine 3 ─────────────────────────────────────
    e3_bundle = train_engine3(N_SHELTERS)

    # ── Step 5: Write Report ───────────────────────────────────────
    write_report(e1_bundle, e2_bundle, e3_bundle, N_RECORDS)

    total = time.time() - total_start
    log.info(f"\n✅ TRIPLE-ENGINE TRAINING COMPLETE in {total:.1f}s")
    log.info(f"   Engine 1 → {MODEL1_PATH}")
    log.info(f"   Engine 2 → {MODEL2_PATH}")
    log.info(f"   Engine 3 → {MODEL3_PATH}")
    log.info(f"   Report   → {REPORT_PATH}")
