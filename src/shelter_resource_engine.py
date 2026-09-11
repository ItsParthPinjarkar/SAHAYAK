"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     ANTIGRAVITY AEGIS — SHELTER RESOURCE & REDISTRIBUTION ENGINE            ║
║     Engine 3: Autonomous Shelter Intelligence & Multi-Commodity Logistics   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Features:
1. Physics & Humanitarian Standard Feature Engineering (Sphere & NDMA India)
2. Dual ML Estimators:
   - Resource Exhaustion / Shortage Forecaster (GradientBoosting Regressor)
   - Shelter Urgency / Rebalancing Classifier (Random Forest + GradientBoosting)
3. Dynamic Multi-Commodity Network Optimizer:
   - Calculates distance & impedance-weighted transfer routes
   - Considers road blockage, drone/airdrop requirements, and priority tiers
   - Prevents donor depletion while satisfying recipient deficits
"""

import os
import sys
import math
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.config import (
    SHELTER_MODEL_SAVE_PATH,
    SHELTER_HUMANITARIAN_STANDARDS,
    GEO_BOUNDS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# 1. FEATURE TRANSFORMER FOR SHELTER INTELLIGENCE
# ─────────────────────────────────────────────────────────────────

class ShelterResourceFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Computes domain-specific humanitarian supply chain metrics,
    consumption burn rates, buffer horizons, and vulnerability weights.
    """

    def __init__(self):
        self.standards = SHELTER_HUMANITARIAN_STANDARDS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, dict):
            df = pd.DataFrame([X])
        else:
            df = pd.DataFrame(X)

        # Standard daily consumption requirements
        occ = df.get("current_occupancy", pd.Series(100, index=df.index)).clip(lower=1)
        cap = df.get("capacity_people", pd.Series(200, index=df.index)).clip(lower=1)
        vuln = df.get("vulnerable_ratio", pd.Series(0.2, index=df.index)).clip(lower=0.0, upper=1.0)
        
        food_kg = df.get("food_rations_kg", pd.Series(0.0, index=df.index)).clip(lower=0.0)
        water_l = df.get("water_liters", pd.Series(0.0, index=df.index)).clip(lower=0.0)
        med_kits = df.get("medical_kits", pd.Series(0.0, index=df.index)).clip(lower=0.0)
        blankets = df.get("blankets_count", pd.Series(0.0, index=df.index)).clip(lower=0.0)
        
        days_iso = df.get("days_isolated", pd.Series(1.0, index=df.index)).clip(lower=0.0)
        road_acc = df.get("road_access", pd.Series(1, index=df.index)).clip(lower=0, upper=1)
        pwr_hrs = df.get("power_backup_hours", pd.Series(24.0, index=df.index)).clip(lower=0.0)

        # Daily requirement rates
        req_food_day = occ * self.standards["food_rations_kg_per_person_day"]
        req_water_day = occ * self.standards["water_liters_per_person_day"]
        req_med_total = (occ / 100.0) * self.standards["medical_kits_per_100_people"] * (1.0 + vuln * 1.5)
        req_blankets = occ * self.standards["blankets_per_person"]

        # Days of buffer remaining per commodity (avoid div by zero)
        df["food_buffer_days"] = food_kg / (req_food_day + 1e-4)
        df["water_buffer_days"] = water_l / (req_water_day + 1e-4)
        df["med_buffer_ratio"] = med_kits / (req_med_total + 1e-4)
        df["blanket_coverage_ratio"] = blankets / (req_blankets + 1e-4)

        # Compound burn rate and occupancy pressures
        df["occupancy_rate"] = occ / cap
        df["vulnerability_weighted_load"] = occ * (1.0 + vuln * 0.75)
        
        # Min buffer across critical life-support (Food & Water)
        df["min_lifeline_buffer_days"] = df[["food_buffer_days", "water_buffer_days"]].min(axis=1)
        
        # Access & infrastructure fragility
        df["isolation_stress_factor"] = (days_iso + 1.0) * (2.0 - road_acc)
        df["power_exhaustion_risk"] = 1.0 / (pwr_hrs + 1.0)

        return df


# ─────────────────────────────────────────────────────────────────
# 2. DISTANCE UTILITIES (HAVERSINE)
# ─────────────────────────────────────────────────────────────────

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two GPS coordinates in kilometers."""
    R = 6371.0  # Earth's mean radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return float(R * c)


# ─────────────────────────────────────────────────────────────────
# 3. SHELTER RESOURCE ENGINE CLASS
# ─────────────────────────────────────────────────────────────────

class ShelterResourceIntelligence:
    """
    Engine 3: Cognitive Shelter Resource Management & Dynamic Redistribution System.
    """

    def __init__(self, model_bundle_path: str = SHELTER_MODEL_SAVE_PATH):
        self.model_bundle_path = model_bundle_path
        self.bundle = None
        self.is_loaded = False
        self.transformer = ShelterResourceFeatureTransformer()
        self._try_load()

    def _try_load(self):
        if os.path.exists(self.model_bundle_path):
            try:
                self.bundle = joblib.load(self.model_bundle_path)
                self.is_loaded = True
                logger.info(f"Loaded Shelter Resource Model from {self.model_bundle_path}")
            except Exception as e:
                logger.warning(f"Could not load shelter model bundle: {e}")
                self.is_loaded = False
        else:
            logger.info("Shelter Resource Model not found on disk. Operating in heuristic mode until trained.")

    def analyze_shelter(self, shelter_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes a single shelter's inventory, buffer horizon, and urgency status.
        """
        df_raw = pd.DataFrame([shelter_data])
        df_feats = self.transformer.transform(df_raw)
        row = df_feats.iloc[0]

        occ = float(row.get("current_occupancy", 100))
        cap = float(row.get("capacity_people", 200))
        vuln = float(row.get("vulnerable_ratio", 0.2))
        
        food_days = float(row["food_buffer_days"])
        water_days = float(row["water_buffer_days"])
        med_ratio = float(row["med_buffer_ratio"])
        blanket_ratio = float(row["blanket_coverage_ratio"])
        min_buffer = float(row["min_lifeline_buffer_days"])
        road_acc = int(row.get("road_access", 1))
        
        # Calculate shortage score (0 - 100, where 100 is critical immediate collapse)
        if self.is_loaded and "regressor" in self.bundle:
            try:
                feature_cols = self.bundle.get("feature_names", [])
                X_input = df_feats[feature_cols] if feature_cols else df_feats
                shortage_score = float(self.bundle["regressor"].predict(X_input)[0])
                shortage_score = max(0.0, min(100.0, shortage_score))
            except Exception:
                shortage_score = self._heuristic_shortage_score(min_buffer, med_ratio, road_acc, vuln)
        else:
            shortage_score = self._heuristic_shortage_score(min_buffer, med_ratio, road_acc, vuln)

        # Classification / Urgency Tier
        safe_buffer = SHELTER_HUMANITARIAN_STANDARDS["safe_buffer_days"]
        critical_thresh = SHELTER_HUMANITARIAN_STANDARDS["critical_threshold_days"]

        if min_buffer < critical_thresh or shortage_score >= 70.0:
            status = "CRITICAL_DEFICIT"
            status_desc = "Immediate replenishment required. Lifeline supplies under 36 hours."
            action_type = "RECEIVER_HIGH_PRIORITY"
        elif min_buffer < safe_buffer or shortage_score >= 40.0:
            status = "DEFICIT"
            status_desc = "Replenishment needed within 48-72 hours."
            action_type = "RECEIVER_STANDARD"
        elif min_buffer >= (safe_buffer * 1.75) and shortage_score < 25.0:
            status = "SURPLUS"
            status_desc = f"Excess stock identified (> {min_buffer:.1f} days buffer). Available for redistribution."
            action_type = "DONOR"
        else:
            status = "BALANCED"
            status_desc = "Supplies stable. Operational equilibrium."
            action_type = "STABLE"

        # Daily burn rates
        daily_food_kg = occ * SHELTER_HUMANITARIAN_STANDARDS["food_rations_kg_per_person_day"]
        daily_water_l = occ * SHELTER_HUMANITARIAN_STANDARDS["water_liters_per_person_day"]

        # Computable surplus / deficit volumes in absolute units
        food_available = float(row.get("food_rations_kg", 0.0))
        water_available = float(row.get("water_liters", 0.0))
        
        # Reserve required to keep safe_buffer_days
        food_safe_reserve = daily_food_kg * safe_buffer
        water_safe_reserve = daily_water_l * safe_buffer

        food_net = food_available - food_safe_reserve
        water_net = water_available - water_safe_reserve

        return {
            "shelter_id": str(shelter_data.get("shelter_id", "SH-001")),
            "shelter_name": str(shelter_data.get("shelter_name", "Disaster Relief Camp")),
            "district": str(shelter_data.get("district", "Unknown")),
            "latitude": float(shelter_data.get("latitude", 20.0)),
            "longitude": float(shelter_data.get("longitude", 85.0)),
            "current_occupancy": int(occ),
            "capacity_people": int(cap),
            "occupancy_rate_pct": round((occ / cap) * 100.0, 1),
            "status": status,
            "status_description": status_desc,
            "action_type": action_type,
            "shortage_score": round(shortage_score, 1),
            "lifeline_buffer_days": round(min_buffer, 2),
            "food_buffer_days": round(food_days, 2),
            "water_buffer_days": round(water_days, 2),
            "medical_coverage_pct": round(med_ratio * 100.0, 1),
            "blanket_coverage_pct": round(blanket_ratio * 100.0, 1),
            "daily_burn": {
                "food_kg_day": round(daily_food_kg, 1),
                "water_liters_day": round(daily_water_l, 1),
            },
            "net_surplus_deficit": {
                "food_kg": round(food_net, 1),
                "water_liters": round(water_net, 1),
            },
            "road_accessible": bool(road_acc == 1),
        }

    def _heuristic_shortage_score(self, min_buffer: float, med_ratio: float, road_acc: int, vuln: float) -> float:
        """Heuristic risk score calculation (0 - 100)."""
        score = 0.0
        if min_buffer <= 0.5:
            score += 60.0
        elif min_buffer <= 1.5:
            score += 45.0 + (1.5 - min_buffer) * 15.0
        elif min_buffer <= 3.0:
            score += 20.0 + (3.0 - min_buffer) * 16.6
        else:
            score += max(0.0, 15.0 - (min_buffer - 3.0) * 3.0)

        if med_ratio < 0.5:
            score += 20.0 * (1.0 - med_ratio)
        elif med_ratio < 1.0:
            score += 10.0 * (1.0 - med_ratio)

        if road_acc == 0:
            score += 15.0

        score += vuln * 10.0
        return max(0.0, min(100.0, score))

    def optimize_redistribution(self, shelters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Solves the dynamic multi-commodity redistribution problem across all shelters in a network.
        """
        if not shelters:
            return {"transfers": [], "summary": "No shelter data provided."}

        analyzed_shelters = [self.analyze_shelter(s) for s in shelters]

        donors = []
        receivers = []
        stable = []

        for s in analyzed_shelters:
            if s["action_type"] == "DONOR":
                donors.append(s)
            elif s["action_type"].startswith("RECEIVER"):
                receivers.append(s)
            else:
                stable.append(s)

        receivers.sort(key=lambda x: (x["shortage_score"], -x["lifeline_buffer_days"]), reverse=True)

        transfers = []
        transfer_id_counter = 1

        donor_pools = {}
        for d in donors:
            donor_pools[d["shelter_id"]] = {
                "food_kg": max(0.0, d["net_surplus_deficit"]["food_kg"]),
                "water_liters": max(0.0, d["net_surplus_deficit"]["water_liters"]),
                "info": d
            }

        receiver_needs = {}
        for r in receivers:
            receiver_needs[r["shelter_id"]] = {
                "food_kg": abs(min(0.0, r["net_surplus_deficit"]["food_kg"])),
                "water_liters": abs(min(0.0, r["net_surplus_deficit"]["water_liters"])),
                "info": r
            }

        commodities = ["food_kg", "water_liters"]
        unit_map = {"food_kg": "kg (Rations)", "water_liters": "Liters (Water)"}

        for comm in commodities:
            for r_id, r_data in receiver_needs.items():
                needed = r_data[comm]
                if needed <= 5.0:
                    continue

                r_info = r_data["info"]

                candidate_donors = []
                for d_id, d_data in donor_pools.items():
                    available = d_data[comm]
                    if available > 10.0:
                        d_info = d_data["info"]
                        dist_km = haversine_distance_km(
                            d_info["latitude"], d_info["longitude"],
                            r_info["latitude"], r_info["longitude"]
                        )
                        road_factor = 1.0 if (d_info["road_accessible"] and r_info["road_accessible"]) else 2.5
                        impedance = dist_km * road_factor
                        candidate_donors.append((impedance, dist_km, d_id, d_data))

                candidate_donors.sort(key=lambda x: x[0])

                for impedance, dist_km, d_id, d_data in candidate_donors:
                    if needed <= 5.0:
                        break
                    
                    avail = d_data[comm]
                    if avail <= 5.0:
                        continue

                    qty_to_transfer = min(needed, avail)
                    
                    d_data[comm] -= qty_to_transfer
                    needed -= qty_to_transfer
                    r_data[comm] = needed

                    if not r_info["road_accessible"] or not d_data["info"]["road_accessible"]:
                        if qty_to_transfer <= 150.0:
                            transport_mode = "Autonomous Heavy-Lift Cargo Drone"
                            est_speed_kmh = 65.0
                        else:
                            transport_mode = "NDRF All-Terrain Amphibious Rescue Boat / Helo Drop"
                            est_speed_kmh = 45.0
                    else:
                        transport_mode = "Emergency Logistics Supply Truck (Convoy)"
                        est_speed_kmh = 40.0

                    est_transit_hours = max(0.25, round(dist_km / est_speed_kmh, 2))

                    transfers.append({
                        "transfer_id": f"TRF-{transfer_id_counter:04d}",
                        "commodity": "Food Rations" if "food" in comm else "Potable Water",
                        "quantity": round(qty_to_transfer, 1),
                        "unit": unit_map[comm],
                        "source_shelter_id": d_id,
                        "source_name": d_data["info"]["shelter_name"],
                        "source_district": d_data["info"]["district"],
                        "target_shelter_id": r_id,
                        "target_name": r_info["shelter_name"],
                        "target_district": r_info["district"],
                        "distance_km": round(dist_km, 2),
                        "estimated_transit_hours": est_transit_hours,
                        "transport_mode": transport_mode,
                        "priority": "CRITICAL / EMERGENCY" if r_info["status"] == "CRITICAL_DEFICIT" else "HIGH / REBALANCING",
                        "action_justification": (
                            f"Transfer {round(qty_to_transfer, 1)} {unit_map[comm]} from {d_data['info']['shelter_name']} "
                            f"(Surplus buffer: {d_data['info']['lifeline_buffer_days']:.1f}d) to {r_info['shelter_name']} "
                            f"(Critical buffer: {r_info['lifeline_buffer_days']:.1f}d). Distance: {dist_km:.1f} km."
                        )
                    })
                    transfer_id_counter += 1

        total_food_moved = sum(t["quantity"] for t in transfers if "Food" in t["commodity"])
        total_water_moved = sum(t["quantity"] for t in transfers if "Water" in t["commodity"])
        unresolved_receivers = [r["info"]["shelter_name"] for r in receiver_needs.values() if r["food_kg"] > 20 or r["water_liters"] > 50]

        return {
            "total_shelters_monitored": len(analyzed_shelters),
            "surplus_shelters_count": len(donors),
            "deficit_shelters_count": len(receivers),
            "balanced_shelters_count": len(stable),
            "recommended_transfers_count": len(transfers),
            "total_food_redistributed_kg": round(total_food_moved, 1),
            "total_water_redistributed_liters": round(total_water_moved, 1),
            "shelters_status": analyzed_shelters,
            "transfers": transfers,
            "unresolved_critical_deficits": unresolved_receivers,
            "system_state": "OPTIMAL_DISPATCH_READY" if transfers else "BALANCED_NO_ACTION_NEEDED"
        }


# Singleton helper instance
_shelter_engine_instance: Optional[ShelterResourceIntelligence] = None

def get_shelter_engine() -> ShelterResourceIntelligence:
    global _shelter_engine_instance
    if _shelter_engine_instance is None:
        _shelter_engine_instance = ShelterResourceIntelligence()
    return _shelter_engine_instance
