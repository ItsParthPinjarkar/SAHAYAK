"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     ANTIGRAVITY AEGIS — SHELTER RESOURCE DATASET GENERATOR                   ║
║     Engine 3: Synthetic Disaster Shelter Network Data (India Geobounded)    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.config import (
    SHELTER_DATASET_PATH,
    SHELTER_HUMANITARIAN_STANDARDS,
    RANDOM_SEED,
)

# Indian Disaster-Prone Geographic Hubs
INDIAN_SHELTER_HUBS = [
    {"state": "Odisha", "district": "Puri", "base_lat": 19.8135, "base_lon": 85.8312, "hazard": "cyclone"},
    {"state": "Odisha", "district": "Balasore", "base_lat": 21.4934, "base_lon": 86.9135, "hazard": "cyclone"},
    {"state": "Andhra Pradesh", "district": "Visakhapatnam", "base_lat": 17.6868, "base_lon": 83.2185, "hazard": "cyclone"},
    {"state": "West Bengal", "district": "South 24 Parganas", "base_lat": 22.1352, "base_lon": 88.5426, "hazard": "cyclone"},
    {"state": "Assam", "district": "Dhubri", "base_lat": 26.0207, "base_lon": 89.9742, "hazard": "flood"},
    {"state": "Assam", "district": "Kaziranga", "base_lat": 26.5775, "base_lon": 93.1711, "hazard": "flood"},
    {"state": "Bihar", "district": "Darbhanga", "base_lat": 26.1542, "base_lon": 85.8918, "hazard": "flood"},
    {"state": "Bihar", "district": "Katihar", "base_lat": 25.5541, "base_lon": 87.5716, "hazard": "flood"},
    {"state": "Kerala", "district": "Wayanad", "base_lat": 11.6854, "base_lon": 76.1320, "hazard": "flood"},
    {"state": "Kerala", "district": "Idukki", "base_lat": 9.8494, "base_lon": 76.9810, "hazard": "flood"},
    {"state": "Uttarakhand", "district": "Chamoli", "base_lat": 30.4230, "base_lon": 79.3275, "hazard": "earthquake"},
    {"state": "Uttarakhand", "district": "Uttarkashi", "base_lat": 30.7268, "base_lon": 78.4354, "hazard": "earthquake"},
    {"state": "Gujarat", "district": "Kutch", "base_lat": 23.7337, "base_lon": 69.8597, "hazard": "earthquake"},
    {"state": "Himachal Pradesh", "district": "Kangra", "base_lat": 32.0998, "base_lon": 76.2691, "hazard": "earthquake"},
]

def generate_shelter_dataset(n_samples: int = 5000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generates a realistic multi-shelter dataset across India's hazard zones.
    """
    np.random.seed(seed)
    records = []

    for i in range(n_samples):
        hub = INDIAN_SHELTER_HUBS[np.random.choice(len(INDIAN_SHELTER_HUBS))]
        
        # Spatial jitter around hub (within ~50 km radius)
        lat = hub["base_lat"] + np.random.normal(0, 0.25)
        lon = hub["base_lon"] + np.random.normal(0, 0.25)
        
        shelter_id = f"SH-{hub['district'][:3].upper()}-{i+1:04d}"
        shelter_name = f"{hub['district']} Sector-{np.random.randint(1, 20)} Relief Center"

        # Capacity & Occupancy
        capacity = int(np.random.choice([150, 300, 500, 800, 1200, 2000], p=[0.25, 0.35, 0.20, 0.10, 0.07, 0.03]))
        
        # Load profile: some crowded, some normal, some underutilized
        load_factor = np.random.beta(2, 2) * 1.3  # Range roughly 0.1 to 1.3
        occupancy = int(max(20, min(int(capacity * load_factor), int(capacity * 1.4))))
        
        vulnerable_ratio = float(np.clip(np.random.beta(2, 5), 0.05, 0.65))
        
        # Environmental conditions
        days_isolated = float(np.random.exponential(scale=2.5))
        road_access = int(np.random.choice([1, 0], p=[0.75, 0.25] if days_isolated < 3 else [0.4, 0.6]))
        power_backup_hours = float(np.clip(np.random.normal(24, 18), 0, 96))

        # Initial inventory distribution (some overstocked, some starving)
        # Daily requirements
        req_food_day = occupancy * SHELTER_HUMANITARIAN_STANDARDS["food_rations_kg_per_person_day"]
        req_water_day = occupancy * SHELTER_HUMANITARIAN_STANDARDS["water_liters_per_person_day"]
        req_med_total = (occupancy / 100.0) * SHELTER_HUMANITARIAN_STANDARDS["medical_kits_per_100_people"] * (1.0 + vulnerable_ratio * 1.5)
        req_blankets = occupancy * SHELTER_HUMANITARIAN_STANDARDS["blankets_per_person"]

        # Stock factor: 0.2 (starving) to 6.0 (super-rich donor)
        stock_factor = np.random.choice(
            [0.2, 0.6, 1.2, 2.5, 4.5, 7.0],
            p=[0.15, 0.25, 0.25, 0.18, 0.12, 0.05]
        ) * np.random.uniform(0.8, 1.2)

        food_kg = float(max(0.0, req_food_day * stock_factor))
        water_liters = float(max(0.0, req_water_day * (stock_factor * np.random.uniform(0.7, 1.3))))
        medical_kits = float(max(0.0, req_med_total * (stock_factor * np.random.uniform(0.6, 1.4))))
        blankets_count = float(max(0.0, req_blankets * (stock_factor * np.random.uniform(0.5, 1.5))))

        # Calculate buffer days
        food_buffer_days = food_kg / (req_food_day + 1e-4)
        water_buffer_days = water_liters / (req_water_day + 1e-4)
        min_lifeline_buffer = min(food_buffer_days, water_buffer_days)

        # Ground truth target: Shortage Severity Score (0 - 100)
        # Higher means more desperate
        base_shortage = max(0.0, min(100.0, (5.0 - min_lifeline_buffer) * 20.0))
        if road_access == 0:
            base_shortage += 15.0
        base_shortage += vulnerable_ratio * 12.0
        base_shortage = float(np.clip(base_shortage + np.random.normal(0, 3.0), 0.0, 100.0))

        # Target classification
        if min_lifeline_buffer < 1.5 or base_shortage >= 70.0:
            urgency_class = "CRITICAL_DEFICIT"
            action_code = 0
        elif min_lifeline_buffer < 3.0 or base_shortage >= 40.0:
            urgency_class = "DEFICIT"
            action_code = 1
        elif min_lifeline_buffer >= 5.0 and base_shortage < 25.0:
            urgency_class = "SURPLUS"
            action_code = 3
        else:
            urgency_class = "BALANCED"
            action_code = 2

        records.append({
            "shelter_id": shelter_id,
            "shelter_name": shelter_name,
            "district": hub["district"],
            "state": hub["state"],
            "hazard_type": hub["hazard"],
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "capacity_people": capacity,
            "current_occupancy": occupancy,
            "vulnerable_ratio": round(vulnerable_ratio, 3),
            "days_isolated": round(days_isolated, 2),
            "road_access": road_access,
            "power_backup_hours": round(power_backup_hours, 1),
            "food_rations_kg": round(food_kg, 1),
            "water_liters": round(water_liters, 1),
            "medical_kits": round(medical_kits, 1),
            "blankets_count": round(blankets_count, 1),
            "target_shortage_score": round(base_shortage, 2),
            "target_urgency_class": urgency_class,
            "target_action_code": action_code,
        })

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    os.makedirs(os.path.dirname(SHELTER_DATASET_PATH), exist_ok=True)
    df = generate_shelter_dataset(5000)
    df.to_csv(SHELTER_DATASET_PATH, index=False)
    print(f"Generated {len(df)} shelter records saved to {SHELTER_DATASET_PATH}")
