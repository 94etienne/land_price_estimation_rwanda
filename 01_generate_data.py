import numpy as np
import pandas as pd

np.random.seed(42)

# ----------------------------------------------
# CONFIG
# ----------------------------------------------
N_PARCELS = 5000
TRAIN_FRACTION = 0.8

districts = ["Gasabo", "Kicukiro", "Nyarugenge", "Musanze", "Huye", "Rubavu"]
sectors = ["SectorA", "SectorB", "SectorC", "SectorD"]  # <-- ADDED SECTOR LIST

district_base = {
    "Gasabo": 69000,
    "Kicukiro": 67000,
    "Nyarugenge": 82000,
    "Musanze": 42000,
    "Huye": 39000,
    "Rubavu": 45000,
}

land_use_factor = {
    "Residential": 1.00,
    "Commercial": 1.35,
    "Mixed-use": 1.15,
    "Agricultural": 0.60
}

zonings = ["R1", "R2", "R3", "C1", "C2", "A1"]

def generate_upi(index):
    return f"3/0{index%10}/05/01/{index:04d}"

# ----------------------------------------------
# GENERATE PARCEL MASTER DATASET
# ----------------------------------------------
rows = []
for i in range(N_PARCELS):
    district = np.random.choice(districts)
    sector = np.random.choice(sectors)  # <-- ADDED SECTOR
    land_use = np.random.choice(list(land_use_factor.keys()))
    zoning = np.random.choice(zonings)
    
    area_m2 = np.random.randint(200, 1500)
    distance_to_cbd = np.random.uniform(1, 15)
    distance_to_road = np.random.uniform(5, 300)
    slope = np.random.uniform(0, 15)
    amenities = np.random.uniform(0.2, 1.0)

    rows.append({
        "upi": generate_upi(i),
        "district": district,
        "sector": sector,                     # <-- ADDED HERE
        "land_use": land_use,
        "zoning": zoning,
        "area_m2": area_m2,
        "distance_to_cbd_km": distance_to_cbd,
        "distance_to_paved_road_m": distance_to_road,
        "slope_deg": slope,
        "amenities_score": amenities,
    })

parcel_master = pd.DataFrame(rows)

# ----------------------------------------------
# GENERATE LABELS (VERY LOW NOISE)
# ----------------------------------------------
def compute_price(row):
    base = district_base[row["district"]]

    price_per_m2 = (
        base * land_use_factor[row["land_use"]]
        * (1 - 0.02 * (row["distance_to_cbd_km"] / 10))
        * (1 - 0.01 * (row["distance_to_paved_road_m"] / 300))
        * (1 - 0.005 * (row["slope_deg"] / 10))
        * (0.9 + 0.2 * row["amenities_score"])
    )

    noise = np.random.normal(0, 0.005 * price_per_m2)  # VERY LOW NOISE

    total_price = (price_per_m2 + noise) * row["area_m2"]
    return total_price

# Pick training subset
training_idx = np.random.choice(parcel_master.index, size=int(N_PARCELS * TRAIN_FRACTION), replace=False)
training_parcels = parcel_master.loc[training_idx].copy()

training_parcels["sale_price"] = training_parcels.apply(compute_price, axis=1)
training_parcels["price_per_m2"] = training_parcels["sale_price"] / training_parcels["area_m2"]

# ----------------------------------------------
# SAVE DATA
# ----------------------------------------------
parcel_master.to_csv("parcel_master.csv", index=False)
training_parcels.to_csv("training_data.csv", index=False)

print("SUCCESS → Data generated!")
print("Parcels:", len(parcel_master))
print("Training:", len(training_parcels))
