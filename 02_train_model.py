import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------------------------
# LOAD TRAINING DATA
# ----------------------------------------------
df = pd.read_csv("training_data.csv")

# We will predict total sale_price
TARGET_COL = "sale_price"

# Features to use (NO UPI, NO price columns)
FEATURE_COLS = [
    "district",
    "sector",
    "land_use",
    "zoning",
    "area_m2",
    "distance_to_cbd_km",
    "distance_to_paved_road_m",
    "slope_deg",
    "amenities_score",
]

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# ----------------------------------------------
# TRAIN / TEST SPLIT
# ----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------
# PREPROCESSING
# ----------------------------------------------
categorical_features = ["district", "sector", "land_use", "zoning"]
numeric_features = [
    "area_m2",
    "distance_to_cbd_km",
    "distance_to_paved_road_m",
    "slope_deg",
    "amenities_score",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features),
    ]
)

# ----------------------------------------------
# MODEL
# ----------------------------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ]
)

# ----------------------------------------------
# TRAIN
# ----------------------------------------------
print("Training model...")
pipeline.fit(X_train, y_train)

# ----------------------------------------------
# EVALUATE
# ----------------------------------------------
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:,.0f} RWF")
print(f"R² : {r2:.3f}")

# ----------------------------------------------
# SAVE MODEL & FEATURE INFO
# ----------------------------------------------
joblib.dump(
    {
        "pipeline": pipeline,
        "feature_cols": FEATURE_COLS,
    },
    "model_land_price.pkl"
)

print(" Saved model_land_price.pkl")
