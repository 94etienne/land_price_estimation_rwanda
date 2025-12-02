from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

# ----------------------------------------------------
# LOAD MODEL + DATA
# ----------------------------------------------------
model_bundle = joblib.load("model_land_price.pkl")
pipeline = model_bundle["pipeline"]
FEATURE_COLS = model_bundle["feature_cols"]

parcel_master = pd.read_csv("parcel_master.csv")
parcel_master["upi"] = parcel_master["upi"].astype(str)

# ----------------------------------------------------
# FLASK APP SETUP
# ----------------------------------------------------
app = Flask(__name__)
CORS(app)

PRED_FILE = "predictions.csv"

# Create predictions file if not exists
if not os.path.exists(PRED_FILE):
    pd.DataFrame(columns=[
        "upi","district","sector","land_use","zoning",
        "area_m2","distance_to_cbd_km","distance_to_paved_road_m",
        "slope_deg","amenities_score","estimated_price","price_per_m2"
    ]).to_csv(PRED_FILE, index=False)


# ----------------------------------------------------
# HELPER FUNCTION
# ----------------------------------------------------
def get_features_by_upi(upi):
    """Returns the row (all fields) and the subset of columns used by model"""
    subset = parcel_master[parcel_master["upi"] == upi]
    if subset.empty:
        return None, None
    row = subset.iloc[0]
    features = row[FEATURE_COLS]  # only model features
    return row, features


# ----------------------------------------------------
# PREDICTION API ENDPOINT
# ----------------------------------------------------
@app.route("/predict", methods=["GET"])
def predict():
    upi = request.args.get("upi")

    if not upi:
        return jsonify({"error": "UPI is required"}), 400

    row, features = get_features_by_upi(upi)
    if row is None:
        return jsonify({"error": "UPI not found"}), 404

    X_input = pd.DataFrame([features.to_dict()])
    estimated_price = float(pipeline.predict(X_input)[0])
    price_per_m2 = estimated_price / float(row["area_m2"])

    # ---------------- SAVE PREDICTION ----------------
    df = pd.read_csv(PRED_FILE)

    df.loc[len(df)] = [
        upi,
        row["district"],
        row["sector"],
        row["land_use"],
        row["zoning"],
        row["area_m2"],
        row["distance_to_cbd_km"],
        row["distance_to_paved_road_m"],
        row["slope_deg"],
        row["amenities_score"],
        estimated_price,
        price_per_m2
    ]

    df.to_csv(PRED_FILE, index=False)

    # ---------------- RETURN JSON RESPONSE ----------------
    return jsonify({
        "upi": upi,
        "district": row["district"],
        "sector": row["sector"],
        "land_use": row["land_use"],
        "zoning": row["zoning"],
        "area_m2": float(row["area_m2"]),
        "distance_to_cbd_km": float(row["distance_to_cbd_km"]),
        "distance_to_paved_road_m": float(row["distance_to_paved_road_m"]),
        "slope_deg": float(row["slope_deg"]),
        "amenities_score": float(row["amenities_score"]),
        "estimated_price": estimated_price,
        "estimated_price_per_m2": price_per_m2
    })


# ----------------------------------------------------
# DASHBOARD API - RETURN ALL PREDICTIONS
# ----------------------------------------------------
@app.route("/predictions", methods=["GET"])
def get_predictions():
    df = pd.read_csv(PRED_FILE)

    stats = {
        "total_predictions": len(df),
        "avg_price": float(df["estimated_price"].mean()) if len(df) > 0 else 0,
        "max_price": float(df["estimated_price"].max()) if len(df) > 0 else 0,
        "min_price": float(df["estimated_price"].min()) if len(df) > 0 else 0
    }

    return jsonify({
        "statistics": stats,
        "records": df.to_dict(orient="records")
    })


# ----------------------------------------------------
# RUN FLASK APP
# ----------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
