import streamlit as st
import pandas as pd
import joblib

# Load model + data
model_bundle = joblib.load("model_land_price.pkl")
pipeline = model_bundle["pipeline"]
FEATURE_COLS = model_bundle["feature_cols"]

parcel_master = pd.read_csv("parcel_master.csv")
parcel_master["upi"] = parcel_master["upi"].astype(str)

def get_features_by_upi(upi):
    subset = parcel_master[parcel_master["upi"] == upi]
    if subset.empty:
        return None, None
    row = subset.iloc[0]
    features = row[FEATURE_COLS]
    return row, features

st.title("Land Price Estimator (UPI-based)")

upi_input = st.text_input("Enter UPI (e.g., 3/08/05/01/0001)")

if st.button("Estimate Price"):
    row, features = get_features_by_upi(upi_input)

    if row is None:
        st.error("UPI not found!")
    else:
        X_input = pd.DataFrame([features.to_dict()])
        estimated_price = float(pipeline.predict(X_input)[0])
        price_per_m2 = estimated_price / float(row["area_m2"])

        st.subheader("Parcel Information")
        st.json(row.to_dict())

        st.subheader("Estimated Market Price")
        st.write(f"**RWF {estimated_price:,.0f}**")

        st.subheader("Price per m²")
        st.write(f"**RWF {price_per_m2:,.0f}**")
