import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Title
st.title("🍹 Bar Inventory Forecasting App")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("xgb_inventory_model.pkl")

model = load_model()

# Feature template (as seen during training)
@st.cache_data
def get_feature_template():
    # Should match the one used during training
    return pd.read_pickle("X_feature_template.pkl")

feature_template = get_feature_template()
all_bars = [col.replace("Bar_", "") for col in feature_template.columns if col.startswith("Bar_")]
all_brands = [col.replace("Brand_", "") for col in feature_template.columns if col.startswith("Brand_")]
all_alcohols = [col.replace("Alcohol_", "") for col in feature_template.columns if col.startswith("Alcohol_")]

# User Inputs
st.subheader("📥 Enter Input Details")

bar = st.selectbox("Select Bar", all_bars)
brand = st.selectbox("Select Brand", all_brands)
alcohol = st.selectbox("Select Alcohol Type", all_alcohols)
opening_balance = st.number_input("Opening Balance (ml)", min_value=0.0)
purchase = st.number_input("Purchase (ml)", min_value=0.0)
closing_balance = st.number_input("Closing Balance (ml)", min_value=0.0)
day_of_week = st.selectbox("Day of Week (0=Mon, 6=Sun)", list(range(7)))
month = st.selectbox("Month", list(range(1, 13)))
hour = st.selectbox("Hour (0-23)", list(range(24)))
lag_1 = st.number_input("Lag 1 (Previous day consumption)", min_value=0.0)
lag_2 = st.number_input("Lag 2", min_value=0.0)
roll_3 = st.number_input("Rolling Mean (3 days)", min_value=0.0)

# Prediction Trigger
if st.button("🔮 Predict Consumption"):
    # Build input row
    input_dict = {
        "Opening Balance (ml)": opening_balance,
        "Purchase (ml)": purchase,
        "Closing Balance (ml)": closing_balance,
        "DayOfWeek": day_of_week,
        "Month": month,
        "Hour": hour,
        "lag_1": lag_1,
        "lag_2": lag_2,
        "roll_3": roll_3
    }

    # Add one-hot columns
    for col in feature_template.columns:
        if col.startswith("Bar_"):
            input_dict[col] = 1 if col == f"Bar_{bar}" else 0
        elif col.startswith("Brand_"):
            input_dict[col] = 1 if col == f"Brand_{brand}" else 0
        elif col.startswith("Alcohol_"):
            input_dict[col] = 1 if col == f"Alcohol_{alcohol}" else 0
        elif col not in input_dict:
            input_dict[col] = 0  # fill any other column with 0

    input_df = pd.DataFrame([input_dict])[feature_template.columns]

    # Predict
    prediction = model.predict(input_df)[0]
    recommended = np.ceil(prediction * 1.15)

    st.success(f"📊 Predicted Consumption: **{prediction:.2f} ml**")
    st.info(f"✅ Recommended Par Level (with 15% buffer): **{recommended:.0f} ml**")