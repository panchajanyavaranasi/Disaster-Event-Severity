import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import pickle


# -----------------------------
# Load Model Artifacts
# -----------------------------
# Streamlit Cloud working directory is the repo root
# BASE_DIR = os.path.dirname(__file__)
# MODELS_DIR = os.path.join(BASE_DIR, "models")

try:
    import sklearn
    st.success(f"sklearn installed: {sklearn.__version__}")
except Exception as e:
    st.error(f"sklearn NOT installed: {e}")


try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "..", "models", "knn_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")
    imputer_path = os.path.join(BASE_DIR, "..", "models", "imputer.pkl")
    columns_path = os.path.join(BASE_DIR, "..", "models", "columns.pkl")

    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    imputer = pickle.load(open(imputer_path, "rb"))
    columns = pickle.load(open(columns_path, "rb"))
    
    
except FileNotFoundError as e:
    st.write(f"sklearn version: {sklearn.__version__}")
    st.error(f"⚠️ Model file not found: {e}")
    
    # st.stop()

st.title("📊 Sales Revenue Prediction")

st.write("Enter campaign details below:")

# -----------------------------
# Categorical Inputs
# -----------------------------
region = st.selectbox("Region", ["North", "South", "East", "West"])
channel = st.selectbox("Channel", ["Social Media", "Email", "TV", "SEO"])
product_category = st.selectbox("Product Category", ["Electronics", "Clothing", "Food"])
customer_segment = st.selectbox("Customer Segment", ["Retail", "Corporate", "Home Office"])

# -----------------------------
# Numerical Inputs
# -----------------------------
ad_spend = st.number_input("Ad Spend", min_value=0.0)
price = st.number_input("Price", min_value=0.0)
discount_rate = st.number_input("Discount Rate", min_value=0.0)
market_reach = st.number_input("Market Reach", min_value=0.0)
impressions = st.number_input("Impressions", min_value=0)
ctr = st.number_input("Click Through Rate", min_value=0.0)
competition_index = st.number_input("Competition Index", min_value=0.0)
seasonality_index = st.number_input("Seasonality Index", min_value=0.0)
campaign_days = st.number_input("Campaign Duration Days", min_value=0.0)
clv = st.number_input("Customer Lifetime Value", min_value=0.0)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Revenue"):

    input_dict = {
        "region": region,
        "channel": channel,
        "product_category": product_category,
        "customer_segment": customer_segment,
        "ad_spend": ad_spend,
        "price": price,
        "discount_rate": discount_rate,
        "market_reach": market_reach,
        "impressions": impressions,
        "click_through_rate": ctr,
        "competition_index": competition_index,
        "seasonality_index": seasonality_index,
        "campaign_duration_days": campaign_days,
        "customer_lifetime_value": clv
    }

    input_df = pd.DataFrame([input_dict])

    # Separate types
    num_cols = input_df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = input_df.select_dtypes(include=['object']).columns

    # Impute numeric
    input_df[num_cols] = imputer.transform(input_df[num_cols])

    # Encode categorical
    input_encoded = pd.get_dummies(input_df)

    # Align columns
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_encoded)

    # Predict
    prediction = model.predict(input_scaled)

    st.success(f"💰 Predicted Revenue: ₹ {prediction[0]:,.2f}")