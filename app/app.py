import streamlit as st
import pandas as pd
import os
import sys

# -----------------------------
# Setup
# -----------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import ModelInference

st.set_page_config(
    page_title="Disaster Severity Intelligence System",
    page_icon="🚨",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_pipeline():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'pipeline.pkl')
    return ModelInference(model_path=model_path)

try:
    inferencer = load_pipeline()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model load failed: {e}")

# -----------------------------
# Title
# -----------------------------
st.title("🚨 Disaster Severity Intelligence System")

st.markdown("""
This system predicts disaster severity using historical environmental and infrastructure data.
It also classifies whether the event is a **Major Disaster**.
""")

# -----------------------------
# Main UI
# -----------------------------
if model_loaded:

    st.subheader("📥 Enter Incident Parameters")

    with st.form("input_form"):

        col1, col2 = st.columns(2)

        # -------------------------
        # INPUTS
        # -------------------------
        with col1:
            disaster_type = st.selectbox(
                "Disaster Type",
                ["Flood", "Earthquake", "Hurricane", "Wildfire"]
            )

            location = st.selectbox(
                "Location Type",
                ["Urban", "Suburban", "Rural"]
            )

            latitude = st.number_input("Latitude", value=12.97)
            longitude = st.number_input("Longitude", value=77.59)

            affected_population = st.number_input("Affected Population", value=5000)

        with col2:
            economic_loss = st.number_input("Estimated Economic Loss (USD)", value=100000)

            response_time = st.number_input("Response Time (Hours)", value=2)

            aid_provided = st.selectbox(
                "Aid Provided",
                ["Yes", "No"]
            )

            infra_damage = st.slider(
                "Infrastructure Damage Index",
                0.0, 1.0, 0.5
            )

        submit = st.form_submit_button("Predict Severity")

    # -----------------------------
    # PREDICTION
    # -----------------------------
    if submit:

        input_df = pd.DataFrame({
            "disaster_type": [disaster_type],
            "location": [location],
            "latitude": [latitude],
            "longitude": [longitude],
            "affected_population": [affected_population],
            "estimated_economic_loss_usd": [economic_loss],
            "response_time_hours": [response_time],
            "aid_provided": [aid_provided],
            "infrastructure_damage_index": [infra_damage]
        })

        with st.spinner("Analyzing disaster risk..."):
            prediction = inferencer.predict(input_df)[0]
            probabilities = inferencer.predict_proba(input_df)

        # -----------------------------
        # INTERPRETATION LAYER
        # -----------------------------
        st.subheader("📊 Prediction Result")

        severity_map = {
            0: "LOW",
            1: "MEDIUM",
            2: "HIGH"
        }

        try:
            severity_level = int(prediction)
            severity_label = severity_map.get(severity_level, str(prediction))
        except:
            severity_label = str(prediction)

        # Major disaster logic
        is_major_disaster = severity_level >= 2 if isinstance(prediction, (int, float)) else False

        # -----------------------------
        # DISPLAY RESULT
        # -----------------------------
        if severity_label == "HIGH":
            st.error(f"🚨 SEVERITY: {severity_label}")
        elif severity_label == "MEDIUM":
            st.warning(f"⚠️ SEVERITY: {severity_label}")
        else:
            st.success(f"🟢 SEVERITY: {severity_label}")

        if is_major_disaster:
            st.error("🚨 MAJOR DISASTER ALERT")
        else:
            st.success("🟢 NON-MAJOR INCIDENT")

        # -----------------------------
        # CONFIDENCE SCORES
        # -----------------------------
        if probabilities is not None:
            st.write("### Confidence Scores")

            try:
                classes = inferencer.pipeline.classes_
                prob_df = pd.DataFrame(probabilities, columns=classes)
                st.dataframe(prob_df.style.format("{:.2%}"))
            except:
                st.write(probabilities)