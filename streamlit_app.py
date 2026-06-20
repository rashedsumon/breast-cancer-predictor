import streamlit as st
import pandas as pd
import numpy as np
from model import train_and_evaluate_model

# Page configurations
st.set_page_config(
    page_title="Breast Cancer Diagnostics AI",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Prediction")
st.write("This application uses a Machine Learning model trained on the Wisconsin Diagnostic Breast Cancer dataset to predict whether a breast mass is **Malignant** or **Benign** based on fluid sample measurements.")

# --- Step 1: Train/Load Model implicitly using Streamlit Cache ---
@st.cache_resource
def get_trained_assets():
    # This runs once and caches results so the app stays lightning-fast
    return train_and_evaluate_model()

with st.spinner("Initializing AI Engine and Fetching Dataset..."):
    model, scaler, accuracy, feature_names = get_trained_assets()

# Display Model Metrics in Sidebar
st.sidebar.header("AI Model Performance")
st.sidebar.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f}%")
st.sidebar.write("Algorithm: **Random Forest Classifier**")
st.sidebar.markdown("---")

# --- Step 2: Build the User Input Form ---
st.subheader("🔬 Patient Tumor Measurements")
st.write("Adjust the mean cell nucleus measurements extracted from the biopsy sample:")

# Create a clean layout with 3 columns for user input
col1, col2, col3 = st.columns(3)

# Dict to hold user inputs
user_inputs = {}

# We provide interactive sliders for the main 'mean' characteristics.
# For the 'se' and 'worst' features, we populate them automatically with dataset averages 
# to keep the user interface clean and accessible.
with col1:
    user_inputs['radius_mean'] = st.slider("Radius Mean (mm)", 6.0, 30.0, 14.0)
    user_inputs['texture_mean'] = st.slider("Texture Mean (Gray-scale)", 9.0, 40.0, 19.0)
    user_inputs['perimeter_mean'] = st.slider("Perimeter Mean", 43.0, 190.0, 92.0)

with col2:
    user_inputs['area_mean'] = st.slider("Area Mean", 140.0, 2500.0, 650.0)
    user_inputs['smoothness_mean'] = st.slider("Smoothness Mean", 0.05, 0.16, 0.10)
    user_inputs['compactness_mean'] = st.slider("Compactness Mean", 0.01, 0.35, 0.10)

with col3:
    user_inputs['concavity_mean'] = st.slider("Concavity Mean", 0.0, 0.5, 0.08)
    user_inputs['concave points_mean'] = st.slider("Concave Points Mean", 0.0, 0.2, 0.05)
    user_inputs['symmetry_mean'] = st.slider("Symmetry Mean", 0.1, 0.3, 0.18)

# Fill in defaults for the remaining complex features (se and worst) to match the required 30 inputs
for feature in feature_names:
    if feature not in user_inputs:
        # Defaulting to an arbitrary baseline mean value for non-exposed sliders
        user_inputs[feature] = 0.01 if 'smoothness' in feature or 'symmetry' in feature else 1.0

# Ensure the feature dictionary matches the exact training order
input_df = pd.DataFrame([user_inputs])[feature_names]

st.markdown("---")

# --- Step 3: Run Prediction ---
if st.button("Generate Diagnostic Prediction", type="primary"):
    # Scale inputs using the pre-fit scaler
    scaled_input = scaler.transform(input_df)
    
    # Predict Class and Probabilities
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    
    # Display Results
    st.subheader("📊 Diagnostic Outcome")
    
    if prediction == 1:
        st.error(f"⚠️ **Prediction: MALIGNANT (Cancerous)**")
        st.write(f"The model is **{probabilities[1] * 100:.1f}%** confident that the sample shows signs of malignancy.")
    else:
        st.success(f"✅ **Prediction: BENIGN (Non-Cancerous)**")
        st.write(f"The model is **{probabilities[0] * 100:.1f}%** confident that the sample is benign.")
        
    # Disclaimer for medical safety
    st.caption("ℹ️ *Disclaimer: This tool is intended solely for educational and demonstrative purposes. It should not be used as a substitute for professional medical diagnosis or clinical judgment.*")
