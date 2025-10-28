# stream-app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# === Page Configuration ===
st.set_page_config(page_title="House Price Prediction", layout="centered")

# === Page Title ===
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Enter the property details to predict the estimated price and see the most influential factors.</p>", unsafe_allow_html=True)

# === Load Model and Features ===
with open('random_forest_model.sav', 'rb') as f:
    model = pickle.load(f)
st.sidebar.success("Random Forest model loaded.")

with open('model_features.sav', 'rb') as f:
    features = pickle.load(f)
st.sidebar.success("Model features loaded.")

# === User Input ===
st.subheader("Input Property Details")

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Location", [col.replace('loc_', '') for col in features if col.startswith('loc_')])
    bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1, value=3)
    toilet = st.number_input("Number of Bathrooms", min_value=0, step=1, value=2)
    garage = st.number_input("Number of Garages", min_value=0, step=1, value=1)

with col2:
    LT = st.number_input("Land Area (m²)", min_value=0.0, step=1.0, value=100.0)
    LB = st.number_input("Building Area (m²)", min_value=0.0, step=1.0, value=80.0)
    price_ref = st.number_input("Actual Price (optional)", min_value=0.0, step=1000000.0, value=0.0)

# === Prepare Input Data ===
input_data = pd.DataFrame({
    'bedrooms': [bedrooms],
    'toilet': [toilet],
    'garage': [garage],
    'LT': [LT],
    'LB': [LB]
})

# Add One-Hot encoded location columns
for loc_col in [col for col in features if col.startswith('loc_')]:
    input_data[loc_col] = 1 if loc_col == f'loc_{location}' else 0

# Ensure same feature order as the model
input_data = input_data[features]

# === Prediction ===
if st.button("Predict"):
    try:
        predicted_price = model.predict(input_data)[0]
        st.success(f"Predicted House Price: Rp {predicted_price:,.0f}")

        # === Feature Importance ===
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        main_features = ['bedrooms', 'toilet', 'garage', 'LT', 'LB'] + [col for col in importance['Feature'] if col.startswith('loc_')]
        importance = importance[importance['Feature'].isin(main_features)]

        importance['Percentage'] = (importance['Importance'] / importance['Importance'].sum()) * 100

        st.subheader("Feature Importance (Percentage Influence)")
        st.bar_chart(importance.set_index('Feature')['Percentage'])

        top_feature = importance.iloc[0]
        st.info(f"The most influential factor is '{top_feature['Feature']}' with approximately {top_feature['Percentage']:.2f}% impact.")

        # Optional: Compare with actual price
        if price_ref > 0:
            error = abs(predicted_price - price_ref)
            st.write(f"Difference from actual price: Rp {error:,.0f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# === Footer ===
st.markdown("---")
st.markdown("<p style='text-align: center;'>© 2025 House Price Prediction | by Raka Arrayan</p>", unsafe_allow_html=True)
