import streamlit as st
import pickle
import pandas as pd
import numpy as np

# === Page Configuration ===
st.set_page_config(page_title="House Price Prediction", layout="centered")

# === Page Title ===
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Enter the property details to predict the estimated price and see the most influential features.</p>", unsafe_allow_html=True)

# === Load Model and Features ===
with open('random_forest_model.sav', 'rb') as f:
    model = pickle.load(f)
st.sidebar.write("Random Forest model loaded.")

with open('model_features.sav', 'rb') as f:
    features = pickle.load(f)
st.sidebar.write("Model features loaded.")

# === User Input Section ===
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

# Add One-Hot Encoded Location Columns
for loc_col in [col for col in features if col.startswith('loc_')]:
    input_data[loc_col] = 1 if loc_col == f'loc_{location}' else 0

# Ensure all features exist in the input_data (to avoid KeyError)
for col in features:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns to match model training
input_data = input_data[features]

# === Prediction ===
if st.button("Predict"):
    try:
        predicted_price = model.predict(input_data)[0]
        st.success(f"Predicted House Price: Rp {predicted_price:,.0f}")

        # === Feature Importance Calculation ===
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        })

        # Combine all location columns into a single "Location" feature
        loc_importance = importance_df[importance_df['Feature'].str.startswith('loc_')]['Importance'].sum()

        # Keep only main numeric features
        main_features = importance_df[~importance_df['Feature'].str.startswith('loc_')].copy()

        # Add combined location importance
        main_features = pd.concat([
            main_features,
            pd.DataFrame({'Feature': ['Location'], 'Importance': [loc_importance]})
        ])

        # Normalize to percentage
        main_features['Percentage'] = (main_features['Importance'] / main_features['Importance'].sum()) * 100
        main_features = main_features.sort_values(by='Percentage', ascending=False)

        # === Display Feature Importance ===
        st.subheader("Feature Importance (Percentage Influence)")

        for _, row in main_features.iterrows():
            st.write(f"**{row['Feature']}** — {row['Percentage']:.0f}%")
            st.progress(row['Percentage'] / 100)

        # Display the most influential feature
        top_feature = main_features.iloc[0]
        st.info(f"The most influential factor is '{top_feature['Feature']}' with approximately {top_feature['Percentage']:.1f}% impact on the predicted price.")

        # Optional: Compare with actual price if provided
        if price_ref > 0:
            diff = abs(predicted_price - price_ref)
            st.write(f"Difference from actual price: Rp {diff:,.0f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# === Footer ===
st.markdown("---")
st.markdown("<p style='text-align: center;'>© 2025 House Price Prediction | by Raka Arrayan</p>", unsafe_allow_html=True)
