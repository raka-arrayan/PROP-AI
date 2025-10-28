import streamlit as st
import pickle
import pandas as pd
import shap
import numpy as np

# === Page Configuration ===
st.set_page_config(page_title="House Price Prediction", layout="centered")

# === Page Title ===
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Enter property details to predict the estimated price and see which features most influence your result.</p>", unsafe_allow_html=True)

# === Load Model and Features ===
with open('random_forest_model.sav', 'rb') as f:
    model = pickle.load(f)
st.sidebar.write("Random Forest model loaded")

with open('model_features.sav', 'rb') as f:
    features = pickle.load(f)
st.sidebar.write("Model features loaded")

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

# Add one-hot encoded location columns efficiently
location_df = pd.DataFrame({
    col: [1 if col == f'loc_{location}' else 0] for col in features if col.startswith('loc_')
})

# Combine both dataframes at once (avoid fragmentation)
input_data = pd.concat([input_data, location_df], axis=1)

# Ensure all features exist
for col in features:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[features]


# === Prediction ===
if st.button("Predict"):
    try:
        predicted_price = model.predict(input_data)[0]
        st.success(f"Predicted House Price: Rp {predicted_price:,.0f}")

        # === SHAP Interpretation (Dynamic Feature Influence) ===
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        # Get absolute contribution per feature for this prediction
        abs_contrib = np.abs(shap_values[0])
        contribution_df = pd.DataFrame({
            'Feature': features,
            'Contribution': abs_contrib
        })

        # Combine location columns
        loc_contrib = contribution_df[contribution_df['Feature'].str.startswith('loc_')]['Contribution'].sum()
        main_contrib = contribution_df[~contribution_df['Feature'].str.startswith('loc_')].copy()
        main_contrib = pd.concat([
            main_contrib,
            pd.DataFrame({'Feature': ['Location'], 'Contribution': [loc_contrib]})
        ])

        # Normalize to percentage
        main_contrib['Percentage'] = (main_contrib['Contribution'] / main_contrib['Contribution'].sum()) * 100
        main_contrib = main_contrib.sort_values(by='Percentage', ascending=False)

        # === Display Dynamic Feature Influence ===
        st.subheader("Feature Importance ")
        for _, row in main_contrib.iterrows():
            st.write(f"**{row['Feature']}** {row['Percentage']:.1f}%")
            st.progress(row['Percentage'] / 100)

        top_feature = main_contrib.iloc[0]
        st.info(f"The most influential factor for this prediction is **{top_feature['Feature']}**, contributing approximately {top_feature['Percentage']:.1f}% to the predicted price.")

        # === Optional: Compare with actual price
        if price_ref > 0:
            diff = abs(predicted_price - price_ref)
            st.write(f"Difference from actual price: Rp {diff:,.0f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# === Footer ===
st.markdown("---")
st.markdown("<p style='text-align: center;'>© 2025 House Price Prediction | by Raka Arrayan</p>", unsafe_allow_html=True)
