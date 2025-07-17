import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("best_model.joblib")
encoders = joblib.load("encoders.joblib")
scalers = joblib.load("scaled.joblib")

st.set_page_config(page_title="🏠 House Price Predictor", layout="wide")
st.title("🏠 House Price Prediction App")
st.sidebar.header("🔧 Input Features")

columns = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
    '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
    'GarageFinish', 'KitchenQual', 'BsmtQual', 'ExterQual'
]

numerics = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
            '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']

categoricals = ['GarageFinish', 'KitchenQual', 'BsmtQual', 'ExterQual']

user_input = {}

st.sidebar.markdown("### Numeric Features")
for col in numerics:
    value = st.sidebar.number_input(f"{col}", min_value=0, value=1000 if "Area" in col or "SF" in col else 5)
    user_input[col] = value

st.sidebar.markdown("### Categorical Features")
for col in categoricals:
    options = encoders[col].classes_.tolist()
    choice = st.sidebar.selectbox(col, options)
    user_input[col] = choice

input_df = pd.DataFrame([user_input])

for col in categoricals:
    input_df[col] = encoders[col].transform(input_df[col])

for col in scalers:
    input_df[col] = scalers[col].transform(input_df[[col]])

def validate_inputs(data):
    errors = []

    if data["YearBuilt"] < 1800 or data["YearBuilt"] > 2025:
        errors.append("🛠️ YearBuilt must be between 1800 and 2025.")
    
    if data["YearRemodAdd"] < data["YearBuilt"]:
        errors.append("🔧 YearRemodAdd cannot be earlier than YearBuilt.")

    for area in ['GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF']:
        if data[area] <= 0:
            errors.append(f"📏 {area} must be greater than 0.")

    if data["OverallQual"] < 1 or data["OverallQual"] > 10:
        errors.append("⭐ OverallQual must be between 1 and 10.")

    return errors

# Predict
if st.button("💰 Predict Price"):
    errors = validate_inputs(user_input)

    if errors:
        for err in errors:
            st.error(err)
        st.warning("⚠️ Please correct the inputs above to get a valid prediction.")
    else:
        st.write("🧪 Input to model:", input_df)
        prediction_log = model.predict(input_df)[0]
        st.write(f"📈 Raw Prediction (log scale): {prediction_log}")
        pred_price = np.expm1(prediction_log)
        st.success(f"🏡 Estimated Sale Price: ${pred_price:,.2f}")
 
with st.expander("🔍 See Input Details"):
    st.write("### Processed Input Features")
    st.dataframe(input_df)

with st.expander("📊 About this App"):
    st.markdown("""
        This app uses an XGBoost model trained on housing data to predict the sale price of a house.

        **Features Used:**
        - Overall Quality
        - Living Area (GrLivArea)
        - Garage Capacity and Area
        - Basement and Floor Square Footage
        - Bathroom Count
        - Room Count
        - Year Built & Renovated
        - Encoded Categorical Quality Measures

        **Author:** Your Name · **Model Accuracy:** ~86% R²
    """)
