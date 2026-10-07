import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Ethiopian Agricultural Yield & Revenue Predictor",
    page_icon="🌾",
    layout="wide"
)

# App Header
st.title("🌾 Agricultural Yield, Market Price & Revenue Dashboard")
st.markdown("Forecast crop yields using your trained machine learning model across complete agronomic and socio-economic feature sets.")

# 2. Load Model Safely
@st.cache_resource
def load_model():
    model_path = 'models/best_yield_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# 3. Sidebar Configuration for ALL Inputs
st.sidebar.header("🔍 Input Parameters")

def user_input_features():
    # Location & Crop Details
    region = st.sidebar.selectbox("Region", ["Amhara", "Oromia", "SNNPR", "Tigray", "Sidama", "Central Ethiopia"])
    crop_type = st.sidebar.selectbox("Crop Type", ["Teff", "Maize", "Wheat", "Barley", "Sorghum"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌍 Environmental & Geography")
    altitude = st.sidebar.slider("Altitude (meters above sea level)", min_value=500.0, max_value=3500.0, value=2000.0, step=50.0)
    soil_quality = st.sidebar.selectbox("Soil Quality", ["Poor", "Medium", "Good", "Fertile"])
    rainfall_meher = st.sidebar.slider("Meher Season Rainfall (mm)", min_value=200.0, max_value=1400.0, value=650.0, step=25.0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🚜 Farm Management & Inputs")
    farm_size = st.sidebar.number_input("Farm Size (hectares)", min_value=0.1, max_value=50.0, value=1.5, step=0.1)
    fertilizer_amount = st.sidebar.number_input("Fertilizer Applied (kg/ha)", min_value=0.0, max_value=500.0, value=100.0, step=10.0)
    improved_seed = st.sidebar.selectbox("Improved Seed Used?", ["Yes", "No"])
    labor_days = st.sidebar.number_input("Labor Days Invested", min_value=1.0, max_value=300.0, value=45.0, step=5.0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚠️ Risks & Infrastructure")
    pest_flag = st.sidebar.selectbox("Pest / Disease Flag", ["None", "Low", "Moderate", "Severe"])
    distance_to_market = st.sidebar.slider("Distance to Market (km)", min_value=0.5, max_value=100.0, value=10.0, step=0.5)
    
    # Map inputs to dictionary structure matching your data pipeline
    data = {
        'region': region,
        'crop_type': crop_type,
        'altitude': altitude,
        'soil_quality': soil_quality,
        'rainfall_meher': rainfall_meher,
        'farm_size': farm_size,
        'fertilizer_amount': fertilizer_amount,
        'improved_seed_used': 1 if improved_seed == "Yes" else 0,
        'labor_days': labor_days,
        'pest_flag': 0 if pest_flag == "None" else (1 if pest_flag == "Low" else (2 if pest_flag == "Moderate" else 3)),
        'distance_to_market': distance_to_market
    }
    return pd.DataFrame([data], index=[0])

input_df = user_input_features()

# Display current selected inputs for verification
st.subheader("📋 Current Selection Summary")
st.dataframe(input_df, use_container_width=True)

# 4. Dynamic Market Price Lookup Dictionary (ETB per Quintal)
market_prices = {
    "Teff": 7630,
    "Maize": 3500,
    "Wheat": 4800,
    "Barley": 4200,
    "Sorghum": 4100
}

selected_crop = input_df['crop_type'].iloc[0]
current_market_price = market_prices.get(selected_crop, 4000)

# 5. Prediction Execution
if st.button("🚀 Calculate Yield & Revenue", type="primary"):
    if model is None:
        st.error("❌ Model artifact not found! Please ensure `models/best_yield_model.pkl` exists in your directory.")
    else:
        try:
            # Align features with model expectations if model has `feature_names_in_`
            processed_input = input_df.copy()
            if hasattr(model, "feature_names_in_"):
                for col in model.feature_names_in_:
                    if col not in processed_input.columns:
                        processed_input[col] = 0  # Fallback for missing structural columns
                processed_input = processed_input[model.feature_names_in_]

            # Predict Yield (tons per hectare)
            predicted_yield_per_ha = model.predict(processed_input)[0]
            
            # Scale yield by farm size for total volume
            total_farm_size = input_df['farm_size'].iloc[0]
            total_predicted_yield_tons = predicted_yield_per_ha * total_farm_size
            
            # Convert tons to quintals (1 ton = 10 quintals)
            quintals_per_ha = predicted_yield_per_ha * 10
            total_quintals = total_predicted_yield_tons * 10
            
            # Calculate Gross Revenue (ETB)
            gross_revenue_per_ha = quintals_per_ha * current_market_price
            total_gross_revenue = total_quintals * current_market_price

            # 6. Display Results Dashboard
            st.success("Prediction Generated Successfully!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="🌾 Yield per Hectare", 
                    value=f"{predicted_yield_per_ha:.2f} tons/ha",
                    delta=f"{quintals_per_ha:.1f} qt/ha"
                )
                
            with col2:
                st.metric(
                    label="💰 Market Price", 
                    value=f"{current_market_price:,.2f} ETB/qt",
                    delta=selected_crop
                )
                
            with col3:
                st.metric(
                    label="📈 Total Est. Gross Revenue", 
                    value=f"{total_gross_revenue:,.2f} ETB",
                    delta=f"For {total_farm_size} hectares"
                )
                
            # Additional Breakdown Insight
            st.info(
                f"**Farm Financial Breakdown:** Managing **{total_farm_size} ha** of **{selected_crop}** in **{input_df['region'].iloc[0]}** "
                f"generates an estimated total harvest of **{total_quintals:,.1f} quintals**, bringing an aggregate gross revenue of **{total_gross_revenue:,.2f} ETB** "
                f"({gross_revenue_per_ha:,.2f} ETB/ha)."
            )

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Ensure your categorical variables (like `soil_quality`) match the encoding format used when training the model.")

# Footer
st.markdown("---")
st.markdown("🛠️ *Ethiopian Agricultural Forecasting Pipeline | Powered by Machine Learning & Streamlit*")