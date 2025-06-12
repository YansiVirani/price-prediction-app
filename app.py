import streamlit as st
import pandas as pd
import numpy as np
from house_price_model import HousePriceModel

# Set page config
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 House Price Prediction")
st.markdown("""
This application predicts house prices based on various features using the California Housing dataset.
Enter the house details below to get a price prediction.
""")

# Initialize the model
@st.cache_resource
def load_model():
    model = HousePriceModel()
    model.load_data()
    model.train_models()
    return model

# Load the model
try:
    model = load_model()
    
    # Sidebar for model information
    st.sidebar.header("Model Information")
    st.sidebar.markdown("""
    ### Features Used:
    - MedInc: Median income in block group
    - HouseAge: House age in years
    - AveRooms: Average number of rooms per household
    - AveBedrms: Average number of bedrooms per household
    - Population: Block group population
    - AveOccup: Average number of household members
    - Latitude: Block group latitude
    - Longitude: Block group longitude
    """)
    
    # Main content
    st.header("Enter House Details")
    
    # Create input fields for features
    col1, col2 = st.columns(2)
    
    with col1:
        med_inc = st.number_input("Median Income (in $10,000s)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
        house_age = st.number_input("House Age (years)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
        ave_rooms = st.number_input("Average Rooms", min_value=1.0, max_value=20.0, value=6.0, step=0.1)
        ave_bedrms = st.number_input("Average Bedrooms", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
    
    with col2:
        population = st.number_input("Population", min_value=0, max_value=10000, value=1000, step=100)
        ave_occup = st.number_input("Average Occupancy", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
        latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=37.0, step=0.1)
        longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-120.0, step=0.1)
    
    # Create feature array
    features = np.array([[
        med_inc, house_age, ave_rooms, ave_bedrms,
        population, ave_occup, latitude, longitude
    ]])
    
    # Prediction button
    if st.button("Predict Price"):
        try:
            # Make prediction
            prediction = model.predict_price(features)
            
            # Display prediction
            st.success(f"Predicted House Price: ${prediction*100000:,.2f}")
            
            # Display model performance metrics
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("R² Score", f"{model.train_models()['XGBoost']['r2']:.3f}")
            with col2:
                st.metric("Mean Squared Error", f"{model.train_models()['XGBoost']['mse']:.3f}")
            
            # Display visualizations
            st.subheader("Data Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.pyplot(model.plot_price_distribution())
            with col2:
                st.pyplot(model.plot_feature_importance())
            
            st.pyplot(model.plot_correlation_matrix())
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")
    st.error("Please try refreshing the page.")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>⚠️ Note: This model uses the California Housing dataset. Predictions are in $100,000s and may not reflect actual market prices.</p>
</div>
""", unsafe_allow_html=True) 