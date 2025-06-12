# House Price Prediction

This project implements a machine learning model to predict house prices based on various features using the California Housing dataset. The application includes a Streamlit web interface for easy interaction.

## Features

- Uses the California Housing dataset
- Implements multiple regression models:
  - Linear Regression
  - Decision Tree
  - XGBoost
- Provides interactive web interface
- Shows model performance metrics
- Visualizes feature importance and correlations
- Real-time price predictions

## Requirements

- Python 3.8+
- Required packages are listed in `requirements.txt`

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. The web interface will open in your default browser. You can:
   - Enter house details using the input fields
   - Click "Predict Price" to get a price prediction
   - View model performance metrics
   - Explore data visualizations

## Features Used

The model uses the following features from the California Housing dataset:
- MedInc: Median income in block group
- HouseAge: House age in years
- AveRooms: Average number of rooms per household
- AveBedrms: Average number of bedrooms per household
- Population: Block group population
- AveOccup: Average number of household members
- Latitude: Block group latitude
- Longitude: Block group longitude

## Model Performance

The application shows:
- R² Score: Measures the proportion of variance in the dependent variable predictable from the independent variable(s)
- Mean Squared Error: Measures the average squared difference between predicted and actual values
- Feature Importance: Shows which features have the most impact on price predictions
- Price Distribution: Visualizes the distribution of house prices in the dataset
- Correlation Matrix: Shows relationships between different features

## Disclaimer

This model uses the California Housing dataset. The predictions are in $100,000s and may not reflect actual market prices. This tool is for educational purposes only. 