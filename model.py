import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def load_data():
    """Load California Housing dataset"""
    california = fetch_california_housing()
    X = pd.DataFrame(california.data, columns=california.feature_names)
    y = california.target
    return X, y

def preprocess_data(X, y):
    """Preprocess the data"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_models(X_train, y_train):
    """Train multiple regression models"""
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate model performance"""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {
            'MSE': mse,
            'R2': r2
        }
    return results

def save_models(models, scaler):
    """Save trained models and scaler"""
    if not os.path.exists('models'):
        os.makedirs('models')
    
    for name, model in models.items():
        joblib.dump(model, f'models/{name.lower().replace(" ", "_")}.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')

def main():
    # Load data
    X, y = load_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Print results
    print("\nModel Performance:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"MSE: {metrics['MSE']:.4f}")
        print(f"R2 Score: {metrics['R2']:.4f}")
    
    # Save models
    save_models(models, scaler)
    print("\nModels saved successfully!")

if __name__ == "__main__":
    main() 