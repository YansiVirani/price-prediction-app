import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class HousePriceModel:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'XGBoost': XGBRegressor(random_state=42)
        }
        self.best_model = None
        self.feature_names = None
        
    def load_data(self):
        """Load and prepare the California Housing dataset"""
        # Load the dataset
        california = fetch_california_housing()
        self.data = pd.DataFrame(california.data, columns=california.feature_names)
        self.data['PRICE'] = california.target
        self.feature_names = california.feature_names
        
        # Split features and target
        self.X = self.data.drop('PRICE', axis=1)
        self.y = self.data['PRICE']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.data
    
    def train_models(self):
        """Train all models and select the best one"""
        results = {}
        
        for name, model in self.models.items():
            # Train the model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'r2': r2
            }
        
        # Select the best model based on R² score
        best_model_name = max(results, key=lambda x: results[x]['r2'])
        self.best_model = results[best_model_name]['model']
        
        return results
    
    def predict_price(self, features):
        """Predict house price for given features"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Scale the features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.best_model.predict(features_scaled)
        
        return prediction[0]
    
    def plot_feature_importance(self):
        """Plot feature importance for the best model"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        plt.figure(figsize=(10, 6))
        
        if hasattr(self.best_model, 'feature_importances_'):
            # For tree-based models
            importance = self.best_model.feature_importances_
        else:
            # For linear regression
            importance = np.abs(self.best_model.coef_)
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Plot
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_price_distribution(self):
        """Plot the distribution of house prices"""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['PRICE'], kde=True)
        plt.title('Distribution of House Prices')
        plt.xlabel('Price (in $100,000s)')
        plt.ylabel('Count')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix of features"""
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        return plt.gcf() 