"""
Data Preprocessing and Model Training Module
Handles data loading, preprocessing, and model training
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

class ModelTrainer:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_data(self, dataset_type='diabetes'):
        """Load dataset"""
        print(f"Loading {dataset_type} dataset...")
        
        if dataset_type == 'diabetes':
            data = load_diabetes()
            X = data.data
            y = data.target
            self.feature_names = data.feature_names
        elif dataset_type == 'california_housing':
            data = fetch_california_housing()
            X = data.data
            y = data.target
            self.feature_names = data.feature_names
        else:
            raise ValueError(f"Unknown dataset: {dataset_type}")
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y
    
    def preprocess_data(self, X, y):
        """Split and scale the data"""
        print("Preprocessing data...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        """Train the Random Forest model"""
        print("Training Random Forest model...")
        
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed!")
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate the model on test set"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        y_pred = self.model.predict(self.X_test)
        
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print("="*50 + "\n")
        
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def save_artifacts(self, model_path='model.pkl', scaler_path='scaler.pkl'):
        """Save model and scaler"""
        print(f"Saving model to {model_path}...")
        joblib.dump(self.model, model_path)
        
        print(f"Saving scaler to {scaler_path}...")
        joblib.dump(self.scaler, scaler_path)
        
        print("Artifacts saved successfully!")
    
    def predict(self, X):
        """Make predictions"""
        if self.scaler is None or self.model is None:
            raise ValueError("Model or scaler not trained. Please train first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def train_pipeline(self, dataset_type='diabetes', save=True):
        """Complete training pipeline"""
        X, y = self.load_data(dataset_type)
        self.preprocess_data(X, y)
        self.train_model()
        metrics = self.evaluate_model()
        
        if save:
            self.save_artifacts()
        
        return metrics

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_pipeline(dataset_type='diabetes', save=True)
