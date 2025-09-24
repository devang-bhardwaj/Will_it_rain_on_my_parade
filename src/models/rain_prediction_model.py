"""
Weather prediction model implementations
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, accuracy_score

class RainPredictionModel:
    """Base class for rain prediction models"""
    
    def __init__(self, model_dir: str = None):
        """
        Initialize the model.
        
        Args:
            model_dir: Directory to save/load model files.
        """
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models"
        )
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.model = None
        self.feature_columns = None
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X: Features DataFrame.
            y: Target Series.
            
        Returns:
            Dictionary with training metrics.
        """
        raise NotImplementedError("Subclasses must implement train()")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features DataFrame.
            
        Returns:
            Array of predictions.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")
        
        # Ensure X has the expected columns
        if not set(self.feature_columns).issubset(set(X.columns)):
            missing_cols = set(self.feature_columns) - set(X.columns)
            raise ValueError(f"Missing columns in input data: {missing_cols}")
        
        # Select only the expected columns in the correct order
        X = X[self.feature_columns]
        
        return self.model.predict(X)
    
    def save(self, filename: str = None) -> str:
        """
        Save the model to disk.
        
        Args:
            filename: Filename to save to. If None, a default name will be used.
            
        Returns:
            Path to the saved model file.
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if filename is None:
            filename = f"{self.__class__.__name__}.pkl"
            
        filepath = os.path.join(self.model_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_columns': self.feature_columns
            }, f)
            
        return filepath
    
    def load(self, filename: str = None) -> None:
        """
        Load a model from disk.
        
        Args:
            filename: Filename to load from. If None, a default name will be used.
        """
        if filename is None:
            filename = f"{self.__class__.__name__}.pkl"
            
        filepath = os.path.join(self.model_dir, filename)
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_columns = data['feature_columns']


class RainfallAmountModel(RainPredictionModel):
    """Model to predict rainfall amount"""
    
    def __init__(self, model_dir: str = None):
        """Initialize rainfall amount prediction model"""
        super().__init__(model_dir)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the rainfall amount model.
        
        Args:
            X: Weather features DataFrame.
            y: Rainfall amount Series (in mm).
            
        Returns:
            Dictionary with training metrics (RMSE, MAE).
        """
        # Store feature columns for prediction
        self.feature_columns = list(X.columns)
        
        # Create a pipeline with preprocessing and model
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Train the model
        self.model.fit(X, y)
        
        # Calculate metrics
        y_pred = self.model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        return {
            'rmse': rmse,
            'mae': mae
        }


class RainOccurrenceModel(RainPredictionModel):
    """Model to predict rain occurrence (binary classification)"""
    
    def __init__(self, model_dir: str = None, threshold_mm: float = 0.2):
        """
        Initialize rain occurrence prediction model.
        
        Args:
            model_dir: Directory to save/load model files.
            threshold_mm: Rainfall threshold in mm to consider as rain (default: 0.2mm).
        """
        super().__init__(model_dir)
        self.threshold_mm = threshold_mm
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the rain occurrence model.
        
        Args:
            X: Weather features DataFrame.
            y: Rainfall amount Series (in mm).
            
        Returns:
            Dictionary with training metrics (accuracy, F1 score).
        """
        # Convert rainfall amount to binary rain/no-rain
        y_binary = (y >= self.threshold_mm).astype(int)
        
        # Store feature columns for prediction
        self.feature_columns = list(X.columns)
        
        # Create a pipeline with preprocessing and model
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train the model
        self.model.fit(X, y_binary)
        
        # Calculate metrics
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y_binary, y_pred)
        f1 = f1_score(y_binary, y_pred)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1
        }
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of rain.
        
        Args:
            X: Features DataFrame.
            
        Returns:
            Array of rain probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")
        
        # Ensure X has the expected columns and order
        X = X[self.feature_columns]
        
        # Return probability of rain (class 1)
        return self.model.predict_proba(X)[:, 1]