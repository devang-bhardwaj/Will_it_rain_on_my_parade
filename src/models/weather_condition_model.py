"""
Weather condition probability model implementations for NASA Space Apps Challenge.
This module provides models for calculating the probability of different weather conditions
based on historical data for specific locations and times of the year.
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class WeatherConditionModel:
    """Base class for weather condition probability models"""
    
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
        self.threshold = None
    
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
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of the condition.
        
        Args:
            X: Features DataFrame.
            
        Returns:
            Array of condition probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")
        
        # Ensure X has the expected columns and order
        X = X[self.feature_columns]
        
        # Return probability of the positive class (class 1)
        return self.model.predict_proba(X)[:, 1]
    
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
                'feature_columns': self.feature_columns,
                'threshold': self.threshold
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
            self.threshold = data.get('threshold')


class VeryHotConditionModel(WeatherConditionModel):
    """Model to predict 'very hot' weather conditions"""
    
    def __init__(self, model_dir: str = None, threshold_celsius: float = 35.0):
        """
        Initialize 'very hot' condition prediction model.
        
        Args:
            model_dir: Directory to save/load model files.
            threshold_celsius: Temperature threshold in celsius to consider as 'very hot'.
        """
        super().__init__(model_dir)
        self.threshold = threshold_celsius
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the 'very hot' condition model.
        
        Args:
            X: Weather features DataFrame.
            y: Temperature Series (in celsius).
            
        Returns:
            Dictionary with training metrics (accuracy, F1 score).
        """
        # Convert temperature to binary hot/not-hot
        y_binary = (y >= self.threshold).astype(int)
        
        # Store feature columns for prediction
        self.feature_columns = list(X.columns)
        
        # Create a pipeline with preprocessing and model
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train the model
        self.model.fit(X, y_binary)
        
        # Calculate metrics
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y_binary, y_pred)
        f1 = f1_score(y_binary, y_pred)
        precision = precision_score(y_binary, y_pred)
        recall = recall_score(y_binary, y_pred)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }


class VeryColdConditionModel(WeatherConditionModel):
    """Model to predict 'very cold' weather conditions"""
    
    def __init__(self, model_dir: str = None, threshold_celsius: float = 0.0):
        """
        Initialize 'very cold' condition prediction model.
        
        Args:
            model_dir: Directory to save/load model files.
            threshold_celsius: Temperature threshold in celsius to consider as 'very cold'.
        """
        super().__init__(model_dir)
        self.threshold = threshold_celsius
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the 'very cold' condition model.
        
        Args:
            X: Weather features DataFrame.
            y: Temperature Series (in celsius).
            
        Returns:
            Dictionary with training metrics (accuracy, F1 score).
        """
        # Convert temperature to binary cold/not-cold
        y_binary = (y <= self.threshold).astype(int)
        
        # Store feature columns for prediction
        self.feature_columns = list(X.columns)
        
        # Create a pipeline with preprocessing and model
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train the model
        self.model.fit(X, y_binary)
        
        # Calculate metrics
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y_binary, y_pred)
        f1 = f1_score(y_binary, y_pred)
        precision = precision_score(y_binary, y_pred)
        recall = recall_score(y_binary, y_pred)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }


class VeryWindyConditionModel(WeatherConditionModel):
    """Model to predict 'very windy' weather conditions"""
    
    def __init__(self, model_dir: str = None, threshold_kph: float = 30.0):
        """
        Initialize 'very windy' condition prediction model.
        
        Args:
            model_dir: Directory to save/load model files.
            threshold_kph: Wind speed threshold in km/h to consider as 'very windy'.
        """
        super().__init__(model_dir)
        self.threshold = threshold_kph
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the 'very windy' condition model.
        
        Args:
            X: Weather features DataFrame.
            y: Wind speed Series (in km/h).
            
        Returns:
            Dictionary with training metrics (accuracy, F1 score).
        """
        # Convert wind speed to binary windy/not-windy
        y_binary = (y >= self.threshold).astype(int)
        
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
        precision = precision_score(y_binary, y_pred)
        recall = recall_score(y_binary, y_pred)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }


class VeryWetConditionModel(WeatherConditionModel):
    """Model to predict 'very wet' weather conditions"""
    
    def __init__(self, model_dir: str = None, threshold_mm: float = 10.0):
        """
        Initialize 'very wet' condition prediction model.
        
        Args:
            model_dir: Directory to save/load model files.
            threshold_mm: Precipitation threshold in mm to consider as 'very wet'.
        """
        super().__init__(model_dir)
        self.threshold = threshold_mm
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the 'very wet' condition model.
        
        Args:
            X: Weather features DataFrame.
            y: Precipitation amount Series (in mm).
            
        Returns:
            Dictionary with training metrics (accuracy, F1 score).
        """
        # Convert precipitation to binary wet/not-wet
        y_binary = (y >= self.threshold).astype(int)
        
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
        precision = precision_score(y_binary, y_pred)
        recall = recall_score(y_binary, y_pred)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }


class VeryUncomfortableConditionModel(WeatherConditionModel):
    """Model to predict 'very uncomfortable' weather conditions using heat index and other factors"""
    
    def __init__(self, model_dir: str = None):
        """
        Initialize 'very uncomfortable' condition prediction model.
        This model uses multiple weather factors to determine discomfort levels.
        """
        super().__init__(model_dir)
        self.threshold = None  # Will be learned from the data
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the 'very uncomfortable' condition model.
        
        Args:
            X: Weather features DataFrame.
            y: Discomfort index Series or binary uncomfortable/comfortable labels.
            
        Returns:
            Dictionary with training metrics (accuracy, F1 score).
        """
        # If y is not binary, convert it based on a threshold
        if not set(y.unique()).issubset({0, 1}):
            # Calculate a suitable threshold (e.g., top 10% values are "uncomfortable")
            self.threshold = np.percentile(y, 90)
            y_binary = (y >= self.threshold).astype(int)
        else:
            y_binary = y
            
        # Store feature columns for prediction
        self.feature_columns = list(X.columns)
        
        # Create a pipeline with preprocessing and model
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train the model
        self.model.fit(X, y_binary)
        
        # Calculate metrics
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y_binary, y_pred)
        f1 = f1_score(y_binary, y_pred)
        precision = precision_score(y_binary, y_pred)
        recall = recall_score(y_binary, y_pred)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }