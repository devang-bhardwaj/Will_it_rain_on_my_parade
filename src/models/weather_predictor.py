"""
Will It Rain On My Parade? - Weather Prediction Model
NASA Space Apps Challenge Entry

Predicts weather conditions for outdoor events focusing on:
- Very Hot conditions
- Very Cold conditions  
- Very Windy conditions
- Very Wet conditions
- Very Uncomfortable conditions
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WeatherPredictor:
    """
    Main weather prediction model for outdoor events
    Predicts NASA Space Apps Challenge conditions
    """
    
    def __init__(self, db_path: str = "data/weather_database.db"):
        self.db_path = db_path
        self.models = {}
        self.feature_columns = [
            'temperature_avg', 'temperature_max', 'temperature_min',
            'precipitation', 'wind_speed', 'humidity', 'heat_index',
            'month', 'day_of_year', 'latitude', 'longitude'
        ]
        self.target_conditions = [
            'is_very_hot', 'is_very_cold', 'is_very_windy', 
            'is_very_wet', 'is_uncomfortable'
        ]
    
    def load_data(self) -> pd.DataFrame:
        """Load weather data from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT 
            w.*,
            c.latitude,
            c.longitude,
            c.name as city_name,
            c.state
        FROM weather_data w
        JOIN cities c ON w.city_id = c.id
        ORDER BY c.name, w.date
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Add temporal features
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        
        logging.info(f"Loaded {len(df)} weather records from database")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare features for model training"""
        # Features for prediction
        X = df[self.feature_columns].values
        
        # Target conditions
        y = {}
        for condition in self.target_conditions:
            y[condition] = df[condition].values
        
        return X, y
    
    def train_models(self) -> Dict[str, float]:
        """Train prediction models for each weather condition"""
        logging.info("🚀 TRAINING WEATHER PREDICTION MODELS")
        logging.info("=" * 60)
        
        # Load and prepare data
        df = self.load_data()
        if len(df) == 0:
            raise ValueError("No data found in database. Run data collection first.")
        
        X, y = self.prepare_features(df)
        
        # Train separate model for each condition
        accuracies = {}
        
        for condition in self.target_conditions:
            logging.info(f"Training model for: {condition}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y[condition], test_size=0.2, random_state=42, stratify=y[condition]
            )
            
            # Train Random Forest model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies[condition] = accuracy
            
            # Store model
            self.models[condition] = model
            
            logging.info(f"  Accuracy: {accuracy:.3f}")
            
            # Print classification report for positive cases
            positive_cases = sum(y_test)
            logging.info(f"  Positive cases in test set: {positive_cases}/{len(y_test)}")
        
        logging.info("=" * 60)
        logging.info("✅ MODEL TRAINING COMPLETE")
        
        return accuracies
    
    def save_models(self, model_dir: str = "models"):
        """Save trained models to disk"""
        os.makedirs(model_dir, exist_ok=True)
        
        for condition, model in self.models.items():
            model_path = os.path.join(model_dir, f"{condition}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        logging.info(f"Models saved to {model_dir}/")
    
    def load_models(self, model_dir: str = "models"):
        """Load trained models from disk"""
        for condition in self.target_conditions:
            model_path = os.path.join(model_dir, f"{condition}_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[condition] = pickle.load(f)
        
        logging.info(f"Models loaded from {model_dir}/")
    
    def predict_conditions(self, features: Dict) -> Dict[str, float]:
        """
        Predict weather conditions for outdoor event planning
        
        Args:
            features: Dictionary with weather features
                - temperature_avg, temperature_max, temperature_min (°C)
                - precipitation (mm)
                - wind_speed (m/s)
                - humidity (%)
                - heat_index (°C)
                - month (1-12)
                - day_of_year (1-365)
                - latitude, longitude
        
        Returns:
            Dictionary with probability of each condition
        """
        if not self.models:
            raise ValueError("Models not trained or loaded. Train models first.")
        
        # Prepare feature vector
        feature_vector = np.array([[
            features['temperature_avg'],
            features['temperature_max'],
            features['temperature_min'],
            features['precipitation'],
            features['wind_speed'],
            features['humidity'],
            features['heat_index'],
            features['month'],
            features['day_of_year'],
            features['latitude'],
            features['longitude']
        ]])
        
        # Predict probabilities for each condition
        predictions = {}
        for condition in self.target_conditions:
            if condition in self.models:
                prob = self.models[condition].predict_proba(feature_vector)[0][1]  # Probability of positive class
                predictions[condition] = prob
        
        return predictions
    
    def get_event_recommendation(self, predictions: Dict[str, float], threshold: float = 0.3) -> Dict:
        """
        Get outdoor event recommendation based on predictions
        
        Args:
            predictions: Dictionary of condition probabilities
            threshold: Threshold for considering a condition likely
        
        Returns:
            Dictionary with recommendation and reasoning
        """
        high_risk_conditions = []
        moderate_risk_conditions = []
        
        condition_names = {
            'is_very_hot': 'Very Hot (35°C+)',
            'is_very_cold': 'Very Cold (-5°C or lower)',
            'is_very_windy': 'Very Windy (8+ m/s)',
            'is_very_wet': 'Heavy Rain (10+ mm)',
            'is_uncomfortable': 'Uncomfortable Conditions'
        }
        
        for condition, prob in predictions.items():
            if prob > 0.5:  # High probability
                high_risk_conditions.append(f"{condition_names[condition]} ({prob:.1%})")
            elif prob > threshold:  # Moderate probability
                moderate_risk_conditions.append(f"{condition_names[condition]} ({prob:.1%})")
        
        # Overall recommendation
        if high_risk_conditions:
            recommendation = "❌ NOT RECOMMENDED"
            risk_level = "HIGH"
        elif moderate_risk_conditions:
            recommendation = "⚠️ PROCEED WITH CAUTION"
            risk_level = "MODERATE"
        else:
            recommendation = "✅ GOOD CONDITIONS"
            risk_level = "LOW"
        
        return {
            'recommendation': recommendation,
            'risk_level': risk_level,
            'high_risk_conditions': high_risk_conditions,
            'moderate_risk_conditions': moderate_risk_conditions,
            'all_predictions': predictions
        }

    def analyze_historical_patterns(self) -> Dict:
        """Analyze historical weather patterns for insights"""
        df = self.load_data()
        
        analysis = {
            'total_records': len(df),
            'date_range': f"{df['date'].min().date()} to {df['date'].max().date()}",
            'cities_count': df['city_name'].nunique(),
            'condition_frequencies': {}
        }
        
        for condition in self.target_conditions:
            freq = df[condition].mean()
            analysis['condition_frequencies'][condition] = f"{freq:.1%}"
        
        return analysis

def main():
    """Main execution - train models and demonstrate prediction"""
    predictor = WeatherPredictor()
    
    print("🌦️ Will It Rain On My Parade? - Weather Prediction Model")
    print("=" * 70)
    print("NASA Space Apps Challenge - Outdoor Event Weather Prediction")
    print("=" * 70)
    
    try:
        # Analyze current data
        analysis = predictor.analyze_historical_patterns()
        print(f"📊 Data Analysis:")
        print(f"  Records: {analysis['total_records']}")
        print(f"  Date Range: {analysis['date_range']}")
        print(f"  Cities: {analysis['cities_count']}")
        print(f"  Condition Frequencies:")
        for condition, freq in analysis['condition_frequencies'].items():
            print(f"    {condition}: {freq}")
        
        print("\n🚀 Training prediction models...")
        accuracies = predictor.train_models()
        
        print(f"\n📊 Model Accuracies:")
        for condition, accuracy in accuracies.items():
            print(f"  {condition}: {accuracy:.1%}")
        
        # Save models
        predictor.save_models()
        
        print("\n🎯 Example Prediction:")
        # Example: Summer day in Phoenix
        example_features = {
            'temperature_avg': 30,
            'temperature_max': 38,
            'temperature_min': 22,
            'precipitation': 0,
            'wind_speed': 3,
            'humidity': 40,
            'heat_index': 32,
            'month': 7,  # July
            'day_of_year': 200,
            'latitude': 33.4484,  # Phoenix
            'longitude': -112.0740
        }
        
        predictions = predictor.predict_conditions(example_features)
        recommendation = predictor.get_event_recommendation(predictions)
        
        print(f"Location: Phoenix, AZ (July)")
        print(f"Temperature: {example_features['temperature_max']}°C max, {example_features['temperature_min']}°C min")
        print(f"Recommendation: {recommendation['recommendation']}")
        print(f"Risk Level: {recommendation['risk_level']}")
        
        if recommendation['high_risk_conditions']:
            print(f"High Risk: {', '.join(recommendation['high_risk_conditions'])}")
        if recommendation['moderate_risk_conditions']:
            print(f"Moderate Risk: {', '.join(recommendation['moderate_risk_conditions'])}")
        
        print("\n✅ Weather prediction system ready!")
        print("Use predict_conditions() to forecast weather for outdoor events")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Run the data collector first to generate weather data")

if __name__ == "__main__":
    main()