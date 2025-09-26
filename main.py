"""
Will It Rain On My Parade? - Main Launcher
NASA Space Apps Challenge Entry

Complete weather prediction system for outdoor events
"""

import os
import sys
from datetime import datetime

# Add src to path
sys.path.append('src')

from data_collection.weather_collector import WeatherCollector
from models.weather_predictor import WeatherPredictor

def main():
    """Main launcher for the weather prediction system"""
    print("🌦️ WILL IT RAIN ON MY PARADE?")
    print("=" * 60)
    print("NASA Space Apps Challenge - Weather Prediction System")
    print("Predicting weather conditions for outdoor events")
    print("=" * 60)
    
    while True:
        print("\nWhat would you like to do?")
        print("1. 📊 Collect Weather Data")
        print("2. 🚀 Train Prediction Models")
        print("3. 🎯 Make Weather Prediction")
        print("4. 🌐 Start API Server")
        print("5. 📈 View Data Analysis")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == '1':
            collect_weather_data()
        elif choice == '2':
            train_models()
        elif choice == '3':
            make_prediction()
        elif choice == '4':
            start_api_server()
        elif choice == '5':
            view_analysis()
        elif choice == '0':
            print("👋 Goodbye! Thanks for using Will It Rain On My Parade!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

def collect_weather_data():
    """Collect weather data"""
    print("\n📊 WEATHER DATA COLLECTION")
    print("-" * 40)
    
    try:
        num_cities = int(input("Number of cities to collect (default 25): ") or "25")
        years = int(input("Number of years of data (default 5): ") or "5")
        
        collector = WeatherCollector()
        total_records = collector.collect_for_cities(num_cities=num_cities, years=years)
        
        print(f"\n✅ SUCCESS!")
        print(f"Generated {total_records:,} weather records")
        print("Ready for model training!")
        
    except Exception as e:
        print(f"❌ Error collecting data: {e}")

def train_models():
    """Train prediction models"""
    print("\n🚀 MODEL TRAINING")
    print("-" * 40)
    
    try:
        predictor = WeatherPredictor()
        
        # Check if data exists
        analysis = predictor.analyze_historical_patterns()
        if analysis['total_records'] == 0:
            print("❌ No weather data found!")
            print("Please collect weather data first (option 1)")
            return
        
        print(f"📊 Found {analysis['total_records']:,} weather records")
        print(f"📅 Date range: {analysis['date_range']}")
        print(f"🏙️ Cities: {analysis['cities_count']}")
        
        print("\nTraining models...")
        accuracies = predictor.train_models()
        predictor.save_models()
        
        print(f"\n✅ TRAINING COMPLETE!")
        print("Model Accuracies:")
        for condition, accuracy in accuracies.items():
            print(f"  {condition}: {accuracy:.1%}")
        
    except Exception as e:
        print(f"❌ Error training models: {e}")

def make_prediction():
    """Make weather prediction"""
    print("\n🎯 WEATHER PREDICTION")
    print("-" * 40)
    
    try:
        predictor = WeatherPredictor()
        predictor.load_models()
        
        if not predictor.models:
            print("❌ No trained models found!")
            print("Please train models first (option 2)")
            return
        
        print("Enter weather conditions for prediction:")
        
        # Get input from user
        temp_avg = float(input("Average temperature (°C): "))
        temp_max = float(input("Maximum temperature (°C): "))
        temp_min = float(input("Minimum temperature (°C): "))
        precipitation = float(input("Precipitation (mm): "))
        wind_speed = float(input("Wind speed (m/s): "))
        humidity = float(input("Humidity (%): "))
        month = int(input("Month (1-12): "))
        day_of_year = int(input("Day of year (1-365): "))
        latitude = float(input("Latitude: "))
        longitude = float(input("Longitude: "))
        
        # Calculate heat index
        heat_index = temp_avg + (humidity / 100) * 5
        
        features = {
            'temperature_avg': temp_avg,
            'temperature_max': temp_max,
            'temperature_min': temp_min,
            'precipitation': precipitation,
            'wind_speed': wind_speed,
            'humidity': humidity,
            'heat_index': heat_index,
            'month': month,
            'day_of_year': day_of_year,
            'latitude': latitude,
            'longitude': longitude
        }
        
        # Make prediction
        predictions = predictor.predict_conditions(features)
        recommendation = predictor.get_event_recommendation(predictions)
        
        print(f"\n🎯 PREDICTION RESULTS")
        print("=" * 40)
        print(f"Recommendation: {recommendation['recommendation']}")
        print(f"Risk Level: {recommendation['risk_level']}")
        
        if recommendation['high_risk_conditions']:
            print(f"\n🚨 High Risk Conditions:")
            for condition in recommendation['high_risk_conditions']:
                print(f"  • {condition}")
        
        if recommendation['moderate_risk_conditions']:
            print(f"\n⚠️ Moderate Risk Conditions:")
            for condition in recommendation['moderate_risk_conditions']:
                print(f"  • {condition}")
        
        print(f"\n📊 Detailed Probabilities:")
        for condition, prob in predictions.items():
            print(f"  {condition}: {prob:.1%}")
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")

def start_api_server():
    """Start the API server"""
    print("\n🌐 STARTING API SERVER")
    print("-" * 40)
    print("The API will start on http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        from api.app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Error starting API server: {e}")

def view_analysis():
    """View data analysis"""
    print("\n📈 DATA ANALYSIS")
    print("-" * 40)
    
    try:
        predictor = WeatherPredictor()
        analysis = predictor.analyze_historical_patterns()
        
        print(f"📊 Dataset Overview:")
        print(f"  Total Records: {analysis['total_records']:,}")
        print(f"  Date Range: {analysis['date_range']}")
        print(f"  Cities: {analysis['cities_count']}")
        
        print(f"\n🌦️ Weather Condition Frequencies:")
        for condition, freq in analysis['condition_frequencies'].items():
            condition_name = condition.replace('is_', '').replace('_', ' ').title()
            print(f"  {condition_name}: {freq}")
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")

if __name__ == "__main__":
    main()