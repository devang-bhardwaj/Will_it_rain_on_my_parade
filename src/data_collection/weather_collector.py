"""
Will It Rain On My Parade? - NASA Space Apps Challenge
Simple, focused weather data collector for outdoor event prediction
"""

import sqlite3
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import csv
import random
import math
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WeatherCollector:
    """
    Streamlined weather data collector for NASA Space Apps Challenge
    Focus: Predict weather conditions for outdoor events
    """
    
    def __init__(self, db_path: str = "data/weather_database.db"):
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WeatherCollector-NASA-Challenge/1.0'
        })
        
    def init_database(self):
        """Initialize clean database structure"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Cities table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cities (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            state TEXT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            UNIQUE(name, state)
        )
        ''')
        
        # Weather data table - focused on outdoor event conditions
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_data (
            id INTEGER PRIMARY KEY,
            city_id INTEGER,
            date DATE,
            temperature_avg REAL,      -- Average temperature (°C)
            temperature_max REAL,      -- Maximum temperature (°C)
            temperature_min REAL,      -- Minimum temperature (°C)
            precipitation REAL,        -- Precipitation (mm)
            wind_speed REAL,          -- Wind speed (m/s)
            humidity REAL,            -- Relative humidity (%)
            heat_index REAL,          -- Heat index for comfort
            is_very_hot INTEGER,      -- 1 if conditions are very hot
            is_very_cold INTEGER,     -- 1 if conditions are very cold
            is_very_windy INTEGER,    -- 1 if conditions are very windy
            is_very_wet INTEGER,      -- 1 if conditions are very wet
            is_uncomfortable INTEGER, -- 1 if conditions are uncomfortable
            data_source TEXT,         -- Source of the data
            FOREIGN KEY (city_id) REFERENCES cities (id),
            UNIQUE(city_id, date)
        )
        ''')
        
        conn.commit()
        conn.close()
        logging.info("Database initialized successfully")
    
    def load_us_cities(self, csv_file: str = "cities.csv") -> List[Dict]:
        """Load US cities from CSV file"""
        cities = []
        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    cities.append({
                        'name': row['City'],
                        'state': row['State'],
                        'latitude': float(row['Latitude']),
                        'longitude': float(row['Longitude'])
                    })
            logging.info(f"Loaded {len(cities)} cities from {csv_file}")
        except Exception as e:
            logging.error(f"Error loading cities: {e}")
        return cities
    
    def insert_cities(self, cities: List[Dict]):
        """Insert cities into database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for city in cities:
            try:
                cursor.execute('''
                INSERT OR IGNORE INTO cities (name, state, latitude, longitude)
                VALUES (?, ?, ?, ?)
                ''', (city['name'], city['state'], city['latitude'], city['longitude']))
            except Exception as e:
                logging.error(f"Error inserting city {city['name']}: {e}")
        
        conn.commit()
        conn.close()
        logging.info(f"Inserted {len(cities)} cities into database")
    
    def generate_synthetic_weather(self, city: Dict, start_date: datetime, days: int = 365) -> List[Dict]:
        """
        Generate realistic synthetic weather data
        Based on US climate patterns and geographic location
        """
        weather_data = []
        lat = city['latitude']
        
        # Climate patterns based on latitude and common US weather
        base_temp = 15  # Base temperature in Celsius
        if lat > 45:  # Northern states
            base_temp = 5
        elif lat > 35:  # Middle states  
            base_temp = 15
        else:  # Southern states
            base_temp = 25
            
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Seasonal variation
            day_of_year = current_date.timetuple().tm_yday
            seasonal_factor = math.sin(2 * math.pi * (day_of_year - 80) / 365)
            
            # Temperature with seasonal variation and random fluctuation
            temp_avg = base_temp + (seasonal_factor * 15) + random.gauss(0, 5)
            temp_max = temp_avg + random.uniform(3, 8)
            temp_min = temp_avg - random.uniform(3, 8)
            
            # Precipitation (mm/day)
            precip = max(0, np.random.exponential(2) if random.random() < 0.3 else 0)
            
            # Wind speed (m/s)
            wind = max(0.5, np.random.gamma(2, 1.5))
            
            # Humidity (%)
            humidity = max(20, min(100, random.gauss(65, 20)))
            
            # Heat index calculation (simplified)
            heat_index = temp_avg + (humidity / 100) * 5
            
            # NASA Challenge Conditions
            is_very_hot = 1 if temp_max > 35 else 0  # Very hot
            is_very_cold = 1 if temp_min < -5 else 0  # Very cold
            is_very_windy = 1 if wind > 8 else 0  # Very windy
            is_very_wet = 1 if precip > 10 else 0  # Very wet
            is_uncomfortable = 1 if heat_index > 30 or temp_min < 0 or wind > 10 else 0
            
            weather_data.append({
                'date': current_date.date(),
                'temperature_avg': round(temp_avg, 2),
                'temperature_max': round(temp_max, 2),
                'temperature_min': round(temp_min, 2),
                'precipitation': round(precip, 2),
                'wind_speed': round(wind, 2),
                'humidity': round(humidity, 2),
                'heat_index': round(heat_index, 2),
                'is_very_hot': is_very_hot,
                'is_very_cold': is_very_cold,
                'is_very_windy': is_very_windy,
                'is_very_wet': is_very_wet,
                'is_uncomfortable': is_uncomfortable,
                'data_source': 'synthetic'
            })
        
        return weather_data
    
    def insert_weather_data(self, city_id: int, weather_data: List[Dict]):
        """Insert weather data into database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for data in weather_data:
            try:
                cursor.execute('''
                INSERT OR REPLACE INTO weather_data 
                (city_id, date, temperature_avg, temperature_max, temperature_min,
                 precipitation, wind_speed, humidity, heat_index,
                 is_very_hot, is_very_cold, is_very_windy, is_very_wet, is_uncomfortable,
                 data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    city_id, data['date'], data['temperature_avg'],
                    data['temperature_max'], data['temperature_min'],
                    data['precipitation'], data['wind_speed'], data['humidity'],
                    data['heat_index'], data['is_very_hot'], data['is_very_cold'],
                    data['is_very_windy'], data['is_very_wet'], data['is_uncomfortable'],
                    data['data_source']
                ))
            except Exception as e:
                logging.error(f"Error inserting weather data: {e}")
        
        conn.commit()
        conn.close()
    
    def get_city_id(self, city_name: str, state: str) -> Optional[int]:
        """Get city ID from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM cities WHERE name = ? AND state = ?', (city_name, state))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def collect_for_cities(self, num_cities: int = 50, years: int = 5):
        """
        Main collection function
        Generate synthetic weather data for specified number of cities
        """
        logging.info("🌦️ STARTING WEATHER DATA COLLECTION")
        logging.info("=" * 60)
        
        # Initialize database
        self.init_database()
        
        # Load and insert cities
        cities = self.load_us_cities()[:num_cities]
        self.insert_cities(cities)
        
        # Generate weather data
        start_date = datetime(2020, 1, 1)  # Start from 2020
        total_records = 0
        
        for i, city in enumerate(cities, 1):
            logging.info(f"Processing {city['name']}, {city['state']} ({i}/{len(cities)})")
            
            city_id = self.get_city_id(city['name'], city['state'])
            if not city_id:
                logging.error(f"Could not find city ID for {city['name']}")
                continue
            
            # Generate weather data
            weather_data = self.generate_synthetic_weather(city, start_date, days=years*365)
            self.insert_weather_data(city_id, weather_data)
            
            total_records += len(weather_data)
            logging.info(f"Generated {len(weather_data)} records for {city['name']}")
        
        logging.info("=" * 60)
        logging.info("COLLECTION COMPLETE!")
        logging.info(f"Cities processed: {len(cities)}")
        logging.info(f"Total weather records: {total_records}")
        logging.info(f"Database: {self.db_path}")
        
        return total_records

def main():
    """Main execution function"""
    collector = WeatherCollector("data/weather_database.db")
    
    print("🌦️ Will It Rain On My Parade? - Data Collector")
    print("=" * 60)
    print("NASA Space Apps Challenge Weather Prediction System")
    print("Generating synthetic weather data for outdoor event prediction")
    print("=" * 60)
    
    # Collect data for 25 major cities over 5 years
    total_records = collector.collect_for_cities(num_cities=25, years=5)
    
    print(f"\n✅ SUCCESS: Generated {total_records} weather records")
    print("Ready for weather prediction model development!")

if __name__ == "__main__":
    main()