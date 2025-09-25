"""
Data processing utilities for NASA Earth observation weather data
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime

class WeatherDataProcessor:
    """Class for processing and preparing historical weather data for condition analysis"""
    
    def __init__(self, raw_data_dir: str = None, processed_data_dir: str = None):
        """
        Initialize the data processor.
        
        Args:
            raw_data_dir: Directory containing raw data files.
            processed_data_dir: Directory to save processed data files.
        """
        base_dir = Path(__file__).parent.parent.parent
        
        self.raw_data_dir = raw_data_dir or str(base_dir / "data" / "raw")
        self.processed_data_dir = processed_data_dir or str(base_dir / "data" / "processed")
        
        os.makedirs(self.processed_data_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def load_nasa_data(self, dataset: str = None) -> pd.DataFrame:
        """
        Load raw NASA Earth observation data.
        
        Args:
            dataset: NASA dataset name (e.g., 'MODIS', 'GPM', 'MERRA-2', 'POWER').
                   If None, load all available NASA data.
            
        Returns:
            DataFrame with combined data.
        """
        data_frames = []
        
        nasa_dir = os.path.join(self.raw_data_dir, 'nasa')
        if not os.path.exists(nasa_dir):
            os.makedirs(nasa_dir, exist_ok=True)
            self.logger.warning(f"NASA data directory not found, created: {nasa_dir}")
            return pd.DataFrame()  # Return empty dataframe if no data exists yet
        
        if dataset:
            dataset_dir = os.path.join(nasa_dir, dataset)
            if not os.path.exists(dataset_dir):
                self.logger.warning(f"Dataset directory not found: {dataset_dir}")
                return pd.DataFrame()
            directories = [dataset_dir]
        else:
            # Get all subdirectories in the NASA data directory
            directories = [
                d for d in os.listdir(nasa_dir) 
                if os.path.isdir(os.path.join(nasa_dir, d))
            ]
            directories = [os.path.join(nasa_dir, d) for d in directories]
        
        # Load all data files in the specified directories
        for directory in directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(('.csv', '.nc', '.hdf', '.json')):
                        file_path = os.path.join(root, file)
                        try:
                            # Different handling based on file type
                            if file.endswith('.csv'):
                                df = pd.read_csv(file_path)
                            elif file.endswith('.json'):
                                df = pd.read_json(file_path)
                            elif file.endswith('.nc') or file.endswith('.hdf'):
                                # For netCDF or HDF5 files, we'll need xarray or h5py
                                # This is a placeholder - we'll need to implement specific parsing
                                self.logger.info(f"Skipping {file_path} - requires specialized parsing")
                                continue
                            
                            # Add dataset information
                            dataset_name = os.path.basename(directory)
                            df['dataset'] = dataset_name
                            df['filename'] = file
                            
                            data_frames.append(df)
                            self.logger.info(f"Loaded {file_path}")
                        except Exception as e:
                            self.logger.error(f"Error loading {file_path}: {e}")
        
        if not data_frames:
            self.logger.warning("No data files found")
            return pd.DataFrame()
        
        # Combine all data frames
        try:
            combined_data = pd.concat(data_frames, ignore_index=True)
            self.logger.info(f"Combined {len(data_frames)} data files")
            return combined_data
        except Exception as e:
            self.logger.error(f"Error combining data frames: {e}")
            raise
    
    def preprocess_temperature_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess temperature data for analysis.
        
        Args:
            data: Raw temperature data DataFrame.
            
        Returns:
            Preprocessed temperature data.
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure datetime column is properly formatted
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'])
        
        # Extract day of year for seasonal analysis
        if 'date' in df.columns:
            df['day_of_year'] = df['date'].dt.dayofyear
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
        
        # Convert temperature to celsius if needed
        if 'temperature_kelvin' in df.columns:
            df['temperature_celsius'] = df['temperature_kelvin'] - 273.15
        elif 'temperature_fahrenheit' in df.columns:
            df['temperature_celsius'] = (df['temperature_fahrenheit'] - 32) * 5/9
            
        return df
    
    def preprocess_precipitation_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess precipitation data for analysis.
        
        Args:
            data: Raw precipitation data DataFrame.
            
        Returns:
            Preprocessed precipitation data.
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure datetime column is properly formatted
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'])
        
        # Extract day of year for seasonal analysis
        if 'date' in df.columns:
            df['day_of_year'] = df['date'].dt.dayofyear
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
        
        # Convert precipitation to mm if needed
        if 'precipitation_inches' in df.columns:
            df['precipitation_mm'] = df['precipitation_inches'] * 25.4
            
        return df
    
    def preprocess_wind_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess wind data for analysis.
        
        Args:
            data: Raw wind data DataFrame.
            
        Returns:
            Preprocessed wind data.
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure datetime column is properly formatted
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'])
        
        # Extract day of year for seasonal analysis
        if 'date' in df.columns:
            df['day_of_year'] = df['date'].dt.dayofyear
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
        
        # Convert wind speed to km/h if needed
        if 'wind_speed_mph' in df.columns:
            df['wind_speed_kph'] = df['wind_speed_mph'] * 1.60934
        elif 'wind_speed_ms' in df.columns:
            df['wind_speed_kph'] = df['wind_speed_ms'] * 3.6
            
        return df
    
    def calculate_discomfort_index(self, temp_celsius: float, humidity_percent: float) -> float:
        """
        Calculate a discomfort index based on temperature and humidity.
        
        Args:
            temp_celsius: Temperature in Celsius
            humidity_percent: Relative humidity percentage
            
        Returns:
            Discomfort index value
        """
        # Heat index calculation (simplified)
        if temp_celsius < 20:  # Below this temperature, heat index is less relevant
            return 0
            
        # Simple heat index formula
        hi = 0.5 * (temp_celsius + 61.0 + ((temp_celsius - 68.0) * 1.2) + (humidity_percent * 0.094))
        
        # Convert to common scale (0-100)
        # 0: Comfortable, 100: Extremely uncomfortable
        if hi < 70:  # Comfortable
            return 0
        elif hi >= 90:  # Very uncomfortable
            return 100
        else:  # Linear scaling between comfortable and very uncomfortable
            return (hi - 70) * 5  # Scale from 0-100
    
    def aggregate_by_location_and_day(self, data: pd.DataFrame, 
                                      location_cols: List[str] = ['latitude', 'longitude'],
                                      time_col: str = 'day_of_year') -> pd.DataFrame:
        """
        Aggregate data by location and day of year to produce historical statistics.
        
        Args:
            data: Preprocessed weather data
            location_cols: Columns identifying the location
            time_col: Column for time grouping (usually day_of_year for seasonal patterns)
            
        Returns:
            Aggregated data with statistics by location and day
        """
        if not all(col in data.columns for col in location_cols + [time_col]):
            missing_cols = [col for col in location_cols + [time_col] if col not in data.columns]
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Group by location and day of year
        groupby_cols = location_cols + [time_col]
        aggregated = data.groupby(groupby_cols).agg({
            'temperature_celsius': ['mean', 'min', 'max', 'std', 'count'] if 'temperature_celsius' in data.columns else 'count',
            'precipitation_mm': ['mean', 'max', 'sum', 'count'] if 'precipitation_mm' in data.columns else 'count',
            'wind_speed_kph': ['mean', 'max', 'std', 'count'] if 'wind_speed_kph' in data.columns else 'count',
            'humidity_percent': ['mean', 'min', 'max'] if 'humidity_percent' in data.columns else 'count'
        }).reset_index()
        
        # Flatten multi-level column names
        aggregated.columns = ['_'.join(col).strip('_') for col in aggregated.columns.values]
        
        return aggregated
    
    def calculate_condition_probabilities(self, 
                                         aggregated_data: pd.DataFrame,
                                         temp_threshold_hot: float = 35.0,
                                         temp_threshold_cold: float = 0.0,
                                         wind_threshold: float = 30.0,
                                         precip_threshold: float = 10.0,
                                         discomfort_threshold: float = 70.0) -> pd.DataFrame:
        """
        Calculate probabilities of different weather conditions.
        
        Args:
            aggregated_data: Aggregated data by location and day of year
            temp_threshold_hot: Temperature threshold (celsius) for 'very hot'
            temp_threshold_cold: Temperature threshold (celsius) for 'very cold'
            wind_threshold: Wind speed threshold (km/h) for 'very windy'
            precip_threshold: Precipitation threshold (mm) for 'very wet'
            discomfort_threshold: Discomfort index threshold for 'very uncomfortable'
            
        Returns:
            DataFrame with condition probabilities
        """
        df = aggregated_data.copy()
        
        # Calculate probabilities based on historical data
        if 'temperature_celsius_max' in df.columns:
            df['prob_very_hot'] = df['temperature_celsius_max'].apply(
                lambda x: 1 if x >= temp_threshold_hot else 0)
            
        if 'temperature_celsius_min' in df.columns:
            df['prob_very_cold'] = df['temperature_celsius_min'].apply(
                lambda x: 1 if x <= temp_threshold_cold else 0)
            
        if 'wind_speed_kph_max' in df.columns:
            df['prob_very_windy'] = df['wind_speed_kph_max'].apply(
                lambda x: 1 if x >= wind_threshold else 0)
            
        if 'precipitation_mm_sum' in df.columns:
            df['prob_very_wet'] = df['precipitation_mm_sum'].apply(
                lambda x: 1 if x >= precip_threshold else 0)
            
        # Calculate discomfort index if we have both temperature and humidity
        if 'temperature_celsius_mean' in df.columns and 'humidity_percent_mean' in df.columns:
            df['discomfort_index'] = df.apply(
                lambda row: self.calculate_discomfort_index(
                    row['temperature_celsius_mean'], 
                    row['humidity_percent_mean']
                ), 
                axis=1
            )
            
            df['prob_very_uncomfortable'] = df['discomfort_index'].apply(
                lambda x: 1 if x >= discomfort_threshold else 0)
        
        return df
    
    def process_and_save(self) -> str:
        """
        Process all data and save the results.
        
        Returns:
            Path to processed data file.
        """
        # Load NASA data
        nasa_data = self.load_nasa_data()
        
        if nasa_data.empty:
            self.logger.warning("No NASA data available for processing")
            return ""
        
        # Process based on data types
        temp_data = self.preprocess_temperature_data(nasa_data)
        precip_data = self.preprocess_precipitation_data(nasa_data)
        wind_data = self.preprocess_wind_data(nasa_data)
        
        # Merge datasets if they have common identifiers
        # This is a simplified approach - may need more sophisticated joining in real scenarios
        
        # Aggregate data
        aggregated_temp = self.aggregate_by_location_and_day(temp_data)
        aggregated_precip = self.aggregate_by_location_and_day(precip_data)
        aggregated_wind = self.aggregate_by_location_and_day(wind_data)
        
        # Calculate condition probabilities
        temp_prob = self.calculate_condition_probabilities(aggregated_temp)
        precip_prob = self.calculate_condition_probabilities(aggregated_precip)
        wind_prob = self.calculate_condition_probabilities(aggregated_wind)
        
        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.processed_data_dir, f"condition_probabilities_{timestamp}.csv")
        
        # In a real scenario, we would merge these probability datasets
        # Here we'll just save the temperature one as an example
        temp_prob.to_csv(output_path, index=False)
        self.logger.info(f"Saved processed data to {output_path}")
        
        return output_path