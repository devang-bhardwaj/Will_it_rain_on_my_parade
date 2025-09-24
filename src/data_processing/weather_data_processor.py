"""
Data processing utilities for weather data
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import logging

class WeatherDataProcessor:
    """Class for processing and preparing weather data for model training"""
    
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
    
    def load_data(self, source: str = None) -> pd.DataFrame:
        """
        Load raw data from files.
        
        Args:
            source: Data source name (e.g., 'nasa', 'noaa', 'openweathermap').
                   If None, load all available data.
            
        Returns:
            DataFrame with combined data.
        """
        data_frames = []
        
        if source:
            source_dir = os.path.join(self.raw_data_dir, source)
            if not os.path.exists(source_dir):
                raise ValueError(f"Source directory not found: {source_dir}")
            directories = [source_dir]
        else:
            # Get all subdirectories in the raw data directory
            directories = [
                d for d in os.listdir(self.raw_data_dir) 
                if os.path.isdir(os.path.join(self.raw_data_dir, d))
            ]
            directories = [os.path.join(self.raw_data_dir, d) for d in directories]
        
        # Load all data files in the specified directories
        for directory in directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        try:
                            df = pd.read_csv(file_path)
                            
                            # Add source information
                            source_name = os.path.basename(directory)
                            df['data_source'] = source_name
                            df['filename'] = file
                            
                            data_frames.append(df)
                            self.logger.info(f"Loaded {file_path}")
                        except Exception as e:
                            self.logger.error(f"Error loading {file_path}: {e}")
                    
                    elif file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        try:
                            df = pd.read_json(file_path)
                            
                            # Add source information
                            source_name = os.path.basename(directory)
                            df['data_source'] = source_name
                            df['filename'] = file
                            
                            data_frames.append(df)
                            self.logger.info(f"Loaded {file_path}")
                        except Exception as e:
                            self.logger.error(f"Error loading {file_path}: {e}")
        
        if not data_frames:
            raise ValueError("No data files found")
        
        # Combine all data frames
        try:
            combined_data = pd.concat(data_frames, ignore_index=True)
            self.logger.info(f"Combined {len(data_frames)} data files")
            return combined_data
        except Exception as e:
            self.logger.error(f"Error combining data frames: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling missing values, duplicates, and outliers.
        
        Args:
            df: DataFrame to clean.
            
        Returns:
            Cleaned DataFrame.
        """
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle missing values
        self.logger.info(f"Missing values before cleaning: {df_clean.isna().sum().sum()}")
        
        # For numeric columns, fill missing values with the median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isna().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
        
        # For categorical columns, fill missing values with the mode
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df_clean[col].isna().any():
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
        
        self.logger.info(f"Missing values after cleaning: {df_clean.isna().sum().sum()}")
        
        # Remove duplicates
        n_before = len(df_clean)
        df_clean.drop_duplicates(inplace=True)
        n_after = len(df_clean)
        self.logger.info(f"Removed {n_before - n_after} duplicate rows")
        
        # Handle outliers using IQR method for numeric columns
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            df_clean[col] = np.where(
                df_clean[col] < lower_bound,
                lower_bound,
                np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
            )
        
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for model training.
        
        Args:
            df: DataFrame to process.
            
        Returns:
            DataFrame with engineered features.
        """
        df_featured = df.copy()
        
        # If we have datetime columns, extract useful time features
        datetime_cols = [col for col in df_featured.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        for col in datetime_cols:
            try:
                # Convert to datetime if not already
                df_featured[col] = pd.to_datetime(df_featured[col])
                
                # Extract features
                col_prefix = col.lower().replace('date', '').replace('time', '').strip('_')
                if not col_prefix:
                    col_prefix = 'time'
                
                df_featured[f'{col_prefix}_year'] = df_featured[col].dt.year
                df_featured[f'{col_prefix}_month'] = df_featured[col].dt.month
                df_featured[f'{col_prefix}_day'] = df_featured[col].dt.day
                df_featured[f'{col_prefix}_hour'] = df_featured[col].dt.hour
                df_featured[f'{col_prefix}_dayofweek'] = df_featured[col].dt.dayofweek
                df_featured[f'{col_prefix}_dayofyear'] = df_featured[col].dt.dayofyear
                
                # Create cyclical features for month, day, and hour
                df_featured[f'{col_prefix}_month_sin'] = np.sin(2 * np.pi * df_featured[col].dt.month / 12)
                df_featured[f'{col_prefix}_month_cos'] = np.cos(2 * np.pi * df_featured[col].dt.month / 12)
                
                df_featured[f'{col_prefix}_hour_sin'] = np.sin(2 * np.pi * df_featured[col].dt.hour / 24)
                df_featured[f'{col_prefix}_hour_cos'] = np.cos(2 * np.pi * df_featured[col].dt.hour / 24)
                
                df_featured[f'{col_prefix}_dayofyear_sin'] = np.sin(2 * np.pi * df_featured[col].dt.dayofyear / 365.25)
                df_featured[f'{col_prefix}_dayofyear_cos'] = np.cos(2 * np.pi * df_featured[col].dt.dayofyear / 365.25)
                
            except Exception as e:
                self.logger.warning(f"Could not process datetime column {col}: {e}")
        
        # If we have temperature and humidity, create heat index feature
        if 'temperature' in df_featured.columns and 'humidity' in df_featured.columns:
            df_featured['heat_index'] = self._calculate_heat_index(
                df_featured['temperature'], 
                df_featured['humidity']
            )
        
        # If we have wind speed and direction, create wind components
        if 'wind_speed' in df_featured.columns and 'wind_direction' in df_featured.columns:
            wind_components = self._calculate_wind_components(
                df_featured['wind_speed'],
                df_featured['wind_direction']
            )
            df_featured['wind_u'] = wind_components[0]
            df_featured['wind_v'] = wind_components[1]
        
        return df_featured
    
    def _calculate_heat_index(self, temperature: pd.Series, humidity: pd.Series) -> pd.Series:
        """
        Calculate heat index from temperature and humidity.
        
        Args:
            temperature: Temperature series in Celsius.
            humidity: Relative humidity series (0-100).
            
        Returns:
            Heat index series.
        """
        # Convert to Fahrenheit for the standard heat index formula
        t_f = temperature * 9/5 + 32
        
        # Simple formula for heat index
        hi = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (humidity * 0.094))
        
        # Convert back to Celsius
        return (hi - 32) * 5/9
    
    def _calculate_wind_components(self, 
                                speed: pd.Series, 
                                direction: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate wind components from speed and direction.
        
        Args:
            speed: Wind speed series.
            direction: Wind direction series in degrees (0-360, 0 = North).
            
        Returns:
            Tuple of (u, v) wind components.
        """
        # Convert direction to radians and adjust for meteorological convention
        direction_rad = np.radians(270 - direction)
        
        # Calculate U (east-west) and V (north-south) components
        u = -speed * np.cos(direction_rad)  # Positive = from west, negative = from east
        v = -speed * np.sin(direction_rad)  # Positive = from south, negative = from north
        
        return u, v
    
    def prepare_for_model(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Prepare data for model training by splitting features and target.
        
        Args:
            df: DataFrame to prepare.
            
        Returns:
            Tuple of (features DataFrame, target Series if available)
        """
        # Check if target variable is present
        if 'precipitation' in df.columns:
            target = df['precipitation']
            features = df.drop(['precipitation'], axis=1)
        elif 'rainfall' in df.columns:
            target = df['rainfall']
            features = df.drop(['rainfall'], axis=1)
        else:
            target = None
            features = df
        
        # Drop non-feature columns
        cols_to_drop = [
            'filename', 'id', 'data_source',
            'timestamp', 'datetime', 'date', 'time'
        ]
        
        for col in cols_to_drop:
            if col in features.columns:
                features = features.drop(col, axis=1)
        
        # Drop any remaining datetime columns
        datetime_cols = [col for col in features.columns if pd.api.types.is_datetime64_any_dtype(features[col])]
        if datetime_cols:
            features = features.drop(datetime_cols, axis=1)
        
        # One-hot encode categorical variables
        cat_cols = features.select_dtypes(include=['object']).columns
        if not cat_cols.empty:
            features = pd.get_dummies(features, columns=cat_cols, drop_first=True)
        
        return features, target
    
    def process_pipeline(self, source: str = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Run the full data processing pipeline.
        
        Args:
            source: Data source name. If None, process all sources.
            
        Returns:
            Tuple of (features DataFrame, target Series if available)
        """
        # Load raw data
        df_raw = self.load_data(source)
        
        # Clean data
        df_clean = self.clean_data(df_raw)
        
        # Engineer features
        df_featured = self.engineer_features(df_clean)
        
        # Prepare for model
        features, target = self.prepare_for_model(df_featured)
        
        # Save processed data
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        if source:
            processed_file = f"{source}_processed_{timestamp}.csv"
        else:
            processed_file = f"all_sources_processed_{timestamp}.csv"
        
        processed_path = os.path.join(self.processed_data_dir, processed_file)
        
        # If target is available, combine with features for saving
        if target is not None:
            save_df = features.copy()
            if 'precipitation' in df_raw.columns:
                save_df['precipitation'] = target
            else:
                save_df['rainfall'] = target
        else:
            save_df = features
            
        save_df.to_csv(processed_path, index=False)
        self.logger.info(f"Saved processed data to {processed_path}")
        
        return features, target