"""
NASA Earth Data API Client
Handles authentication and data retrieval from NASA Earth observation datasets
"""
import os
import requests
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class NASAEarthDataClient:
    """Client for accessing NASA Earth observation data"""
    
    def __init__(self):
        """Initialize NASA Earth Data client with credentials"""
        self.username = os.getenv('NASA_EARTHDATA_USERNAME')
        self.password = os.getenv('NASA_EARTHDATA_PASSWORD')
        self.ges_disc_url = os.getenv('NASA_GES_DISC_URL', 'https://disc.gsfc.nasa.gov')
        self.power_url = os.getenv('NASA_POWER_URL', 'https://power.larc.nasa.gov')
        self.laads_url = os.getenv('NASA_LAADS_DAAC_URL', 'https://ladsweb.modaps.eosdis.nasa.gov')
        
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with NASA Earthdata Login"""
        if not self.username or not self.password:
            raise ValueError("NASA Earthdata credentials not found in environment variables")
        
        # Set up authentication
        self.session.auth = (self.username, self.password)
        self.logger.info("NASA Earthdata authentication configured")
    
    def get_power_data(self, 
                      latitude: float, 
                      longitude: float, 
                      start_date: str, 
                      end_date: str,
                      parameters: List[str] = None) -> pd.DataFrame:
        """
        Get meteorological data from NASA POWER API
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate  
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            parameters: List of parameters to retrieve
            
        Returns:
            DataFrame with weather data
        """
        if parameters is None:
            parameters = [
                'T2M',      # Temperature at 2 meters
                'T2M_MAX',  # Maximum temperature
                'T2M_MIN',  # Minimum temperature
                'PRECTOTCORR',  # Precipitation
                'WS2M',     # Wind speed at 2 meters
                'RH2M'      # Relative humidity
            ]
        
        # Build API URL
        api_url = f"{self.power_url}/api/temporal/daily/point"
        params = {
            'parameters': ','.join(parameters),
            'community': 'RE',
            'longitude': longitude,
            'latitude': latitude,
            'start': start_date,
            'end': end_date,
            'format': 'JSON'
        }
        
        try:
            response = self.session.get(api_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            properties = data['properties']['parameter']
            dates = list(properties[parameters[0]].keys())
            
            df_data = {'date': dates}
            for param in parameters:
                df_data[param] = list(properties[param].values())
            
            df = pd.DataFrame(df_data)
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df['latitude'] = latitude
            df['longitude'] = longitude
            
            self.logger.info(f"Retrieved {len(df)} records from NASA POWER")
            return df
            
        except Exception as e:
            self.logger.error(f"Error retrieving POWER data: {e}")
            raise
    
    def get_giovanni_data_url(self, 
                             dataset: str,
                             variable: str,
                             start_date: str,
                             end_date: str,
                             bbox: Tuple[float, float, float, float]) -> str:
        """
        Generate Giovanni data access URL
        
        Args:
            dataset: Dataset identifier (e.g., 'GPM_3IMERGDF')
            variable: Variable name
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            bbox: Bounding box (west, south, east, north)
            
        Returns:
            Giovanni data access URL
        """
        giovanni_url = os.getenv('NASA_GIOVANNI_URL', 'https://giovanni.gsfc.nasa.gov')
        
        # Build Giovanni URL for data access
        west, south, east, north = bbox
        url = f"{giovanni_url}/giovanni/#service=TmAvMp&starttime={start_date}T00:00:00Z&endtime={end_date}T23:59:59Z&bbox={west},{south},{east},{north}&data={dataset}%3A{variable}"
        
        return url
    
    def download_file(self, url: str, local_path: str) -> bool:
        """
        Download a file from NASA servers
        
        Args:
            url: URL to download from
            local_path: Local path to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"Downloaded file to {local_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading file: {e}")
            return False
    
    def search_datasets(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Search for available datasets
        
        Args:
            keywords: Keywords to search for
            
        Returns:
            List of dataset information
        """
        # This is a placeholder - in practice, you'd use NASA's Common Metadata Repository (CMR) API
        datasets = [
            {
                'id': 'GPM_3IMERGDF',
                'title': 'GPM IMERG Final Precipitation L3 1 day',
                'description': 'Daily precipitation data from GPM mission',
                'variables': ['precipitationCal'],
                'temporal_coverage': '2014-present',
                'spatial_resolution': '0.1 degree'
            },
            {
                'id': 'MOD11A1',
                'title': 'MODIS/Terra Land Surface Temperature/Emissivity Daily L3',
                'description': 'Daily land surface temperature from MODIS Terra',
                'variables': ['LST_Day_1km', 'LST_Night_1km'],
                'temporal_coverage': '2000-present',
                'spatial_resolution': '1 km'
            },
            {
                'id': 'MERRA2_400',
                'title': 'MERRA-2 tavg1_2d_slv_Nx: 2d,1-Hourly,Time-Averaged,Single-Level,Assimilation,Single-Level Diagnostics',
                'description': 'Hourly surface meteorological data',
                'variables': ['T2M', 'U2M', 'V2M', 'PRECTOT'],
                'temporal_coverage': '1980-present',
                'spatial_resolution': '0.5 x 0.625 degree'
            }
        ]
        
        # Filter by keywords
        filtered_datasets = []
        for dataset in datasets:
            for keyword in keywords:
                if keyword.lower() in dataset['title'].lower() or keyword.lower() in dataset['description'].lower():
                    filtered_datasets.append(dataset)
                    break
        
        return filtered_datasets