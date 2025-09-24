"""
OpenWeatherMap API Client implementation
"""
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .weather_api_client import WeatherAPIClient

class OpenWeatherMapClient(WeatherAPIClient):
    """Client for the OpenWeatherMap API"""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenWeatherMap client"""
        super().__init__(api_key)
        self.logger = logging.getLogger(__name__)
    
    def _get_api_key_from_env(self) -> str:
        """Get API key from environment variables"""
        api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
        if not api_key:
            raise ValueError("OpenWeatherMap API key not found in environment variables")
        return api_key
    
    def get_current_weather(self, location: Dict[str, float]) -> Dict[str, Any]:
        """
        Get current weather for a location.
        
        Args:
            location: Dictionary with 'lat' and 'lon' keys.
            
        Returns:
            Current weather data.
        """
        url = f"{self.BASE_URL}/weather"
        params = {
            "lat": location["lat"],
            "lon": location["lon"],
            "appid": self.api_key,
            "units": "metric"  # Use metric units (Celsius, meters/sec, etc.)
        }
        
        self.logger.info(f"Getting current weather for location: {location}")
        result = self._make_request(url, params)
        
        # Add timestamp for when we retrieved this data
        result["_metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "source": "openweathermap"
        }
        
        return result
    
    def get_forecast(self, location: Dict[str, float], days: int = 5) -> Dict[str, Any]:
        """
        Get weather forecast for a location.
        
        Args:
            location: Dictionary with 'lat' and 'lon' keys.
            days: Number of days to forecast (max 5 for free tier).
            
        Returns:
            Forecast data.
        """
        url = f"{self.BASE_URL}/forecast"
        params = {
            "lat": location["lat"],
            "lon": location["lon"],
            "appid": self.api_key,
            "units": "metric"
        }
        
        self.logger.info(f"Getting {days}-day forecast for location: {location}")
        result = self._make_request(url, params)
        
        # Add metadata
        result["_metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "source": "openweathermap",
            "forecast_days": days
        }
        
        return result
    
    def get_historical_data(self, location: Dict[str, float], 
                          start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get historical weather data for a location.
        
        Note: This requires a paid plan on OpenWeatherMap.
        
        Args:
            location: Dictionary with 'lat' and 'lon' keys.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            
        Returns:
            Historical weather data.
        """
        # Convert dates to UNIX timestamps
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        
        url = f"{self.BASE_URL}/history/city"
        params = {
            "lat": location["lat"],
            "lon": location["lon"],
            "appid": self.api_key,
            "units": "metric",
            "start": start_ts,
            "end": end_ts
        }
        
        self.logger.info(f"Getting historical data for location: {location} from {start_date} to {end_date}")
        
        try:
            result = self._make_request(url, params)
            
            # Add metadata
            result["_metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "source": "openweathermap",
                "start_date": start_date,
                "end_date": end_date
            }
            
            return result
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            self.logger.warning("Historical data requires a paid OpenWeatherMap plan")
            return {"error": "Failed to retrieve historical data. This may require a paid API plan."}

    def get_air_pollution(self, location: Dict[str, float]) -> Dict[str, Any]:
        """
        Get current air pollution data for a location.
        
        Args:
            location: Dictionary with 'lat' and 'lon' keys.
            
        Returns:
            Air pollution data.
        """
        url = f"{self.BASE_URL}/air_pollution"
        params = {
            "lat": location["lat"],
            "lon": location["lon"],
            "appid": self.api_key
        }
        
        self.logger.info(f"Getting air pollution data for location: {location}")
        result = self._make_request(url, params)
        
        # Add metadata
        result["_metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "source": "openweathermap"
        }
        
        return result