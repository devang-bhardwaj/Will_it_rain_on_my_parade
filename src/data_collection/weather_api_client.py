"""
Weather API Client - Base class for all weather API interactions
"""
import os
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

class WeatherAPIClient(ABC):
    """Base class for all weather API clients"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the API client.
        
        Args:
            api_key: API key for the service. If not provided, will try to load from environment variables.
        """
        self.api_key = api_key or self._get_api_key_from_env()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def _get_api_key_from_env(self) -> str:
        """Get API key from environment variables"""
        pass
    
    @abstractmethod
    def get_current_weather(self, location: Dict[str, float]) -> Dict[str, Any]:
        """
        Get current weather for a location.
        
        Args:
            location: Dictionary with 'lat' and 'lon' keys for latitude and longitude.
            
        Returns:
            Dictionary with weather data.
        """
        pass
    
    @abstractmethod
    def get_forecast(self, location: Dict[str, float], days: int = 5) -> Dict[str, Any]:
        """
        Get weather forecast for a location.
        
        Args:
            location: Dictionary with 'lat' and 'lon' keys for latitude and longitude.
            days: Number of days to forecast.
            
        Returns:
            Dictionary with forecast data.
        """
        pass
    
    @abstractmethod
    def get_historical_data(self, location: Dict[str, float], 
                           start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get historical weather data for a location.
        
        Args:
            location: Dictionary with 'lat' and 'lon' keys for latitude and longitude.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            
        Returns:
            Dictionary with historical weather data.
        """
        pass
    
    def _make_request(self, url: str, params: Dict[str, Any] = None, 
                     headers: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Make a request to the API.
        
        Args:
            url: URL to request.
            params: URL parameters.
            headers: Request headers.
            
        Returns:
            Response JSON as dictionary.
        """
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise
    
    def save_to_file(self, data: Dict[str, Any], file_path: str) -> None:
        """
        Save data to a file.
        
        Args:
            data: Data to save.
            file_path: Path to save the file to.
        """
        import json
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)