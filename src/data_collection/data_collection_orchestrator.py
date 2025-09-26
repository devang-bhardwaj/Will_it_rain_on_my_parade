"""
Data Collection Orchestrator
Coordinates data collection from multiple NASA sources for weather analysis
"""
import asyncio
import os
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path

from .nasa_earth_data_client import NASAEarthDataClient

class WeatherDataCollector:
    """Orchestrates collection of weather data from NASA sources"""
    
    def __init__(self, base_data_dir: str = None):
        """
        Initialize the weather data collector.
        
        Args:
            base_data_dir: Base directory for storing collected data
        """
        self.base_data_dir = base_data_dir or str(Path(__file__).parent.parent.parent / "data" / "raw")
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients
        self.nasa_client = NASAEarthDataClient()
        self.scraper = None
        
        # Try to initialize scraper (optional)
        try:
            from .ai_agent_scraper_integration import NASAWeatherDataScraper
            self.scraper = NASAWeatherDataScraper()
            self.logger.info("AI-Agent-Scraper initialized successfully")
        except ImportError as e:
            self.logger.warning(f"AI-Agent-Scraper not available: {e}")
            self.logger.info("Data collection will use NASA API only")
        
        # Data collection status
        self.collection_status = {}
    
    async def collect_location_data(self, 
                                   latitude: float, 
                                   longitude: float,
                                   location_name: str,
                                   start_year: int = 2010,
                                   end_year: int = 2023) -> Dict[str, str]:
        """
        Collect comprehensive weather data for a specific location.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            location_name: Human-readable location name
            start_year: Start year for data collection
            end_year: End year for data collection
            
        Returns:
            Dictionary with paths to collected data files
        """
        self.logger.info(f"Starting data collection for {location_name} ({latitude}, {longitude})")
        
        # Create location-specific directory
        location_dir = os.path.join(self.base_data_dir, "locations", 
                                  location_name.replace(" ", "_").replace(",", ""))
        os.makedirs(location_dir, exist_ok=True)
        
        collected_files = {}
        
        # 1. Collect NASA POWER data
        self.logger.info("Collecting NASA POWER data...")
        try:
            power_data = await self._collect_power_data(
                latitude, longitude, start_year, end_year, location_dir
            )
            collected_files['power'] = power_data
        except Exception as e:
            self.logger.error(f"Failed to collect POWER data: {e}")
        
        # 2. Collect MODIS temperature data via scraping (if scraper available)
        if self.scraper:
            self.logger.info("Collecting MODIS temperature data...")
            try:
                modis_file = await self.scraper.scrape_ges_disc(
                    dataset="MODIS Terra Land Surface Temperature",
                    variable="LST_Day_1km",
                    region=f"{latitude-0.5},{longitude-0.5},{latitude+0.5},{longitude+0.5}",
                    start_date=f"01/01/{start_year}",
                    end_date=f"12/31/{end_year}"
                )
                collected_files['modis'] = modis_file
            except Exception as e:
                self.logger.error(f"Failed to collect MODIS data: {e}")
        else:
            self.logger.info("Skipping MODIS data collection (scraper not available)")
        
        # 3. Collect GPM precipitation data (if scraper available)
        if self.scraper:
            self.logger.info("Collecting GPM precipitation data...")
            try:
                gpm_file = await self.scraper.scrape_nasa_giovanni(
                    dataset="GPM IMERG Final Precipitation",
                    start_date=f"01/01/{start_year}",
                    end_date=f"12/31/{end_year}",
                    location=f"lat: {latitude}, lon: {longitude}"
                )
                collected_files['gpm'] = gpm_file
            except Exception as e:
                self.logger.error(f"Failed to collect GPM data: {e}")
        else:
            self.logger.info("Skipping GPM data collection (scraper not available)")
        
        # 4. Collect MERRA-2 data (if scraper available)
        if self.scraper:
            self.logger.info("Collecting MERRA-2 meteorological data...")
            try:
                merra2_file = await self.scraper.scrape_ges_disc(
                    dataset="MERRA-2",
                    variable="T2M,U2M,V2M,PRECTOT",
                    region=f"{latitude-0.5},{longitude-0.5},{latitude+0.5},{longitude+0.5}",
                    start_date=f"01/01/{start_year}",
                    end_date=f"12/31/{end_year}"
                )
                collected_files['merra2'] = merra2_file
            except Exception as e:
                self.logger.error(f"Failed to collect MERRA-2 data: {e}")
        else:
            self.logger.info("Skipping MERRA-2 data collection (scraper not available)")
        
        # Save collection metadata
        metadata = {
            'location_name': location_name,
            'latitude': latitude,
            'longitude': longitude,
            'start_year': start_year,
            'end_year': end_year,
            'collection_date': datetime.now().isoformat(),
            'collected_files': collected_files
        }
        
        metadata_path = os.path.join(location_dir, 'collection_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Data collection completed for {location_name}")
        return collected_files
    
    async def _collect_power_data(self, latitude: float, longitude: float, 
                                 start_year: int, end_year: int, location_dir: str) -> str:
        """Collect data from NASA POWER API"""
        # Collect data year by year to avoid API limits
        all_data = []
        
        for year in range(start_year, end_year + 1):
            start_date = f"{year}0101"
            end_date = f"{year}1231"
            
            try:
                yearly_data = self.nasa_client.get_power_data(
                    latitude=latitude,
                    longitude=longitude,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'WS2M', 'RH2M']
                )
                all_data.append(yearly_data)
                self.logger.info(f"Collected POWER data for year {year}")
                
                # Small delay to respect API limits
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.warning(f"Failed to collect POWER data for year {year}: {e}")
        
        if all_data:
            # Combine all years
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Save to file
            power_file = os.path.join(location_dir, 'nasa_power_data.csv')
            combined_data.to_csv(power_file, index=False)
            
            return power_file
        
        return None
    
    async def collect_multiple_locations(self, locations: List[Dict[str, any]]) -> Dict[str, Dict[str, str]]:
        """
        Collect data for multiple locations in parallel.
        
        Args:
            locations: List of location dictionaries with 'name', 'lat', 'lon' keys
            
        Returns:
            Dictionary mapping location names to collected file paths
        """
        tasks = []
        for location in locations:
            task = self.collect_location_data(
                latitude=location['lat'],
                longitude=location['lon'],
                location_name=location['name']
            )
            tasks.append(task)
        
        # Run collections in parallel (with some limit to avoid overwhelming servers)
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent collections
        
        async def limited_collect(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[limited_collect(task) for task in tasks])
        
        # Map results to location names
        location_results = {}
        for i, location in enumerate(locations):
            location_results[location['name']] = results[i]
        
        return location_results
    
    def get_collection_status(self) -> Dict[str, any]:
        """Get the current status of data collection operations"""
        return {
            'nasa_client_status': 'active' if self.nasa_client else 'inactive',
            'scraper_status': 'active' if self.scraper else 'inactive',
            'collection_history': self.collection_status,
            'base_data_directory': self.base_data_dir
        }