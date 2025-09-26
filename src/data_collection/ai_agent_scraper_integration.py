"""
AI-Agent-Scraper integration for NASA weather data collection
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the AI-Agent-Scraper to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent / "Ai-Agent-Scraper"))

# Import the AI-Agent-Scraper API
try:
    from backend.agent import run_agent
    from backend.smart_browser_controller import SmartBrowserController
except ImportError:
    raise ImportError(
        "AI-Agent-Scraper not found. Please make sure it's in the correct location."
    )

class NASAWeatherDataScraper:
    """Class to scrape NASA weather data using AI-Agent-Scraper"""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the NASA weather data scraper.
        
        Args:
            output_dir: Directory to save scraped data. Defaults to data/raw/nasa/.
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir or str(Path(__file__).parent.parent.parent / "data" / "raw" / "nasa")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # NASA credentials from environment
        self.nasa_username = os.getenv('NASA_EARTHDATA_USERNAME')
        self.nasa_password = os.getenv('NASA_EARTHDATA_PASSWORD')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
    
    async def scrape_nasa_giovanni(self, dataset: str, start_date: str, end_date: str, 
                                   location: str = "global", format: str = "json"):
        """
        Scrape NASA Giovanni data portal for weather datasets.
        
        Args:
            dataset: NASA dataset to scrape (e.g., "GPM IMERG precipitation")
            start_date: Start date in MM/DD/YYYY format
            end_date: End date in MM/DD/YYYY format
            location: Location or region of interest
            format: Output format (json, csv, txt, html, pdf)
            
        Returns:
            Path to the saved data file.
        """
        prompt = f"""
        Navigate to NASA Giovanni (https://giovanni.gsfc.nasa.gov/giovanni/).
        Login using username: {self.nasa_username} and password: {self.nasa_password} if required.
        Search for the dataset: {dataset}
        Set the date range from {start_date} to {end_date}
        Set the location/region to: {location}
        Download the data and extract the dataset information including:
        - Data values and timestamps
        - Geographic coordinates
        - Metadata about the dataset
        - Quality flags if available
        Export the results as {format} format.
        """
        
        return await self._execute_scraping_job(prompt, f"giovanni_{dataset}", format)
    
    async def scrape_nasa_power(self, parameters: list, latitude: float, longitude: float,
                               start_date: str, end_date: str, format: str = "json"):
        """
        Scrape NASA POWER data portal for meteorological data.
        
        Args:
            parameters: List of parameters to collect (e.g., ["temperature", "precipitation"])
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            start_date: Start date in MM/DD/YYYY format
            end_date: End date in MM/DD/YYYY format
            format: Output format
            
        Returns:
            Path to the saved data file.
        """
        params_str = ", ".join(parameters)
        prompt = f"""
        Navigate to NASA POWER Data Access Viewer (https://power.larc.nasa.gov/data-access-viewer/).
        Set the following parameters:
        - Parameters: {params_str}
        - Latitude: {latitude}
        - Longitude: {longitude}
        - Start Date: {start_date}
        - End Date: {end_date}
        - Output Format: CSV
        Download the data and extract:
        - Daily weather measurements
        - Parameter values and units
        - Data quality indicators
        - Geographic and temporal metadata
        Export as {format} format with all available metadata.
        """
        
        return await self._execute_scraping_job(prompt, f"power_lat{latitude}_lon{longitude}", format)
    
    async def scrape_ges_disc(self, dataset: str, variable: str, region: str,
                             start_date: str, end_date: str, format: str = "json"):
        """
        Scrape NASA GES DISC for Earth science data.
        
        Args:
            dataset: Dataset name (e.g., "MERRA-2", "GPM IMERG")
            variable: Variable of interest (e.g., "precipitation", "temperature")
            region: Geographic region
            start_date: Start date
            end_date: End date
            format: Output format
            
        Returns:
            Path to the saved data file.
        """
        prompt = f"""
        Navigate to NASA GES DISC (https://disc.gsfc.nasa.gov/).
        Login with Earthdata credentials if required: username {self.nasa_username}
        Search for dataset: {dataset}
        Select variable: {variable}
        Set geographic region: {region}
        Set temporal range from {start_date} to {end_date}
        Access the data through OPeNDAP or direct download
        Extract:
        - Data values for the specified variable
        - Coordinate information (lat, lon, time)
        - Data attributes and metadata
        - File format information
        Export the extracted data as {format}.
        """
        
        return await self._execute_scraping_job(prompt, f"gesdisc_{dataset}_{variable}", format)
    
    async def _execute_scraping_job(self, prompt: str, job_name: str, format: str = "json"):
        """
        Execute a scraping job with AI-Agent-Scraper.
        
        Args:
            prompt: Natural language prompt for scraping
            job_name: Unique name for this job
            format: Output format
            
        Returns:
            Path to the saved data file.
        """
        self.logger.info(f"Starting scraping job: {job_name}")
        
        # Generate a unique job ID with timestamp
        job_id = f"weather_scrape_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Define headless mode (True for production, False for debugging)
        headless = True
        
        # No proxy for simplicity, but could be added
        proxy = None
        
        # Enable streaming for debugging (set to False for production)
        enable_streaming = False
        
        try:
            # Run the AI-Agent-Scraper
            result = await run_agent(
                job_id=job_id,
                prompt=prompt,
                fmt=format,
                headless=headless,
                proxy=proxy,
                enable_streaming=enable_streaming
            )
            
            # Save the result to the output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                self.output_dir, 
                f"weather_data_{timestamp}.{format}"
            )
            
            with open(output_file, "w") as f:
                if format == "json":
                    json.dump(result, f, indent=2)
                else:
                    f.write(result)
                    
            self.logger.info(f"Data saved to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error scraping weather data: {e}")
            raise

    def generate_prompts_for_location(self, location_name: str) -> list:
        """
        Generate a list of prompts for different weather data sources for a specific location.
        
        Args:
            location_name: Name of the location (city, region, etc.)
            
        Returns:
            List of prompts for AI-Agent-Scraper.
        """
        return [
            # Weather.gov for US locations
            f"Go to weather.gov, search for '{location_name}', and extract the current conditions, " +
            "7-day forecast, and any weather alerts or warnings as JSON.",
            
            # Weather Underground
            f"Visit wunderground.com, search for '{location_name}', and collect the detailed " +
            "current conditions, hourly forecast for the next 24 hours, and 10-day forecast as CSV.",
            
            # AccuWeather
            f"Navigate to accuweather.com, find '{location_name}', and extract the current weather, " +
            "hourly forecast, and daily forecast for the next 15 days as JSON.",
            
            # Weather Channel
            f"Go to weather.com, search for '{location_name}', and get the current conditions, " +
            "today's weather details, and 10-day forecast as JSON.",
        ]

async def main():
    """Example usage of the NASAWeatherDataScraper"""
    scraper = NASAWeatherDataScraper()
    
    # Example: Scrape NASA POWER data for New York City
    latitude, longitude = 40.7128, -74.0060  # NYC coordinates
    start_date, end_date = "01/01/2020", "12/31/2020"
    parameters = ["temperature", "precipitation", "wind speed"]
    
    # Scrape NASA POWER data
    output_file = await scraper.scrape_nasa_power(
        parameters=parameters,
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date
    )
    print(f"NASA POWER data saved to {output_file}")
    
    # Example: Scrape Giovanni for precipitation data
    giovanni_file = await scraper.scrape_nasa_giovanni(
        dataset="GPM IMERG precipitation",
        start_date=start_date,
        end_date=end_date,
        location="North America"
    )
    print(f"Giovanni data saved to {giovanni_file}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())