"""
AI-Agent-Scraper integration for weather data collection
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

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

class WeatherDataScraper:
    """Class to scrape weather data using AI-Agent-Scraper"""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the weather data scraper.
        
        Args:
            output_dir: Directory to save scraped data. Defaults to data/raw/scraped/.
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir or str(Path(__file__).parent.parent.parent / "data" / "raw" / "scraped")
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def scrape_weather_website(self, prompt: str, format: str = "json"):
        """
        Scrape weather data from a website using AI-Agent-Scraper.
        
        Args:
            prompt: Natural language prompt for what to scrape.
            format: Output format (json, csv, txt, html, pdf).
            
        Returns:
            Path to the saved data file.
        """
        self.logger.info(f"Scraping with prompt: {prompt}")
        
        # Generate a unique job ID
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
    """Example usage of the WeatherDataScraper"""
    scraper = WeatherDataScraper()
    
    # Example location
    location = "New York City"
    
    # Get prompts for this location
    prompts = scraper.generate_prompts_for_location(location)
    
    # Scrape the first prompt as an example
    output_file = await scraper.scrape_weather_website(prompts[0])
    print(f"Scraped data saved to {output_file}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())