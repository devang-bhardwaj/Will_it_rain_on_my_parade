# AI-Agent-Scraper Integration Guide

This guide explains how to use the AI-Agent-Scraper tool for collecting weather data for our "Will it rain on my parade?" project.

## What is AI-Agent-Scraper?

AI-Agent-Scraper is an intelligent web scraping tool that uses AI vision to navigate and extract data from websites. It can:

1. Understand natural language instructions
2. Navigate complex websites
3. Extract structured data from various sources
4. Handle CAPTCHAs and anti-bot measures
5. Export data in multiple formats (JSON, CSV, etc.)

## Setup

1. Ensure you have the AI-Agent-Scraper subdirectory in your project
2. Set up the Google API key for the AI-Agent-Scraper:

```bash
$env:GOOGLE_API_KEY='your-api-key-here'  # PowerShell
```

or 

```bash
export GOOGLE_API_KEY='your-api-key-here'  # Bash
```

3. Run the setup script in the AI-Agent-Scraper directory:

```bash
cd Ai-Agent-Scraper
python setup.py
```

## Using AI-Agent-Scraper for Weather Data Collection

There are two ways to use AI-Agent-Scraper for our project:

### 1. Using our Integration Class

We've created a `WeatherDataScraper` class in `src/data_collection/ai_agent_scraper_integration.py` that provides a simple interface:

```python
from src.data_collection.ai_agent_scraper_integration import WeatherDataScraper
import asyncio

# Create the scraper
scraper = WeatherDataScraper()

# Define a prompt for scraping
prompt = "Go to weather.gov, search for 'New York City', and extract the current conditions and 7-day forecast as JSON."

# Run the scraper
async def main():
    output_file = await scraper.scrape_weather_website(prompt, format="json")
    print(f"Data saved to {output_file}")

asyncio.run(main())
```

### 2. Direct Usage

You can also use the AI-Agent-Scraper directly through its web interface:

1. Start the AI-Agent-Scraper:

```bash
cd Ai-Agent-Scraper
python launch.py
```

2. Open your browser to `http://localhost:3000`
3. Enter your scraping prompt in the interface
4. Choose the output format (JSON recommended for structured data)
5. Click "Start Browser Pilot"
6. Wait for the scraping to complete
7. Download the results

## Recommended Data Sources for Scraping

The following websites contain valuable weather data for our project:

1. **Weather.gov** - Official US weather service with accurate forecasts
2. **WeatherUnderground** - Detailed weather data including historical records
3. **AccuWeather** - Long-range forecasts and precipitation predictions
4. **Earth.nullschool.net** - Global weather visualization with access to underlying data
5. **Windy.com** - Interactive weather maps with precipitation forecasts

## Example Prompts

Here are some effective prompts for collecting weather data:

```
Go to weather.gov, search for "{city_name}", and extract the current conditions, hourly forecast for today, and 7-day forecast as JSON.
```

```
Visit earth.nullschool.net, navigate to coordinates {lat},{lon}, focus on the precipitation layer, and collect precipitation forecasts for the next 72 hours as CSV.
```

```
Go to wunderground.com, search for "{city_name}", and extract historical rainfall data for the last 30 days as JSON. Include daily precipitation amounts and timestamps.
```

## Best Practices

1. **Be specific** in your prompts - mention exactly what data you want
2. **Include formats** - specify the output format in your prompt (JSON, CSV)
3. **Test scraping locally** before scheduling automated collection
4. **Check output quality** to ensure the expected data is being collected
5. **Respect website terms** and avoid excessive requests to any single site
6. **Use a VPN or proxy rotation** for larger scraping projects

## Troubleshooting

If you encounter issues with the AI-Agent-Scraper:

1. Check that your Google API key is set correctly
2. Verify that you have all dependencies installed
3. Look for error messages in the terminal or browser console
4. Try running with `enable_streaming=True` to watch the browser's actions
5. Simplify your prompt if the AI seems confused

## Moving Scraped Data

After scraping, move your data to the appropriate directory in our project structure:

```bash
# Example: Move JSON data files to the raw/scraped directory
move outputs/*.json data/raw/scraped/
```