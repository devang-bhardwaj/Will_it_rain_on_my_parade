# Data Collection Strategy

This document outlines the strategy for collecting weather and rainfall data for our "Will it rain on my parade?" project.

## Data Sources

### 1. NASA Earth Data

NASA provides several datasets that are relevant to our project:

- **IMERG (Integrated Multi-satellitE Retrievals for GPM)**: Provides precipitation estimates worldwide.
  - URL: [https://gpm.nasa.gov/data/imerg](https://gpm.nasa.gov/data/imerg)
  - Parameters: Precipitation rate, precipitation type

- **MODIS Atmosphere**: Cloud and aerosol properties.
  - URL: [https://modis-atmosphere.gsfc.nasa.gov/](https://modis-atmosphere.gsfc.nasa.gov/)
  - Parameters: Cloud coverage, cloud optical thickness

- **MERRA-2 (Modern-Era Retrospective analysis for Research and Applications, Version 2)**: Provides historical weather data.
  - URL: [https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/](https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/)
  - Parameters: Temperature, humidity, wind speed, atmospheric pressure

### 2. NOAA Weather Data

- **National Centers for Environmental Information (NCEI)**: Historical weather station data.
  - URL: [https://www.ncdc.noaa.gov/cdo-web/](https://www.ncdc.noaa.gov/cdo-web/)
  - Parameters: Temperature, precipitation, wind speed, humidity

- **Weather Prediction Center**: Forecast data and weather models.
  - URL: [https://www.wpc.ncep.noaa.gov/](https://www.wpc.ncep.noaa.gov/)

### 3. Open Weather Map API

- Provides current weather data, forecasts, and historical data through API.
  - URL: [https://openweathermap.org/api](https://openweathermap.org/api)
  - Parameters: Temperature, humidity, pressure, clouds, wind, rain

### 4. Local Weather Services

- Country-specific weather services that may have more localized data.

## Data Collection Methods

### Using AI-Agent-Scraper

We will use the AI-Agent-Scraper tool for websites that don't provide direct API access. This tool allows us to:

1. Navigate to weather data websites
2. Extract structured data from complex web pages
3. Handle authentication and session management
4. Export data in various formats (JSON, CSV)

#### Example AI-Agent-Scraper Prompts for Weather Data

```bash
Go to https://www.ncdc.noaa.gov/cdo-web/datasets, search for "Global Summary of the Day", and extract the dataset description, coverage period, and update frequency as JSON.
```

```bash
Visit https://earth.nullschool.net/, focus on rainfall visualization for North America region, take screenshots every 6 hours for the next 24 hours forecast, and save as both images and underlying data as CSV.
```

### API Integration

For services that provide APIs (like OpenWeatherMap), we'll create direct API integration scripts in the `src/data_collection` directory:

- `weather_api_client.py`: General interface for all weather APIs
- `nasa_earth_data.py`: Specific implementation for NASA Earth Data
- `noaa_data_collector.py`: Specific implementation for NOAA data
- `openweathermap_collector.py`: Implementation for OpenWeatherMap API

### Scheduled Collection

We'll set up scheduled collection tasks to gather:

1. Real-time weather data (hourly)
2. Short-term forecasts (6-hour intervals)
3. Historical data (daily collection until we have sufficient historical context)

## Data Storage

All collected data will be stored in the following structure:

```bash
data/
├── raw/
│   ├── nasa/
│   │   ├── imerg/
│   │   └── modis/
│   ├── noaa/
│   └── openweathermap/
└── processed/
    ├── combined_datasets/
    ├── feature_extracted/
    └── model_ready/
```

## Ethical and Technical Considerations

1. We will respect rate limits of all APIs and websites
2. Data will be properly cited and attributed to original sources
3. We'll implement error handling and retry mechanisms for API failures
4. All API keys and credentials will be stored securely using environment variables

## Next Steps

1. Set up API accounts and get necessary API keys
2. Configure AI-Agent-Scraper for each data source
3. Create initial data collection scripts
4. Run test collection to verify data quality and formats
5. Implement data quality checks and validation