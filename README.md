# Will It Rain On My Parade?

> A weather prediction system for the NASA Space Apps Challenge 2025

## Project Overview & Progress Tracking Document

This repository contains our team's solution for the "Will It Rain On My Parade?" challenge as part of the NASA Space Apps Challenge 2025. Our project aims to develop a system for analyzing historical weather data to determine the likelihood of various weather conditions for specific locations and times, enabling users to plan their outdoor events accordingly.

### Challenge Details

**Event Date**: October 4-5, 2025

**Challenge Difficulty**: Intermediate

**Challenge Subjects**: Coding, Data Analysis, Data Visualization, Forecasting, Software, Weather, Web Development

**Challenge Summary**:
> If you're planning an outdoor event—like a vacation, a hike on a trail, or fishing on a lake—it would be good to know the chances of adverse weather for the time and location you are considering. There are many types of Earth observation data that can provide information on weather conditions for a particular location and day of the year. Your challenge is to construct an app with a personalized interface that enables users to conduct a customized query to tell them the likelihood of "very hot," "very cold," "very windy," "very wet," or "very uncomfortable" conditions for the location and time they specify.

**Challenge Objectives**:

- Develop an application using NASA Earth observation data for historical weather analysis
- Create a personalized dashboard for users to query weather condition probabilities
- Allow users to specify locations and times (day of the year) for queries
- Provide visualizations and statistics about the likelihood of specific weather conditions
- Include options for data download in formats like CSV or JSON

### Progress Dashboard

| Task | Status | Details |
|------|--------|---------|
| Understand NASA Challenge Requirements | ✅ Complete | Challenge details obtained and analyzed |
| Project Structure Setup | ✅ Complete | Directory structure and documentation established |
| AI-Agent-Scraper Configuration | 🔄 In Progress | Integration for NASA data collection |
| Historical Weather Data Collection | 📝 Not Started | Collecting NASA Earth observation datasets |
| Data Processing Pipeline | 📝 Not Started | For historical weather analysis |
| Statistical Analysis Implementation | 🔄 In Progress | Weather condition probability models |
| User Interface Development | 📝 Not Started | For location and time-based queries |
| Project Documentation | 🔄 In Progress | Ongoing throughout development |

## Project Structure

\\\ash
Will_it_rain_on_my_parade/
├── data/                   # Data storage directory
│   ├── raw/                # Raw, immutable data
│   └── processed/          # Cleaned and processed data ready for modeling
├── docs/                   # Documentation files
├── notebooks/              # Jupyter notebooks for exploration and visualization
│   └── Weather_Data_Exploratory_Analysis.ipynb  # EDA notebook with prototype models
├── scripts/                # Utility scripts
├── src/                    # Source code
│   ├── api/                # API for the model
│   │   └── app.py          # FastAPI application
│   ├── data_collection/    # Scripts for collecting data
│   │   ├── ai_agent_scraper_integration.py  # Integration with AI-Agent-Scraper
│   │   ├── openweathermap_collector.py      # OpenWeatherMap API client
│   │   └── weather_api_client.py            # Abstract base class for weather APIs
│   ├── data_processing/    # Scripts for processing data
│   │   └── weather_data_processor.py        # Data cleaning and transformation
│   ├── models/             # Machine learning models
│   │   └── rain_prediction_model.py         # Renamed to weather_condition_model.py
│   └── visualization/      # Visualization tools
└── tests/                  # Test files
\\\

## Current Development Status

### Completed

- ✅ Project structure setup with clear organization for collaborative development
- ✅ Initial documentation for team collaboration
- ✅ Exploratory Data Analysis (EDA) notebook with:
  - Data visualization and statistical analysis
  - Weather variable correlation analysis
  - Pattern recognition for different weather conditions
  - Feature importance analysis

### In Progress

- 🔄 AI-Agent-Scraper integration for NASA data collection
- 🔄 Documentation updates based on project evolution

### Pending

- 📝 NASA Earth observation data integration
- 📝 Production-ready data processing pipeline
- 📝 Statistical models for condition probability analysis
- 📝 User interface development for location and time-based queries
- 📝 API implementation for data access
- 📝 Challenge submission materials

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- Google AI API key (for AI-Agent-Scraper integration)

### Installation

1. Clone the repository:

   \\\ash
   git clone https://github.com/devang-bhardwaj/Will_it_rain_on_my_parade.git
   cd Will_it_rain_on_my_parade
   \\\

2. Create a virtual environment and activate it:

   \\\ash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   \\\

3. Install the required dependencies:

   \\\ash
   pip install -r requirements.txt
   \\\

## Data Collection Strategy

Our data collection approach focuses on historical weather data analysis:

1. **NASA Earth Observation Data**: Primary source for historical weather condition analysis, including:
   - MODIS (Moderate Resolution Imaging Spectroradiometer) for temperature data
   - GPM (Global Precipitation Measurement) for precipitation data
   - MERRA-2 (Modern-Era Retrospective Analysis for Research and Applications) for comprehensive weather variables
   - POWER (Prediction of Worldwide Energy Resources) for solar radiation and meteorological data

2. **AI-Agent-Scraper Tool**: For efficiently gathering and processing NASA's Earth observation datasets

3. **Supplementary Data**: Additional historical weather records to enhance analysis accuracy

See the [data collection documentation](./docs/data_collection.md) for more details.

## Statistical Analysis Approach

We're developing a historical weather analysis system that provides probabilistic insights:

1. **Condition Probability Analysis**: Calculating the likelihood of specific weather conditions based on historical data
2. **Temporal Pattern Recognition**: Identifying seasonal and daily patterns for different locations
3. **Multi-condition Analysis**: Determining compound probabilities (e.g., hot and windy, cold and wet)

Our approach combines:

- Comprehensive data aggregation by location and time of year
- Statistical modeling for probability distribution
- Advanced visualization techniques for intuitive understanding
- Data export options for user-specific analysis

Refer to the [analysis methodology documentation](./docs/analysis_methodology.md) and the EDA notebook for implementation details.

## Next Steps

1. Complete the AI-Agent-Scraper integration for NASA data collection
2. Acquire and process key NASA Earth observation datasets:
   - Temperature data (MODIS, POWER)
   - Precipitation data (GPM, IMERG)
   - Wind data (MERRA-2)
   - Humidity and comfort indices (POWER)
3. Develop the historical data processing pipeline
4. Create statistical models for weather condition probability analysis
5. Implement user interface for location and time-based queries
6. Add data visualization and export functionality

## Key Notebook: Exploratory Data Analysis

We've created a comprehensive EDA notebook (\
otebooks/Weather_Data_Exploratory_Analysis.ipynb\) that explores historical weather patterns and builds statistical analysis models. The notebook includes:

1. **Data Analysis and Preparation**:
   - Historical weather data for various locations
   - Statistical analysis of key weather variables
   - Threshold determination for condition categories (very hot, very cold, etc.)

2. **Visualization and Pattern Recognition**:
   - Seasonal and daily weather patterns
   - Historical probability distributions for different conditions
   - Geographical analysis of weather condition frequencies
   - Time-series visualization of condition occurrences

3. **Statistical Processing**:
   - Temporal aggregation by day of year
   - Location-specific probability calculations
   - Condition severity classification
   - Comfort index calculations

4. **Probability Models**:
   - Conditional probability analysis for different weather conditions
   - Historical frequency analysis by location and time
   - Confidence interval calculations
   - Visualization of probability distributions

This notebook serves as a foundation for our historical weather analysis approach and will be expanded with NASA Earth observation data.

## Team Members

- [Team Member 1]
- [Team Member 2]
- [Team Member 3]
- [Team Member 4]
- [Team Member 5]

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NASA Space Apps Challenge for the opportunity
- Contributors to the AI-Agent-Scraper tool
- Open-source weather data providers
- NASA Earth observation data teams
