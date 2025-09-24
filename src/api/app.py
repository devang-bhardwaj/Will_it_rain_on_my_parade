"""
API for the weather prediction model
"""
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our model
from models.rain_prediction_model import RainOccurrenceModel, RainfallAmountModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Will it rain on my parade? API",
    description="API for weather and rainfall predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define models
class Location(BaseModel):
    """Location model for predictions"""
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    name: Optional[str] = Field(None, description="Optional location name")

class WeatherFeatures(BaseModel):
    """Weather features for prediction"""
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Relative humidity (0-100%)")
    pressure: float = Field(..., description="Atmospheric pressure in hPa")
    wind_speed: float = Field(..., description="Wind speed in m/s")
    wind_direction: float = Field(..., description="Wind direction in degrees")
    cloud_cover: float = Field(..., description="Cloud cover percentage (0-100%)")
    
    # Optional features
    dew_point: Optional[float] = Field(None, description="Dew point in Celsius")
    uv_index: Optional[float] = Field(None, description="UV index")
    visibility: Optional[float] = Field(None, description="Visibility in kilometers")

class PredictionRequest(BaseModel):
    """Request model for weather predictions"""
    location: Location
    features: WeatherFeatures
    timestamp: Optional[datetime] = Field(None, description="Prediction timestamp, defaults to current time")

class RainPrediction(BaseModel):
    """Response model for rain predictions"""
    location: Location
    will_it_rain: bool = Field(..., description="Whether rain is predicted")
    rain_probability: float = Field(..., description="Probability of rain (0-100%)")
    expected_rainfall_mm: float = Field(..., description="Expected rainfall amount in mm")
    prediction_time: datetime = Field(..., description="Time of prediction")
    forecast_timestamp: datetime = Field(..., description="Time for which the forecast is made")

# Load models
rain_occurrence_model = None
rainfall_amount_model = None

try:
    rain_occurrence_model = RainOccurrenceModel()
    rain_occurrence_model.load()
    logger.info("Rain occurrence model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load rain occurrence model: {e}")
    rain_occurrence_model = None

try:
    rainfall_amount_model = RainfallAmountModel()
    rainfall_amount_model.load()
    logger.info("Rainfall amount model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load rainfall amount model: {e}")
    rainfall_amount_model = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "rain_occurrence_model_loaded": rain_occurrence_model is not None,
        "rainfall_amount_model_loaded": rainfall_amount_model is not None,
        "timestamp": datetime.now()
    }

@app.post("/predict/rain", response_model=RainPrediction)
async def predict_rain(request: PredictionRequest):
    """
    Predict rainfall for a specific location and time
    """
    # Check if models are loaded
    if rain_occurrence_model is None or rainfall_amount_model is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please check with administrator."
        )
    
    try:
        # Convert request to feature format expected by the model
        import pandas as pd
        
        # Use the timestamp if provided, otherwise use current time
        timestamp = request.timestamp or datetime.now()
        
        # Create feature DataFrame
        features = pd.DataFrame({
            # Weather features
            'temperature': [request.features.temperature],
            'humidity': [request.features.humidity],
            'pressure': [request.features.pressure],
            'wind_speed': [request.features.wind_speed],
            'wind_direction': [request.features.wind_direction],
            'cloud_cover': [request.features.cloud_cover],
            
            # Optional features
            'dew_point': [request.features.dew_point or request.features.temperature - 2.5],
            'uv_index': [request.features.uv_index or 0],
            'visibility': [request.features.visibility or 10],
            
            # Location features
            'latitude': [request.location.latitude],
            'longitude': [request.location.longitude],
            
            # Time features
            'month': [timestamp.month],
            'day': [timestamp.day],
            'hour': [timestamp.hour],
            'month_sin': [float(pd.Series([timestamp.month]).apply(
                lambda x: np.sin(2 * np.pi * x / 12))[0])
            ],
            'month_cos': [float(pd.Series([timestamp.month]).apply(
                lambda x: np.cos(2 * np.pi * x / 12))[0])
            ],
            'hour_sin': [float(pd.Series([timestamp.hour]).apply(
                lambda x: np.sin(2 * np.pi * x / 24))[0])
            ],
            'hour_cos': [float(pd.Series([timestamp.hour]).apply(
                lambda x: np.cos(2 * np.pi * x / 24))[0])
            ],
        })
        
        # Make predictions
        import numpy as np
        
        rain_prob = float(rain_occurrence_model.predict_proba(features)[0])
        will_it_rain = rain_prob >= 0.5
        
        # Only predict rainfall amount if rain is predicted
        if will_it_rain:
            rainfall_mm = float(rainfall_amount_model.predict(features)[0])
        else:
            rainfall_mm = 0.0
        
        # Return prediction
        return RainPrediction(
            location=request.location,
            will_it_rain=will_it_rain,
            rain_probability=rain_prob * 100,  # Convert to percentage
            expected_rainfall_mm=rainfall_mm,
            prediction_time=datetime.now(),
            forecast_timestamp=timestamp
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

if __name__ == "__main__":
    # Run the API with Uvicorn when this script is executed directly
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)