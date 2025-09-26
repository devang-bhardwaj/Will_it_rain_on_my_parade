""""""

Will It Rain On My Parade? - API ServerWill It Rain On My Parade? - API Server

NASA Space Apps Challenge EntryNASA Space Apps Challenge Entry



Simple Flask API for weather predictionSimple Flask API for weather prediction

""""""



from flask import Flask, request, jsonifyfrom flask import Flask, request, jsonify

import sysimport sys

import osimport os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



from models.weather_predictor import WeatherPredictorfrom models.weather_predictor import WeatherPredictor

from datetime import datetimefrom datetime import datetime

import loggingimport logging



app = Flask(__name__)app = Flask(__name__)

logging.basicConfig(level=logging.INFO)logging.basicConfig(level=logging.INFO)



# Initialize predictor# Initialize predictor

predictor = WeatherPredictor()predictor = WeatherPredictor()

    humidity: float = Field(..., description="Relative humidity (0-100%)")

@app.route('/', methods=['GET'])    pressure: float = Field(..., description="Atmospheric pressure in hPa")

def home():    wind_speed: float = Field(..., description="Wind speed in m/s")

    """API home page"""    wind_direction: float = Field(..., description="Wind direction in degrees")

    return jsonify({    cloud_cover: float = Field(..., description="Cloud cover percentage (0-100%)")

        'name': 'Will It Rain On My Parade?',    

        'description': 'NASA Space Apps Challenge - Weather prediction for outdoor events',    # Optional features

        'version': '1.0',    dew_point: Optional[float] = Field(None, description="Dew point in Celsius")

        'endpoints': {    uv_index: Optional[float] = Field(None, description="UV index")

            '/predict': 'POST - Predict weather conditions',    visibility: Optional[float] = Field(None, description="Visibility in kilometers")

            '/health': 'GET - Health check'

        }class PredictionRequest(BaseModel):

    })    """Request model for weather predictions"""

    location: Location

@app.route('/health', methods=['GET'])    features: WeatherFeatures

def health():    timestamp: Optional[datetime] = Field(None, description="Prediction timestamp, defaults to current time")

    """Health check endpoint"""

    return jsonify({class RainPrediction(BaseModel):

        'status': 'healthy',    """Response model for rain predictions"""

        'timestamp': datetime.now().isoformat(),    location: Location

        'models_loaded': bool(predictor.models)    will_it_rain: bool = Field(..., description="Whether rain is predicted")

    })    rain_probability: float = Field(..., description="Probability of rain (0-100%)")

    expected_rainfall_mm: float = Field(..., description="Expected rainfall amount in mm")

@app.route('/predict', methods=['POST'])    prediction_time: datetime = Field(..., description="Time of prediction")

def predict():    forecast_timestamp: datetime = Field(..., description="Time for which the forecast is made")

    """

    Predict weather conditions for outdoor events# Load models

    rain_occurrence_model = None

    Expected JSON input:rainfall_amount_model = None

    {

        "temperature_avg": 25,try:

        "temperature_max": 30,    rain_occurrence_model = RainOccurrenceModel()

        "temperature_min": 20,    rain_occurrence_model.load()

        "precipitation": 0,    logger.info("Rain occurrence model loaded successfully")

        "wind_speed": 5,except Exception as e:

        "humidity": 60,    logger.error(f"Failed to load rain occurrence model: {e}")

        "month": 7,    rain_occurrence_model = None

        "day_of_year": 200,

        "latitude": 40.7128,try:

        "longitude": -74.0060    rainfall_amount_model = RainfallAmountModel()

    }    rainfall_amount_model.load()

    """    logger.info("Rainfall amount model loaded successfully")

    try:except Exception as e:

        data = request.get_json()    logger.error(f"Failed to load rainfall amount model: {e}")

            rainfall_amount_model = None

        # Validate required fields

        required_fields = [@app.get("/health")

            'temperature_avg', 'temperature_max', 'temperature_min',async def health_check():

            'precipitation', 'wind_speed', 'humidity',    """Health check endpoint"""

            'month', 'day_of_year', 'latitude', 'longitude'    return {

        ]        "status": "ok",

                "rain_occurrence_model_loaded": rain_occurrence_model is not None,

        missing_fields = [field for field in required_fields if field not in data]        "rainfall_amount_model_loaded": rainfall_amount_model is not None,

        if missing_fields:        "timestamp": datetime.now()

            return jsonify({    }

                'error': f'Missing required fields: {missing_fields}'

            }), 400@app.post("/predict/rain", response_model=RainPrediction)

        async def predict_rain(request: PredictionRequest):

        # Calculate heat index if not provided    """

        if 'heat_index' not in data:    Predict rainfall for a specific location and time

            temp = data['temperature_avg']    """

            humidity = data['humidity']    # Check if models are loaded

            data['heat_index'] = temp + (humidity / 100) * 5  # Simplified heat index    if rain_occurrence_model is None or rainfall_amount_model is None:

                raise HTTPException(

        # Make prediction            status_code=503,

        predictions = predictor.predict_conditions(data)            detail="Models not loaded. Please check with administrator."

        recommendation = predictor.get_event_recommendation(predictions)        )

            

        return jsonify({    try:

            'input': data,        # Convert request to feature format expected by the model

            'predictions': predictions,        import pandas as pd

            'recommendation': recommendation['recommendation'],        

            'risk_level': recommendation['risk_level'],        # Use the timestamp if provided, otherwise use current time

            'high_risk_conditions': recommendation['high_risk_conditions'],        timestamp = request.timestamp or datetime.now()

            'moderate_risk_conditions': recommendation['moderate_risk_conditions'],        

            'timestamp': datetime.now().isoformat()        # Create feature DataFrame

        })        features = pd.DataFrame({

                    # Weather features

    except Exception as e:            'temperature': [request.features.temperature],

        logging.error(f"Prediction error: {e}")            'humidity': [request.features.humidity],

        return jsonify({            'pressure': [request.features.pressure],

            'error': str(e)            'wind_speed': [request.features.wind_speed],

        }), 500            'wind_direction': [request.features.wind_direction],

            'cloud_cover': [request.features.cloud_cover],

@app.route('/load-models', methods=['POST'])            

def load_models():            # Optional features

    """Load pre-trained models"""            'dew_point': [request.features.dew_point or request.features.temperature - 2.5],

    try:            'uv_index': [request.features.uv_index or 0],

        predictor.load_models()            'visibility': [request.features.visibility or 10],

        return jsonify({            

            'status': 'Models loaded successfully',            # Location features

            'models': list(predictor.models.keys())            'latitude': [request.location.latitude],

        })            'longitude': [request.location.longitude],

    except Exception as e:            

        return jsonify({            # Time features

            'error': str(e)            'month': [timestamp.month],

        }), 500            'day': [timestamp.day],

            'hour': [timestamp.hour],

if __name__ == '__main__':            'month_sin': [float(pd.Series([timestamp.month]).apply(

    print("🌦️ Will It Rain On My Parade? - API Server")                lambda x: np.sin(2 * np.pi * x / 12))[0])

    print("=" * 50)            ],

    print("NASA Space Apps Challenge Weather Prediction API")            'month_cos': [float(pd.Series([timestamp.month]).apply(

    print("=" * 50)                lambda x: np.cos(2 * np.pi * x / 12))[0])

                ],

    # Try to load existing models            'hour_sin': [float(pd.Series([timestamp.hour]).apply(

    try:                lambda x: np.sin(2 * np.pi * x / 24))[0])

        predictor.load_models()            ],

        print(f"✅ Loaded {len(predictor.models)} models")            'hour_cos': [float(pd.Series([timestamp.hour]).apply(

    except:                lambda x: np.cos(2 * np.pi * x / 24))[0])

        print("⚠️ No pre-trained models found. Train models first.")            ],

            })

    print("Starting API server on http://localhost:5000")        

    app.run(debug=True, host='0.0.0.0', port=5000)        # Make predictions
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