from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import os
import numpy as np
from sklearn.impute import SimpleImputer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model, scaler, and create imputer
model = joblib.load("backend/knn_model.pkl")
scaler = joblib.load("backend/scaler.pkl")
imputer = SimpleImputer(strategy='mean')  # Use mean imputation for missing values

# Define feature name mapping (without units -> with units)
FEATURE_MAPPING = {
    "wave_height": "wave_height (m)",
    "wave_direction": "wave_direction (°)",
    "wave_period": "wave_period (s)",
    "sea_level_height_msl": "sea_level_height_msl (m)",
    "sea_surface_temperature": "sea_surface_temperature (°C)",
    "ocean_current_direction": "ocean_current_direction (°)",
    "ocean_current_velocity": "ocean_current_velocity (km/h)",
    "swell_wave_direction": "swell_wave_direction (°)",
    "swell_wave_period": "swell_wave_period (s)",
    "temperature_2m": "temperature_2m (°C)",
    "relative_humidity_2m": "relative_humidity_2m (%)",
    "precipitation": "precipitation (mm)",
    "weather_code": "weather_code (wmo code)",
    "pressure_msl": "pressure_msl (hPa)",
    "surface_pressure": "surface_pressure (hPa)",
    "wind_speed_10m": "wind_speed_10m (km/h)",
    "wind_direction_10m": "wind_direction_10m (°)",
    "wind_direction_100m": "wind_direction_100m (°)"
}

# ------------------ Helper Function ------------------ #
def get_merged_api_data(lat, lon):
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    one_hour_later = now + timedelta(hours=1)
    start_hour = now.isoformat() + "Z"
    end_hour = one_hour_later.isoformat() + "Z"

    # Marine API
    marine_url = (
        f"https://marine-api.open-meteo.com/v1/marine?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=wave_height,wave_direction,wave_period,sea_level_height_msl,"
        f"sea_surface_temperature,ocean_current_direction,ocean_current_velocity,"
        f"swell_wave_direction,swell_wave_period"
        f"&start={start_hour}&end={end_hour}"
    )
    marine_resp = requests.get(marine_url).json()

    # Weather API
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relative_humidity_2m,precipitation,weather_code,"
        f"pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,wind_direction_100m"
        f"&start={start_hour}&end={end_hour}"
    )
    weather_resp = requests.get(weather_url).json()

    if "hourly" not in marine_resp or "hourly" not in weather_resp:
        raise ValueError(f"API Error: {marine_resp} | {weather_resp}")

    merged_data = {}

    # Extract first hour's data and map feature names
    for key, mapped_key in FEATURE_MAPPING.items():
        if key in marine_resp["hourly"]:
            value = marine_resp["hourly"][key][0] if marine_resp["hourly"][key] else 0
            merged_data[mapped_key] = value
        elif key in weather_resp["hourly"]:
            value = weather_resp["hourly"][key][0] if weather_resp["hourly"][key] else 0
            merged_data[mapped_key] = value
        else:
            # If the feature is not in either response, set a default value
            merged_data[mapped_key] = 0

    return merged_data

# ------------------ Web Frontend ------------------ #
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    batch_predictions = None

    if request.method == "POST":
        # Manual input
        if "latitude" in request.form and "longitude" in request.form:
            try:
                lat = float(request.form["latitude"])
                lon = float(request.form["longitude"])
                payload = get_merged_api_data(lat, lon)
                df = pd.DataFrame([payload])
                # Ensure column order matches training
                df = df.reindex(columns=FEATURE_MAPPING.values(), fill_value=0)
                
                # Handle missing values
                df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
                
                scaled = scaler.transform(df_imputed)
                pred = model.predict(scaled)[0]
                payload["prediction"] = str(pred)
                prediction = payload
            except Exception as e:
                prediction = {"error": str(e)}

        # JSON file upload
        elif "file" in request.files:
            file = request.files["file"]
            if file.filename.endswith(".json"):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                with open(filepath) as f:
                    data_json = json.load(f)
                batch_predictions = []
                try:
                    if isinstance(data_json, list):
                        for item in data_json:
                            lat, lon = item['latitude'], item['longitude']
                            payload = get_merged_api_data(lat, lon)
                            df = pd.DataFrame([payload])
                            df = df.reindex(columns=FEATURE_MAPPING.values(), fill_value=0)
                            
                            # Handle missing values
                            df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
                            
                            scaled = scaler.transform(df_imputed)
                            pred = model.predict(scaled)[0]
                            payload["prediction"] = str(pred)
                            batch_predictions.append(payload)
                    elif isinstance(data_json, dict):
                        lat, lon = data_json['latitude'], data_json['longitude']
                        payload = get_merged_api_data(lat, lon)
                        df = pd.DataFrame([payload])
                        df = df.reindex(columns=FEATURE_MAPPING.values(), fill_value=0)
                        
                        # Handle missing values
                        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
                        
                        scaled = scaler.transform(df_imputed)
                        pred = model.predict(scaled)[0]
                        payload["prediction"] = str(pred)
                        batch_predictions.append(payload)
                except Exception as e:
                    batch_predictions = {"error": str(e)}
    
    return render_template("index.html", prediction=prediction, batch_predictions=batch_predictions)

# ------------------ API Endpoint ------------------ #
@app.route("/api/predict_from_coords", methods=["GET"])
def api_predict():
    try:
        latitude = float(request.args.get("latitude"))
        longitude = float(request.args.get("longitude"))
    except:
        return jsonify({"error": "Please provide valid latitude and longitude"}), 400

    try:
        payload = get_merged_api_data(latitude, longitude)
        df = pd.DataFrame([payload])
        df = df.reindex(columns=FEATURE_MAPPING.values(), fill_value=0)
        
        # Handle missing values
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        scaled = scaler.transform(df_imputed)
        pred = model.predict(scaled)[0]
        payload["prediction"] = str(pred)
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------ Run Flask ------------------ #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)