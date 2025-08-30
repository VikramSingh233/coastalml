from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and scaler
model = joblib.load("backend/knn_model.pkl")
scaler = joblib.load("backend/scaler.pkl")

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
        f"&start_hour={start_hour}&end_hour={end_hour}"
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

    merged_data = {}
    for key in marine_resp['hourly']:
        merged_data[key] = marine_resp['hourly'][key][0]
    for key in weather_resp['hourly']:
        merged_data[key] = weather_resp['hourly'][key][0]

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
                scaled = scaler.transform(df)
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
                            scaled = scaler.transform(df)
                            pred = model.predict(scaled)[0]
                            payload["prediction"] = str(pred)
                            batch_predictions.append(payload)
                    elif isinstance(data_json, dict):
                        lat, lon = data_json['latitude'], data_json['longitude']
                        payload = get_merged_api_data(lat, lon)
                        df = pd.DataFrame([payload])
                        scaled = scaler.transform(df)
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
        scaled = scaler.transform(df)
        pred = model.predict(scaled)[0]
        payload["prediction"] = str(pred)
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------ Run Flask ------------------ #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)