import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib 

app = FastAPI(title="AgroPulse-BD API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = 'agropulse_lstm_model.h5'
DATASET_PATH = 'Agro_pulsedataset(verified).csv'
SCALER_PATH = 'scaler.pkl'

FEATURE_COLUMNS = [
    "NDVI_Clean", "Rain_Sunamganj_Center", "Rain_Meghalaya_Border",
    "Daily_Water_Level", "Soil_Moisture", "Temperature", "Radar_VV"
]

CLASS_NAMES = ["Safe (No Flood)", "At-Risk (Minor Flood)", "Critical (Severe Flood)"]

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(DATASET_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    print("API Ready: StandardScaler and LSTM Model Loaded.")
except Exception as e:
    print(f"Startup Error: {e}")

class DateInput(BaseModel):
    date: str

@app.post("/predict_by_date")
def predict_by_date(data: DateInput):
    try:
        target_date = pd.to_datetime(data.date)
        history = df[df['Date'] <= target_date].tail(7)
        
        if len(history) < 7:
            raise HTTPException(status_code=400, detail="Insufficient history.")

        raw_features = history[FEATURE_COLUMNS].ffill().bfill().fillna(0).values
        scaled_features = scaler.transform(raw_features)
        input_data = np.reshape(scaled_features, (1, 7, 7))
        
        predictions = model.predict(input_data)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        return {
            "status": "Success",
            "requested_date": data.date,
            "prediction": CLASS_NAMES[predicted_class_index],
            "confidence_score": f"{confidence * 100:.2f}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)