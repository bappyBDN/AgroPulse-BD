import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="AgroPulse-BD 7-Day Forecasting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Updated Configurations ---
MODEL_PATH = 'agropulse_bilstm_model.h5' # Update model name
DATASET_PATH = 'Agro_pulsedataset(verified).csv'
SCALER_PATH = 'scaler_bilstm.pkl' # Update scaler name

FEATURE_COLUMNS = [
    "NDVI_Clean", "Rain_Sunamganj_Center", "Rain_Meghalaya_Border",
    "Daily_Water_Level", "Soil_Moisture", "Temperature", "Radar_VV"
]

CLASS_NAMES = ["Safe (No Flood)", "Warning (Minor Flood)", "Critical (Severe Flood)"]

# Important: Focal Loss requires custom_objects when loading
def sparse_categorical_focal_loss(alpha, gamma=2.0):
    alpha = np.array(alpha, dtype=np.float32)
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=len(alpha))
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        loss = alpha * tf.keras.backend.pow(1 - y_pred, gamma) * (-y_true_one_hot * tf.keras.backend.log(y_pred))
        return tf.keras.backend.sum(loss, axis=-1)
    return focal_loss

try:
    # Load model with custom focal loss
    custom_objects = {"focal_loss": sparse_categorical_focal_loss(alpha=[1.0, 2.0, 5.0])}
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(DATASET_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    print("API Ready: StandardScaler and Bi-LSTM Model Loaded.")
except Exception as e:
    print(f"Startup Error: {e}")

class DateInput(BaseModel):
    date: str

@app.post("/predict_by_date")
def predict_by_date(data: DateInput):
    try:
        target_date = pd.to_datetime(data.date)
        
        # --- MODIFICATION: Fetch 14 days of history instead of 7 ---
        history = df[df['Date'] <= target_date].tail(14)
        
        if len(history) < 14:
            raise HTTPException(status_code=400, detail="Insufficient history. Need 14 days of data.")

        # Preprocessing (Fixed Warnings)
        processed_history = history[FEATURE_COLUMNS].interpolate(method='linear').bfill()
        scaled_features = scaler.transform(processed_history)
        
        # --- MODIFICATION: Reshape for Bi-LSTM (1 batch, 14 timesteps, 7 features) ---
        input_data = np.reshape(scaled_features, (1, 14, 7))
        
        # Inference
        predictions = model.predict(input_data)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Calculate Future Date (Lead Time = 7 Days)
        future_date = target_date + pd.Timedelta(days=7)
        
        return {
            "status": "Success",
            "requested_date": data.date,
            "forecast_target_date": future_date.strftime('%Y-%m-%d'), # Expose the 7-day future date
            "prediction": CLASS_NAMES[predicted_class_index],
            "confidence_score": f"{confidence * 100:.2f}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)