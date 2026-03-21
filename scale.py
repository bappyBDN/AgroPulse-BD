import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load dataset
df = pd.read_csv(r'F:\Agro_PalseBD\Agro_pulsedataset(verified).csv')

# 2. Use the EXACT order from your training script
FEATURE_COLUMNS = [
    "NDVI_Clean",
    "Rain_Sunamganj_Center",
    "Rain_Meghalaya_Border",
    "Daily_Water_Level",
    "Soil_Moisture",
    "Temperature",
    "Radar_VV"
]

# 3. Pre-process exactly like training (Drop NaNs before fitting)
target_col = "Final_AgroPulse_Level"
df_clean = df.dropna(subset=FEATURE_COLUMNS + [target_col])

# 4. Fit the StandardScaler
scaler = StandardScaler()
scaler.fit(df_clean[FEATURE_COLUMNS])

# 5. Save it
joblib.dump(scaler, r'F:\Agro_PalseBD\scaler.pkl')
print("Success! StandardScaler saved to F:\Agro_PalseBD\scaler.pkl")