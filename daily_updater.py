import ee
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import numpy as np
import os
import json

# ==========================================
# 1. Initialization & Setup
# ==========================================
PROJECT_ID = 'my-ai-agent-481120'
JSON_KEY_PATH = 'ee-key.json'  # আপনার ডাউনলোড করা ফাইলের নাম

try:
    # JSON ফাইল থেকে ইমেইল পড়া
    with open(JSON_KEY_PATH) as f:
        key_data = json.load(f)
        service_account_email = key_data['client_email']

    # Service Account দিয়ে ব্রাউজার ছাড়াই লগইন করা
    credentials = ee.ServiceAccountCredentials(service_account_email, JSON_KEY_PATH)
    ee.Initialize(credentials, project=PROJECT_ID)
    print("✅ Earth Engine securely authenticated in background!")
except Exception as e:
    print(f"❌ Authentication Failed: {e}")
    exit() # লগইন ফেইল করলে স্ক্রিপ্ট এখানেই থেমে যাবে

# যে ফাইলটিতে ডেটা আপডেট হবে
CSV_FILE = 'AgroPulse_Test_Dataset_Final.csv'

# ==========================================
# 2. Data Fetching Functions
# ==========================================

def fetch_ffwc_water_level():
    url = "http://old.ffwc.gov.bd/"
    print(f"🌊 Fetching live data from FFWC Legacy Server ({url})...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text_data = soup.get_text(separator=' ')
        match = re.search(r'Sunamganj.*?Water Level\s*:\s*([0-9.]+)', text_data, re.IGNORECASE)
        
        if match:
            wl = float(match.group(1))
            print(f"   ✅ Success! Found Surma Water Level: {wl} meters")
            return wl
        else:
            print("   ❌ Could not extract the Sunamganj data.")
    except Exception as e:
        print(f"   ❌ Connection Error: {e}")
    return np.nan

def fetch_climate_data(target_date_str):
    print("🌡️ Fetching ERA5 Climate Data...")
    roi = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM2_NAME', 'Sunamganj')).geometry()
    crop_mask = ee.Image("ESA/WorldCover/v100/2020").select('Map').eq(40).clip(roi)
    
    climate_col = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
                    .filterBounds(roi).select(['temperature_2m', 'volumetric_soil_water_layer_1'])
                    
    d1 = target_date_str
    d2 = (datetime.strptime(target_date_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    
    try:
        img = climate_col.filterDate(d1, d2).first()
        if img:
            stats = img.updateMask(crop_mask).reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=1000, maxPixels=1e9
            ).getInfo()
            temp_k = stats.get('temperature_2m')
            temp_c = (temp_k - 273.15) if temp_k else np.nan
            sm_val = stats.get('volumetric_soil_water_layer_1')
            return {"Temperature_C": round(temp_c, 2) if temp_c else np.nan, "Soil_Moisture": round(sm_val, 3) if sm_val else np.nan}
    except Exception:
        pass
    return {"Temperature_C": np.nan, "Soil_Moisture": np.nan}

def fetch_rainfall_data(target_date_str):
    print("🌧️ Fetching GPM Rainfall Data...")
    locs = {'Sunamganj_Center': [91.3992, 25.0714], 'Meghalaya_Border': [91.73, 25.27]}
    d_ee = ee.Date(target_date_str)
    row = {}
    try:
        rain_img = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterDate(d_ee, d_ee.advance(1, 'day')).select('precipitation').mean()
        for name, coords in locs.items():
            point = ee.Geometry.Point(coords).buffer(5000)
            stats = rain_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10000).getInfo()
            val = stats.get('precipitation', 0)
            row[f'Rain_{name}'] = (val if val else 0) * 24 
    except Exception:
        row = {'Rain_Sunamganj_Center': np.nan, 'Rain_Meghalaya_Border': np.nan}
    return row

def fetch_satellite_data(target_date_str):
    print("🛰️ Fetching Optical (NDVI) and Radar (VV) Data...")
    roi = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM2_NAME', 'Sunamganj')).geometry()
    crop_mask = ee.Image("ESA/WorldCover/v100/2020").select('Map').eq(40).clip(roi)

    def process_s2(img):
        return img.select(['B8', 'B4'], ['NIR', 'Red']).divide(10000).normalizedDifference(['NIR', 'Red']).rename('NDVI').copyProperties(img, ['system:time_start'])
    
    def process_l8(img):
        return img.select(['SR_B5', 'SR_B4'], ['NIR', 'Red']).multiply(0.0000275).add(-0.2).normalizedDifference(['NIR', 'Red']).rename('NDVI').copyProperties(img, ['system:time_start'])
    
    optical_col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).map(process_s2).merge(
                  ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).map(process_l8))
    
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(roi).filter(ee.Filter.eq('instrumentMode', 'IW')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).select('VV')

    d1 = target_date_str
    # Search within a 1-day window for daily updates
    d2 = (datetime.strptime(target_date_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d') 
    
    opt_val, sar_val = np.nan, np.nan
    try:
        opt_window = optical_col.filterDate(d1, d2)
        if opt_window.size().getInfo() > 0:
            stats = opt_window.qualityMosaic('NDVI').updateMask(crop_mask).clip(roi).reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=30, maxPixels=1e10).getInfo()
            opt_val = stats.get('NDVI', np.nan)
        
        sar_window = s1.filterDate(d1, d2)
        if sar_window.size().getInfo() > 0:
            sar_stats = sar_window.median().updateMask(crop_mask).clip(roi).reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=30, maxPixels=1e10).getInfo()
            sar_val = sar_stats.get('VV', np.nan)
            
    except Exception:
        pass
        
    return {"Mean_NDVI": opt_val, "Radar_VV": sar_val}


# ==========================================
# 3. Execution & Merging Pipeline
# ==========================================

def update_daily_data():
    # 1. আজকের তারিখ নির্ণয়
    # Note: Climate/Rainfall data usually has a 1-2 day lag. Fetching for yesterday is safer.
    target_date = datetime.now() - timedelta(days=1)
    target_date_str = target_date.strftime("%Y-%m-%d")
    
    print(f"\n🚀 Starting Daily AgroPulse Update for: {target_date_str}")
    
    # 2. বিদ্যমান ডেটাসেট রিড করা
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: The file {CSV_FILE} does not exist.")
        return
        
    df = pd.read_csv(CSV_FILE)
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    
    # Check if data already exists for this date to avoid duplicates
    if target_date_str in df['Date'].values:
        print(f"ℹ️ Data for {target_date_str} already exists in the dataset. Exiting.")
        return

    # 3. সব এপিআই থেকে ডেটা কালেক্ট করা
    ffwc_water = fetch_ffwc_water_level()
    climate = fetch_climate_data(target_date_str)
    rain = fetch_rainfall_data(target_date_str)
    satellite = fetch_satellite_data(target_date_str)

    # 4. নতুন ডাটা রো (Row) তৈরি
    new_data = {
        'Date': target_date_str,
        'Temperature': climate.get('Temperature_C', np.nan),     # <-- Changed
        'Soil_Moisture': climate.get('Soil_Moisture', np.nan),
        'Rain_Sunamganj_Center': rain.get('Rain_Sunamganj_Center', np.nan),
        'Rain_Meghalaya_Border': rain.get('Rain_Meghalaya_Border', np.nan),
        'NDVI_Clean': satellite.get('Mean_NDVI', np.nan),        # <-- Changed
        'Radar_VV': satellite.get('Radar_VV', np.nan),
        'Daily_Water_Level': ffwc_water
    }
    
    new_df = pd.DataFrame([new_data])
    
    # 5. মেইন ডেটাসেটের সাথে যুক্ত করা
    df = pd.concat([df, new_df], ignore_index=True)
    
    # 6. Null Values (ফাঁকা ঘর) ইন্টারপোলেট করা
    print("⚙️ Filling null values using Interpolation...")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')
    
    # Linear interpolation for gaps (like satellite which comes every 5 days)
    df = df.interpolate(method='linear', limit_direction='both')
    # Forward and Backward fill for any remaining NaNs at the very edges
    df = df.ffill().bfill()
    
    df = df.reset_index()
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    # 7. সেভ করা
    df.to_csv(CSV_FILE, index=False)
    print(f"🎉 SUCCESS! Database successfully updated and interpolated for {target_date_str}.")
    print("\n--- Latest Database Rows ---")
    print(df.tail(3))

if __name__ == "__main__":
    # ১. প্রথমে প্রতিদিনের নতুন ডেটা কালেক্ট করবে
    try:
        update_daily_data()
    except Exception as e:
        print(f"❌ Error during data update: {e}")
    
    # ২. ডেটা কালেক্ট করা শেষ হলে বা স্কিপ হলেও, স্বয়ংক্রিয়ভাবে সিঙ্ক (Sync) করবে
    print("\n==================================================")
    print("🚀 Initiating Auto-Sync to Verified Database...")
    try:
        # os.system ব্যবহার করা বেশি নিরাপদ, এটি সরাসরি ফাইলটিকে আলাদাভাবে রান করে
        exit_code = os.system('python sync_data.py')
        if exit_code == 0:
            print("✅ Auto-Sync Completed Successfully.")
        else:
            print("❌ Auto-Sync Failed (Check sync_data.py for errors).")
    except Exception as e:
        print(f"❌ System Error during Auto-Sync: {e}")
    print("==================================================\n")