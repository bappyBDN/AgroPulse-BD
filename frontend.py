import streamlit as st
import requests
from datetime import datetime, timedelta
import google.generativeai as genai
import ee
import folium
from streamlit_folium import st_folium
import json
import pandas as pd
import plotly.express as px

# ==========================================
# --- Configuration & Initialization ---
# ==========================================

# Gemini Configuration
# Gemini Configuration (Secured)
gemini_api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=gemini_api_key)
model_gemini = genai.GenerativeModel('gemini-2.5-flash')

# Earth Engine Authentication
PROJECT_ID = 'my-ai-agent-481120'
JSON_KEY_PATH = 'ee-key.json'

@st.cache_resource
def init_earth_engine():
    try:
        with open(JSON_KEY_PATH) as f:
            key_data = json.load(f)
            service_account_email = key_data['client_email']
        credentials = ee.ServiceAccountCredentials(service_account_email, JSON_KEY_PATH)
        ee.Initialize(credentials, project=PROJECT_ID)
        return True
    except Exception as e:
        st.error(f"❌ Earth Engine Authentication Failed: {e}. Please check your ee-key.json file.")
        return False

ee_initialized = init_earth_engine()

# ==========================================
# --- Helper Functions ---
# ==========================================

def get_ai_advice(risk_level_name, confidence, date):
    level_logic = ""
    if "Safe" in risk_level_name:
        level_logic = "বন্যা নেই এবং ফসল নিরাপদ।"
    elif "At-Risk" in risk_level_name:
        level_logic = "বন্যা শুরু হচ্ছে, তবে ফসল এখন বর্ধনশীল (Growing) অথবা কাটার পরবর্তী (Post-harvest) অবস্থায় আছে। এটি একটি মাইনর ঝুঁকি।"
    elif "Critical" in risk_level_name:
        level_logic = "বন্যা আসন্ন এবং ফসল এখন পুরোপুরি পেকেছে (Mature Crop)। এটি অত্যন্ত ঝুঁকিপূর্ণ অবস্থা!"

    prompt = f"""
    You are an AI Agricultural Consultant for Sunamganj Haor areas. 
    Current Prediction: {risk_level_name} (Confidence: {confidence})
    Date: {date}
    Context: {level_logic}
    
    Instructions:
    - Provide advice in simple, encouraging Bengali for a farmer.
    - If it's Level 1 (At-Risk), advise them on how to protect growing crops or manage post-harvest grains.
    - If it's Level 2 (Critical - Mature Crop), strongly urge them to harvest immediately (১০০% পাকলে দেরি না করে কেটে ফেলা).
    - If Level 0, give routine maintenance advice.
    """
    
    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"এআই পরামর্শ লোড করতে সমস্যা হয়েছে: {e}"

def add_ee_layer(self, ee_image_object, vis_params, name):
    try:
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        folium.raster_layers.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr='Map Data © Google Earth Engine',
            name=name,
            overlay=True,
            control=True
        ).add_to(self)
    except Exception as e:
        print(f"Error adding EE layer: {e}")

folium.Map.add_ee_layer = add_ee_layer

# ==========================================
# --- Streamlit UI Setup ---
# ==========================================
st.set_page_config(page_title="AgroPulse-BD AI", layout="wide", page_icon="🌾")
st.title("🌾 AgroPulse-BD: AI Advisor & Crop Health Dashboard")

if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None
if "show_ndvi_map" not in st.session_state:
    st.session_state.show_ndvi_map = False

# Sidebar
# Sidebar
st.sidebar.header("Control Panel")

# আজকের তারিখ থেকে ১ দিন বিয়োগ করে গতকালের তারিখ বের করা
yesterday = datetime.now() - timedelta(days=2)

# ডিফল্ট ভ্যালু হিসেবে yesterday সেট করা হলো
selected_date = st.sidebar.date_input("Select Date for Analysis", yesterday)

st.markdown("---")

# ==========================================
# --- Top Row Layout (Columns) ---
# ==========================================

col1, col2 = st.columns([4, 6], gap="large")

# -----------------------------------
# Left Column: Section 1 (AI Prediction)
# -----------------------------------
# -----------------------------------
# Left Column: Section 1 (AI Prediction)
# -----------------------------------
with col1:
    st.markdown("### 🤖 AI Flood Prediction")
    
    # বাটনটির নাম একটু পরিবর্তন করে দিলাম
    if st.button("📊 Get 7-Day Forecast", type="primary", use_container_width=True):
        API_URL = "http://127.0.0.1:8000/predict_by_date"
        payload = {"date": str(selected_date)}
        
        try:
            with st.spinner('Calculating Bi-LSTM sequence...'):
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    prediction = result['prediction']
                    score = result['confidence_score']
                    forecast_date = result['forecast_target_date']
                    
                    # এখানে AI Advice সাথে সাথে কল না করে শুধু মডেলের ডেটা সেভ রাখছি
                    st.session_state.prediction_data = {
                        "date": selected_date,
                        "forecast_date": forecast_date,
                        "prediction": prediction,
                        "score": score,
                        "advice": None # শুরুতে অ্যাডভাইস ফাঁকা থাকবে
                    }
                else:
                    st.error(f"Error from API: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(" Cannot connect to the Prediction API. Is FastAPI running?")
        except Exception as e:
            st.error(f"Error: {e}")

    # রেজাল্ট দেখানোর অংশ
    if st.session_state.prediction_data:
        data = st.session_state.prediction_data
        risk_color = "green" if "Safe" in data['prediction'] else ("orange" if "Warning" in data['prediction'] else "red")
        
        # একটি সুন্দর বক্সে রেজাল্ট দেখানো
        st.markdown(f"""
        <div style="padding: 15px; border-radius: 10px; border: 2px solid {risk_color}; background-color: rgba(255,255,255,0.05);">
            <h4 style="margin-top:0;">7-Day Forecast Risk: <span style='color:{risk_color}'>{data['prediction']}</span></h4>
            <p style="margin-bottom:0;"><b>Target Date:</b> {data['forecast_date']} (Based on {data['date']}) | <b>Confidence:</b> {data['score']}</p>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        # --- NEW LOGIC: Tips & Details Button ---
        # যদি অ্যাডভাইস আগে থেকে জেনারেট করা না থাকে, তবে বাটনটি দেখাবে
        if data['advice'] is None:
            if st.button("💡 Tips & Details (কৃষি পরামর্শ)", use_container_width=True):
                with st.spinner('🤖 Gemini is writing advice in Bangla...'):
                    # বাটন ক্লিক করার পরেই কেবল LLM-কে কল করা হবে
                    ai_advice = get_ai_advice(data['prediction'], data['score'], data['forecast_date'])
                    st.session_state.prediction_data['advice'] = ai_advice
                    st.rerun() # UI আপডেট করার জন্য রিরান
        else:
            # একবার জেনারেট হয়ে গেলে বারবার টোকেন খরচ না করে সরাসরি মেসেজটা দেখাবে
            st.info(data['advice'])

# -----------------------------------
# Right Column: Section 2 (NDVI Map)
# -----------------------------------
# -----------------------------------
# Right Column: Section 2 (NDVI Map)
# -----------------------------------
with col2:
    st.markdown("### 🛰️ Crop Health (NDVI Map)")
    st.write("🟢 **Green**: Healthy | 🟡 **Yellow**: Weak/Harvested | 🔴 **Red**: Water/Bare")
    
    if st.button("🗺️ Load / Hide Satellite Map", use_container_width=True):
        st.session_state.show_ndvi_map = not st.session_state.show_ndvi_map

    if st.session_state.show_ndvi_map:
        if not ee_initialized:
            st.warning("Earth Engine is not authenticated.")
        else:
            with st.spinner("Processing satellite imagery (This may take a few seconds)..."):
                try:
                    lat, lon = 25.0714, 91.3992
                    district = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM2_NAME', 'Sunamganj'))

                    # পরিবর্তন ১: মেঘের কারণে ৭ দিনের বদলে ৩০ দিনের ইমেজ উইন্ডো নেওয়া হলো
                    end_date_obj = selected_date + timedelta(days=1)
                    start_date_obj = selected_date - timedelta(days=30) 

                    image = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
                        .filterBounds(district) \
                        .filterDate(start_date_obj.strftime("%Y-%m-%d"), end_date_obj.strftime("%Y-%m-%d")) \
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)) \
                        .median().clip(district)

                    if image.bandNames().size().getInfo() > 0:
                        # পরিবর্তন ২: WorldCover থেকে সঠিকভাবে 'Map' ব্যান্ড সিলেক্ট করা
                        crop_mask = ee.Image("ESA/WorldCover/v100/2020").select('Map').eq(40)
                        masked_satellite = image.updateMask(crop_mask)
                        
                        ndvi = masked_satellite.normalizedDifference(['B8', 'B4']).rename('NDVI')

                        m = folium.Map(location=[lat, lon], zoom_start=10)
                        
                        vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000, 'gamma': 1.4}
                        ndvi_vis = {'min': 0, 'max': 0.8, 'palette': ['#e74c3c', '#f1c40f', '#2ecc71']}
                        
                        # পরিবর্তন ৩: আনমাস্কড ম্যাপ ব্যাকগ্রাউন্ডে রেখে, তার ওপর NDVI লেয়ার বসানো
                        m.add_ee_layer(image, vis_params, 'True Color (Background)')
                        m.add_ee_layer(ndvi, ndvi_vis, 'NDVI Health (Croplands Only)')
                        
                        m.add_child(folium.LayerControl())

                        st_folium(m, height=450, use_container_width=True)
                    else:
                        st.error("❌ No clear satellite imagery available due to clouds.")
                except Exception as e:
                    st.error(f"Map Error: {e}")

# ==========================================
# --- Bottom Row: Section 3 (Analytics) ---
# ==========================================
st.markdown("---")
st.markdown("### 📈 Section 3: Hydrological & Rainfall Analytics")

@st.cache_data
def load_historical_data():
    try:
        # আপনার গিটহাব রিপোজিটরির Raw URL টা এখানে বসাবেন
        # উদাহরণ: 'https://raw.githubusercontent.com/YourUsername/YourRepoName/main/AgroPulse_Test_Dataset_Final.csv'
        CSV_URL = 'https://raw.githubusercontent.com/bappyBDN/AgroPulse-BD/main/Agro_pulsedataset(verified).csv'
        df = pd.read_csv(CSV_URL)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        return None

df_history = load_historical_data()

if df_history is not None:
   
    bottom_col1, bottom_col2 = st.columns([3, 7], gap="large")
    
    end_dt = pd.to_datetime(selected_date)
    
    with bottom_col1:
        st.markdown(f"**Last 5 Days Condition**")
        start_dt_5d = end_dt - timedelta(days=4)
        recent_data = df_history[(df_history['Date'] >= start_dt_5d) & (df_history['Date'] <= end_dt)].copy()
        
        if not recent_data.empty:
            display_df = recent_data[['Date', 'Daily_Water_Level', 'Rain_Sunamganj_Center']].copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%b %d')
            display_df.rename(columns={'Daily_Water_Level': 'Water (m)', 'Rain_Sunamganj_Center': 'Rain (mm)'}, inplace=True)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No data available.")

    with bottom_col2:
        st.markdown(f"**1-Month Water Level Trend**")
        month_start_dt = end_dt - timedelta(days=30)
        month_data = df_history[(df_history['Date'] >= month_start_dt) & (df_history['Date'] <= end_dt)].copy()
        
        if not month_data.empty:
            fig = px.line(
                month_data, x='Date', y='Daily_Water_Level', 
                labels={'Daily_Water_Level': 'Water Level (m)', 'Date': ''},
                markers=True, height=350
            )
            fig.add_hline(y=7.5, line_dash="dash", line_color="red", annotation_text="Danger (7.5m)", annotation_position="top left")
            fig.update_layout(hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0))
            fig.update_traces(line_color='#3498db')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical data available.")
