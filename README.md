# AgroPulse-BD: Automated Flood Prediction & Crop Monitoring

AgroPulse-BD is an AI-powered environmental decision support system specifically designed for the **Sunamganj Haor region** in Bangladesh. By integrating multi-modal satellite imagery with deep learning time-series models, the system provides impact-based flood forecasting and real-time crop health monitoring to assist local stakeholders and farmers.

---

## 🏗️ System Architecture

The system is built on a decoupled architecture to ensure scalability and efficient processing of heavy geospatial datasets.

1. **Data Ingestion Layer:** Automated fetching of Sentinel-1 (SAR), Sentinel-2 (Optical), and Landsat data via GitHub Actions and cloud APIs.
2. **Backend (FastAPI):** The core engine that handles pre-processing, feature engineering, and model inference.
3. **Model Layer (Bi-LSTM):** A Bidirectional Long Short-Term Memory network trained on 15 years of hydrological and meteorological data to predict water level trends.
4. **Intelligence Layer (Gemini AI):** Utilizes RAG (Retrieval-Augmented Generation) to translate numerical risk probabilities into practical, human-readable agricultural advice.
5. **Presentation Layer (Streamlit):** An interactive dashboard for visualization and data exploration.

---

## 🛰️ How It Works (The Pipeline)

### 1. Data Acquisition & Processing
The system monitors environmental variables including precipitation, river discharge, and satellite-derived indices.
* **SAR (Sentinel-1):** Essential for flood mapping as it penetrates cloud cover during the monsoon season.
* **NDVI (Sentinel-2):** Calculated to monitor crop vigor and identify areas under stress before damage becomes irreversible.

### 2. Predictive Modeling
The **Bi-LSTM model** processes temporal sequences of weather data. Unlike standard LSTMs, the bidirectional approach allows the model to learn patterns from both past and future states in the time series, providing superior accuracy in detecting sudden flash flood onsets.

### 3. AI-Powered Advisory
Once a risk is identified, the backend sends the technical data to **Google Gemini**. The LLM contextualizes the threat based on the current crop cycle and generates advice (e.g., *"High probability of flash flood in 48 hours; prioritize harvesting Boro rice in low-lying sectors"*).

---

## 🖥️ Frontend: Interactive Dashboard
The frontend is built with **Streamlit**, designed to make complex satellite data accessible to non-technical users.

### How the Frontend Works:
* **Reactive State Management:** Streamlit’s execution model allows the dashboard to update dynamically as users toggle between different indices (NDVI vs. SAR) or time ranges.
* **Geospatial Visualization:** Integrated with `folium` to render interactive heatmaps and inundation zones for the Sunamganj region.
* **Asynchronous API Calls:** The frontend communicates with the **FastAPI backend** using the `requests` library, ensuring that heavy model computations don't freeze the user interface.

### Key Features:
* **Real-time Flood Risk Map:** Visualizes current water extent and predicted inundation zones.
* **Crop Health Analytics:** Provides time-series charts of NDVI values to track growth cycles.
* **Impact-Based Advisory Panel:** Displays Gemini-generated alerts in a clear, actionable format.
* **Historical Data Explorer:** Allows comparison of current seasonal trends against historical flood events.

---

## 🚀 Setup & Installation

### Prerequisites
* Python 3.9+
* FastAPI & Uvicorn
* Streamlit
* TensorFlow/PyTorch (for Bi-LSTM inference)

### 1. Clone the Repository
```bash
git clone [https://github.com/bappyBDN/AgroPulse-BD.git](https://github.com/bappyBDN/AgroPulse-BD.git)
cd AgroPulse-BD
