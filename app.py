import streamlit as np
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# 1. (Title aur Layout)
st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)

#  UI Styling
st.markdown("""
    <style>
    .main {
        background-color: #f4f7f6;
    }
    .title-text {
        text-align: center;
        color: #2e7d32;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: bold;
    }
    .sub-text {
        text-align: center;
        color: #555555;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
        border-radius: 10px;
        padding: 10px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        color: #e0f2f1;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    }
    .result-box {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2e7d32;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: #1b5e20;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Header UI
st.markdown("<h1 class='title-text'>🌱 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Mitti aur Mausam ke parameters dalein, aur jaaniye sabse behtareen fasal!</p>", unsafe_allow_html=True)

# 3. Saved Model 
@st.cache_resource
def load_model():
    with open('crop_recommendation_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ 'crop_recommendation_model.pkl' file nahi mili! Pehle model train karke save karein.")
    st.stop()

# 4. Input Fields 
st.subheader("📊 Mitti aur Mausam Ki Jankari (Soil & Weather Metrics)")

col1, col2 = st.columns(2)

with col1:
    n = st.number_input("Nitrogen (N) - Mitti mein", min_value=0, max_value=150, value=90, step=1)
    p = st.number_input("Phosphorus (P) - Mitti mein", min_value=0, max_value=150, value=42, step=1)
    k = st.number_input("Potassium (K) - Mitti mein", min_value=0, max_value=210, value=43, step=1)
    ph = st.number_input("Mitti ka pH Level (0-14)", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

with col2:
    temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
    humidity = st.number_input("Humidity (Nami %)", min_value=0.0, max_value=100.0, value=80.0, step=0.5)
    rainfall = st.number_input("Rainfall (Baarish mm)", min_value=0.0, max_value=300.0, value=200.0, step=1.0)

# 5. Prediction Logic aur Result Display
st.markdown("<br>", unsafe_allow_html=True) 

if st.button("🌾 Best Crop Predict Karein"):
    # Input data 
    features = np.array([[n, p, k, temp, humidity, ph, rainfall]])
    
   
    prediction = model.predict(features)
    crop_name = prediction[0].upper()
    
   
    st.balloons()
    
    
    st.markdown(f"""
        <div class='result-box'>
            🎉 Aapke khet ke liye sabse sahi fasal hai: 🌟 {crop_name} 🌟
        </div>
    """, unsafe_allow_html=True)