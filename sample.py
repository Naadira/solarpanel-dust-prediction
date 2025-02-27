
import streamlit as st

st.set_page_config(page_title="Solar Panel Classifier")

import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

# Google Drive direct download link (replace with correct FILE_ID)
GDRIVE_URL = "https://drive.google.com/uc?id=12-ysiXzcHfliSI1QMRx3wfcSQkNusRun"
MODEL_PATH = "model.h5"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model... This may take a while ⏳")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# Load model
st.info("Loading model... 📡")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Streamlit UI

st.markdown(
    "<div style='text-align:center; font-size:30px; font-weight:bold; color:#fff; background:#f44336; padding:10px; border-radius:10px;'>Solar Panel Dust Classifier</div>", 
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def make_prediction(image):
    resize = tf.image.resize(image, (256, 256))
    yhat = model.predict(np.expand_dims(resize/255, 0))
    return yhat[0][0]

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_array = np.array(img)
    prediction = make_prediction(img_array)
    
    result_text = "✅ Clean Panel" if prediction <= 0.5 else "🚨 Dusty Panel"
    result_color = "#4CAF50" if prediction <= 0.5 else "#D32F2F"
    
    st.markdown(f"<div style='text-align:center; font-size:20px; font-weight:bold; padding:15px; background-color:{result_color}; color:white; border-radius:8px;'>{result_text}</div>", unsafe_allow_html=True)


