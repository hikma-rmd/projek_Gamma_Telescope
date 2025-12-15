import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==============================
# LOAD MODEL & SCALER
# ==============================
scaler = joblib.load("scaler_magic.pkl")
rf_model = joblib.load("rf_magic_model.pkl")
xgb_model = joblib.load("xgb_magic_model.pkl")

# ==============================
# STREAMLIT CONFIG
# ==============================
st.set_page_config(page_title="MAGIC Gamma Classification", layout="wide")
st.title("🔭 MAGIC Gamma Telescope Classification")
st.write("Aplikasi klasifikasi **Gamma vs Hadron** menggunakan Machine Learning")

# ==============================
# PILIH MODEL
# ==============================
model_choice = st.selectbox(
    "Pilih Model",
    ("Random Forest", "XGBoost")
)

# ==============================
# INPUT FITUR
# ==============================
st.subheader("Masukkan Nilai Fitur")

fLength = st.number_input("fLength", value=50.0)
fWidth = st.number_input("fWidth", value=30.0)
fSize = st.number_input("fSize", value=100.0)
fConc = st.number_input("fConc", value=0.2)
fConc1 = st.number_input("fConc1", value=0.1)
fAsym = st.number_input("fAsym", value=0.0)
fM3Long = st.number_input("fM3Long", value=0.0)
fM3Trans = st.number_input("fM3Trans", value=0.0)
fAlpha = st.number_input("fAlpha", value=30.0)
fDist = st.number_input("fDist", value=100.0)

input_data = np.array([[fLength, fWidth, fSize, fConc, fConc1,
                        fAsym, fM3Long, fM3Trans, fAlpha, fDist]])

# ==============================
# PREDIKSI
# ==============================
if st.button("🔍 Prediksi"):
    input_scaled = scaler.transform(input_data)

    if model_choice == "Random Forest":
        prediction = rf_model.predict(input_scaled)[0]
    else:
        prediction = xgb_model.predict(input_scaled)[0]

    label = "Gamma" if prediction == 1 else "Hadron"

    st.success(f"Hasil Prediksi: **{label}**")