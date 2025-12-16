import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# LOAD MODEL, SCALER, METRICS
# ==============================
scaler = joblib.load("scaler_magic.pkl")
rf_model = joblib.load("rf_magic_model.pkl")
xgb_model = joblib.load("xgb_magic_model.pkl")

# hasil evaluasi model (data uji)
metrics = {
    "rf": {
        "accuracy": 0.8867,
        "confusion_matrix": [
            [2330, 136],
            [295, 1043]
        ]
    },
    "xgb": {
        "accuracy": 0.8850,
        "confusion_matrix": [
            [2337, 129],
            [307, 1031]
        ]
    }
}

# mapping label sesuai dataset
label_map = {0: "Gamma", 1: "Hadron"}

# ==============================
# STREAMLIT CONFIG
# ==============================
st.set_page_config(
    page_title="MAGIC Gamma Classification",
    layout="wide"
)

st.title("🔭 MAGIC Gamma Telescope Classification")
st.write(
    """
    Aplikasi ini digunakan untuk mengklasifikasikan **peristiwa partikel**
    menjadi **Gamma** atau **Hadron** menggunakan model Machine Learning.
    """
)

# ==============================
# SIDEBAR - PILIH MODEL
# ==============================
st.sidebar.header("⚙️ Pengaturan Model")

model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ("Random Forest", "XGBoost")
)

# ==============================
# INPUT FITUR
# ==============================
st.subheader("🧪 Masukkan Nilai Fitur")

col1, col2 = st.columns(2)

with col1:
    fLength = st.number_input("fLength", value=50.0)
    fWidth = st.number_input("fWidth", value=30.0)
    fSize = st.number_input("fSize", value=100.0)
    fConc = st.number_input("fConc", value=0.2)
    fConc1 = st.number_input("fConc1", value=0.1)

with col2:
    fAsym = st.number_input("fAsym", value=0.0)
    fM3Long = st.number_input("fM3Long", value=0.0)
    fM3Trans = st.number_input("fM3Trans", value=0.0)
    fAlpha = st.number_input("fAlpha", value=30.0)
    fDist = st.number_input("fDist", value=100.0)

input_data = np.array([[
    fLength, fWidth, fSize, fConc, fConc1,
    fAsym, fM3Long, fM3Trans, fAlpha, fDist
]])

# ==============================
# PREDIKSI
# ==============================
st.divider()

if st.button("🔍 Prediksi"):
    # scaling
    input_scaled = scaler.transform(input_data)

    # pilih model & metrik
    if model_choice == "Random Forest":
        model = rf_model
        acc = metrics["rf"]["accuracy"]
        cm = np.array(metrics["rf"]["confusion_matrix"])
    else:
        model = xgb_model
        acc = metrics["xgb"]["accuracy"]
        cm = np.array(metrics["xgb"]["confusion_matrix"])

    # prediksi
    prediction = model.predict(input_scaled)[0]
    label = label_map[prediction]

    # ==============================
    # OUTPUT PREDIKSI
    # ==============================
    st.subheader("📌 Hasil Prediksi")

    st.success(f"Hasil Klasifikasi: **{label}**")

    # ==============================
    # INFORMASI PERFORMA MODEL
    # ==============================
    st.subheader("📊 Performa Model (Data Uji)")

    st.info(
        f"""
        **Akurasi Model:** {acc:.2%}  
        *(Akurasi dihitung dari data uji, bukan dari input yang baru diprediksi)*
        """
    )

    # ==============================
    # CONFUSION MATRIX
    # ==============================
    st.subheader("🧮 Confusion Matrix (Data Uji)")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Gamma", "Hadron"],
        yticklabels=["Gamma", "Hadron"],
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"Confusion Matrix - {model_choice}")

    st.pyplot(fig)

# ==============================
# FOOTER
# ==============================
st.divider()
st.caption(
    "Catatan: Akurasi dan confusion matrix ditampilkan sebagai hasil evaluasi model "
    "berdasarkan data uji, bukan berdasarkan input pengguna."
)

