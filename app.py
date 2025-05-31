import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Cargar modelo y scaler
model = load_model("modelo_abandono.h5")
scaler = joblib.load("scaler.pkl")

st.title("🔍 Predicción de Abandono Universitario")

st.markdown("Completa el siguiente formulario con tus datos:")

with st.form("formulario"):
    estudios_previos = st.selectbox("¿Usted tiene estudios universitarios?", ["Sí", "No"])
    inscrito = st.selectbox("¿Está inscrito en la Universidad actualmente?", ["Sí", "No"])
    reprobado = st.selectbox("¿Ha reprobado alguna materia?", ["Sí", "No"])
    solvente = st.selectbox("¿Está solvente actualmente con la Universidad?", ["Sí", "No"])
    empleo = st.selectbox("¿Tienes empleo actualmente?", ["Sí", "No"])
    traslado = st.slider("¿Cuánto tiempo tardas en llegar a la universidad? (en horas)", 0.0, 5.0, 1.0, step=0.5)

    submit = st.form_submit_button("Predecir")

if submit:
    # Codificar respuestas
    map_si_no = {"Sí": 1, "No": 0}

    datos = np.array([[
        map_si_no[estudios_previos],
        map_si_no[inscrito],
        map_si_no[reprobado],
        map_si_no[solvente],
        map_si_no[empleo],
        traslado
    ]])

    # Escalar datos
    datos_escalados = scaler.transform(datos)

    # Predecir
    prob = model.predict(datos_escalados)[0][0]

    st.write(f"📊 Probabilidad de abandono: **{prob:.2%}**")

    if prob > 0.5:
        st.error("🔴 Riesgo ALTO de abandono universitario.")
    else:
        st.success("🟢 Riesgo BAJO de abandono universitario.")
