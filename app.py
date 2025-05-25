import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Cargar modelo y scaler
model = load_model("modelo_abandono.h5")
scaler = joblib.load("scaler.pkl")

st.title("Predicción de Abandono Académico")

# Inputs del usuario
with st.form("formulario"):
    asistencia = st.selectbox("¿Asiste regularmente a clases dominicales?", ["Sí", "No"])
    tareas = st.selectbox("¿Con qué frecuencia entrega sus tareas a tiempo?", ["Siempre", "A veces", "Rara vez", "Nunca"])
    reprobadas = st.number_input("¿Cuántas materias ha reprobado?", min_value=0)
    traslado = st.number_input("¿Cuántos minutos tarda en llegar a la universidad?", min_value=1)
    trabajo = st.selectbox("¿Tiene empleo actualmente?", ["Sí", "No"])
    estres = st.selectbox("Nivel de estrés académico:", ["Bajo", "Medio", "Alto"])
    sueno = st.number_input("¿Cuántas horas duerme al día?", min_value=3, max_value=10)
    solvente = st.selectbox("¿Está solvente con la universidad?", ["Sí", "No"])
    tiene_trabajo = st.selectbox("¿Trabaja actualmente?", ["Sí", "No"])
    
    submit = st.form_submit_button("Predecir")

if submit:
    map_bin = {"Sí": 1, "No": 0}
    map_tareas = {"Siempre": 1.0, "A veces": 0.7, "Rara vez": 0.4, "Nunca": 0.0}
    map_estres = {"Bajo": 0.2, "Medio": 0.5, "Alto": 0.8}

    datos = np.array([[ 
        map_bin[asistencia],
        map_tareas[tareas],
        reprobadas,
        traslado,
        map_bin[trabajo],
        map_estres[estres],
        sueno,
        map_bin[solvente],
        map_bin[tiene_trabajo]
    ]])
    
    datos_escalados = scaler.transform(datos)
    prob = model.predict(datos_escalados)[0][0]

    st.write(f"📊 Probabilidad de abandono: **{prob:.2%}**")
    if prob > 0.5:
        st.error("🔴 Riesgo ALTO de abandono")
    else:
        st.success("🟢 Riesgo BAJO de abandono")
