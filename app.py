# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Monitor de Humedad", layout="centered")
st.title(" Sistema de Monitoreo de Humedad")

# Cargar modelo y escalador
@st.cache_resource
def load_components():
    with open('modelo_humedad.pkl', 'rb') as f:
        model, scaler = pickle.load(f)
    return model, scaler

model, scaler = load_components()

# Estados posibles (deben coincidir con tu entrenamiento)
ESTADOS = {
    0: " Muy Seco - Riego Urgente",
    1: " Seco - Necesita Riego",
    2: " Óptimo - Condición Ideal",
    3: " Húmedo - No requiere agua",
    4: " Saturado - Riesgo de hongos"
}

# Interfaz principal
def main():
    st.sidebar.header("Configuración")
    
    # Entrada de humedad
    humedad = st.sidebar.number_input(
        "Humedad del suelo (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.1
    )
    
    # Botón de predicción
    if st.sidebar.button("Predecir Estado"):
        # Normalización
        humedad_norm = scaler.transform([[humedad]])
        
        # Predicción
        estado_num = model.predict(humedad_norm)[0]
        estado = ESTADOS.get(estado_num, "Desconocido")
        
        # Visualización
        st.success(f"**Estado del suelo:** {estado}")
        
        # Barra de humedad
        st.progress(humedad/100)
        st.caption(f"Valor medido: {humedad}%")
        
        # Detalles técnicos (opcional)
        with st.expander("Detalles técnicos"):
            st.write(f"Valor normalizado: {humedad_norm[0][0]:.4f}")
            st.write(f"Código de estado: {estado_num}")

if __name__ == "__main__":
    main()