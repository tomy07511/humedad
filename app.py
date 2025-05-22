# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(page_title="Monitor de Humedad", layout="centered")
st.title("💧 Sistema Inteligente de Riego")

# Estados posibles (4 categorías)
ESTADOS = [
    "🌵 Muy Seco (Riego urgente)",  # Índice 0
    "☀️ Seco (Necesita agua)",      # Índice 1
    "🌱 Óptimo (Buen estado)",      # Índice 2
    "⚠️ Saturado (Riesgo de hongos)" # Índice 3
]

# Cargar modelo y escalador
@st.cache_resource
def load_model():
    try:
        with open('modelo_humedad.pkl', 'rb') as f:
            saved_data = pickle.load(f)
            return saved_data['model'], saved_data['scaler']
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.stop()

model, scaler = load_model()

# Interfaz principal
def main():
    st.sidebar.header("Parámetros de Entrada")
    
    humedad = st.sidebar.slider(
        "Humedad del suelo (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=45.0,
        step=0.1
    )
    
    if st.sidebar.button("🔍 Analizar Estado"):
        try:
            # Normalización y predicción
            humedad_norm = scaler.transform([[humedad]])
            pred = model.predict(humedad_norm)
            
            # Obtener el índice de la clase predicha
            class_index = np.argmax(pred) if pred.ndim > 1 else pred[0]
            estado = ESTADOS[class_index]
            
            # Mostrar resultados
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Humedad Actual", f"{humedad}%")
                st.progress(humedad/100)
            
            with col2:
                st.metric("Estado Predicho", estado)
                
            # Recomendación basada en el estado
            st.subheader("📋 Recomendación")
            if class_index == 0:
                st.warning("🔴 Regar inmediatamente - El suelo está extremadamente seco")
            elif class_index == 1:
                st.info("🟡 Regar pronto - El suelo está comenzando a secarse")
            elif class_index == 2:
                st.success("🟢 Condición perfecta - Mantener monitoreo")
            else:
                st.error("🔵 Detener riego - Suelo sobresaturado")
                
        except Exception as e:
            st.error(f"Error en la predicción: {str(e)}")

if __name__ == "__main__":
    main()
