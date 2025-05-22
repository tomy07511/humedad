# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(page_title="Monitor de Humedad", layout="centered")
st.title(" Sistema Inteligente de Riego")

# ORDEN CORRECTO DEFINIDO MANUALMENTE (como debe ser)
ESTADOS_CORRECTOS = [
    " Muy Seco (Riego urgente)",  # Índice 0
    " Seco (Necesita agua)",      # Índice 1
    " Óptimo (Buen estado)",      # Índice 2
    " Saturado (Riesgo de hongos)" # Índice 3
]

# Cargar modelo y escalador
@st.cache_resource
def load_model():
    try:
        with open('modelo_humedad.pkl', 'rb') as f:
            saved_data = pickle.load(f)
            ‎app.py


            # Mostrar el orden REAL de clases del modelo para diagnóstico
            if hasattr(saved_data['model'], 'classes_'):
                           st.warning(f" {saved_data['model'].classes_}")
            return saved_data['model'], saved_data['scaler']
            
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.stop()

model, scaler = load_model()

def corregir_orden_prediccion(pred_proba):
    """Corrige el orden invertido entre Óptimo y Muy Seco"""
    # Asumiendo que el modelo tiene: [Óptimo, Saturado, Seco, Muy Seco]
    # Y nosotros queremos: [Muy Seco, Seco, Óptimo, Saturado]
    return np.array([pred_proba[0], pred_proba[2], pred_proba[3], pred_proba[1]])

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
            # Normalización
            humedad_norm = scaler.transform([[humedad]])
            
            # Obtener probabilidades
            pred_proba = model.predict_proba(humedad_norm)[0]
            
            # Aplicar corrección de orden
            prob_corregidas = corregir_orden_prediccion(pred_proba)
            class_index = np.argmax(prob_corregidas)
            estado = ESTADOS_CORRECTOS[class_index]
            
            # Mostrar resultados
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Humedad Actual", f"{humedad}%")
                st.progress(humedad/100)
            
            with col2:
                st.metric("Estado Predicho", estado)
                
            # Mostrar diagnóstico
            with st.expander("🔍 Detalles técnicos"):
                st.write("Probabilidades originales:", dict(zip(model.classes_, pred_proba)))
                st.write("Probabilidades corregidas:", dict(zip(ESTADOS_CORRECTOS, prob_corregidas)))
                st.write("Índice seleccionado:", class_index)
                
            # Recomendación
            st.subheader("📋 Recomendación")
            recomendaciones = [
                " Regar inmediatamente - Suelo extremadamente seco",
                " Regar pronto - Suelo comenzando a secarse",
                " Condición perfecta - Mantener monitoreo",
                " Detener riego - Suelo sobresaturado"
            ]
            st.warning(recomendaciones[class_index])
                
        except Exception as e:
            st.error(f"Error en la predicción: {str(e)}")

if __name__ == "__main__":
    main()
