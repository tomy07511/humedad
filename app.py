# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(page_title="Monitor de Humedad", layout="centered")
st.title("💧 Sistema Inteligente de Riego")

# Definir el ORDEN CORRECTO que deseas (0: Muy Seco, 1: Seco, 2: Óptimo, 3: Saturado)
ESTADOS = [
    "🌵 Muy Seco (Riego urgente)",
    "☀️ Seco (Necesita agua)",
    "🌱 Óptimo (Buen estado)",
    "⚠️ Saturado (Riesgo de hongos)"
]

# Cargar modelo y escalador
@st.cache_resource
def load_model():
    try:
        with open('modelo_humedad.pkl', 'rb') as f:
            saved_data = pickle.load(f)
            
            # Verificar el orden de clases que usa el modelo
            if hasattr(saved_data['model'], 'classes_'):
                st.write("⚠️ Orden de clases en el modelo:", saved_data['model'].classes_)
            
            return saved_data['model'], saved_data['scaler']
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.stop()

model, scaler = load_model()

def corregir_prediccion(pred_proba):
    """Corrige el orden de las predicciones según nuestro orden deseado"""
    # Si el modelo tiene otro orden (ej: ['Óptimo', 'Saturado', 'Seco', 'Muy Seco'])
    # Creamos un mapeo para reordenar las probabilidades
    
    # EJEMPLO DE MApeo (debes ajustar según el orden REAL de tu modelo):
    # mapeo = [3, 2, 0, 1]  # Esto es solo un ejemplo, debes ver el orden real
    
    # Para este caso, asumamos que el modelo tiene este orden:
    # 0: Óptimo, 1: Saturado, 2: Seco, 3: Muy Seco
    mapeo = [3, 2, 0, 1]  # Mapeo al orden deseado
    
    return np.array([pred_proba[i] for i in mapeo])

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
            
            # Obtener probabilidades de cada clase
            pred_proba = model.predict_proba(humedad_norm)[0]
            
            # Corregir el orden si es necesario
            pred_proba_corregida = corregir_prediccion(pred_proba)
            
            # Obtener la clase con mayor probabilidad
            class_index = np.argmax(pred_proba_corregida)
            estado = ESTADOS[class_index]
            
            # Mostrar resultados
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Humedad Actual", f"{humedad}%")
                st.progress(humedad/100)
            
            with col2:
                st.metric("Estado Predicho", estado)
                
            # Mostrar probabilidades para diagnóstico
            with st.expander("🔍 Detalles de predicción"):
                st.write("Probabilidades originales:", pred_proba)
                st.write("Probabilidades corregidas:", pred_proba_corregida)
                st.write("Índice predicho:", class_index)
                
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
