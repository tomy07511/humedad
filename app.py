# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(page_title="Monitor de Humedad", layout="centered")
st.title(" Sistema Inteligente de Riego")

# Estados en el ORDEN CORRECTO (0: Muy Seco, 1: Seco, 2: Óptimo, 3: Saturado)
ESTADOS = [
    " Muy Seco (Riego urgente)",    # Índice 0
    " Seco (Necesita agua)",        # Índice 1
    " Óptimo (Buen estado)",        # Índice 2
    "⚠ Saturado (Riesgo de hongos)"  # Índice 3
]

# Mapeo de clases según el orden correcto
CLASE_A_INDICE = {
    'Muy Seco': 0,
    'Seco': 1,
    'Óptimo': 2,
    'Saturado': 3
}

# Cargar modelo y escalador
@st.cache_resource
def load_model():
    try:
        with open('modelo_humedad.pkl', 'rb') as f:
            saved_data = pickle.load(f)
            
            # Verificar el orden de clases
            if 'classes' in saved_data:
                print("Clases en el modelo:", saved_data['classes'])
            
            return saved_data['model'], saved_data['scaler']
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.stop()

model, scaler = load_model()

def predecir_estado(humedad):
    """Función que maneja la predicción con corrección de orden"""
    humedad_norm = scaler.transform([[humedad]])
    pred_proba = model.predict_proba(humedad_norm)[0]  # Probabilidades para cada clase
    
    # Obtener el índice según el orden deseado
    class_index = np.argmax(pred_proba)
    
    # Si el modelo tiene otro orden, ajustamos aquí
    # (Esto depende de cómo se entrenó el modelo)
    # Ejemplo si el modelo devuelve [Óptimo, Saturado, Seco, Muy Seco]:
    # class_index = 3 - class_index  # Invertir el orden
    
    return class_index, pred_proba

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
            # Predicción
            class_index, probabilidades = predecir_estado(humedad)
            estado = ESTADOS[class_index]
            
            # Mostrar resultados
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Humedad Actual", f"{humedad}%")
                st.progress(humedad/100)
            
            with col2:
                st.metric("Estado Predicho", estado)
                
            # Mostrar probabilidades
            with st.expander(" Probabilidades por estado"):
                for i, prob in enumerate(probabilidades):
                    st.write(f"{ESTADOS[i]}: {prob:.1%}")
                
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
