import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO
import time

# ============================================
# CONFIGURACIÓN DE LA APP
# ============================================

st.set_page_config(
    page_title="Predictor de Supervivencia Titanic",
    page_icon="🚢",
    layout="wide"
)

# ============================================
# CARGAR MODELO DESDE GITHUB CON RETRY
# ============================================

@st.cache_resource(show_spinner=False)
def load_model_from_github(repo_url, max_retries=3):
    """
    Carga el modelo con reintentos automáticos
    """
    for attempt in range(max_retries):
        try:
            with st.spinner(f'Cargando modelo... (Intento {attempt + 1}/{max_retries})'):
                # Cargar modelo
                model_response = requests.get(repo_url + 'titanic_model.pkl', timeout=30)
                model_response.raise_for_status()
                model = joblib.load(BytesIO(model_response.content))
                
                # Cargar encoders
                encoders_response = requests.get(repo_url + 'label_encoders.pkl', timeout=30)
                encoders_response.raise_for_status()
                label_encoders = joblib.load(BytesIO(encoders_response.content))
                
                # Cargar metadata
                metadata_response = requests.get(repo_url + 'model_metadata.pkl', timeout=30)
                metadata_response.raise_for_status()
                metadata = joblib.load(BytesIO(metadata_response.content))
                
                # Cargar scaler si es necesario
                scaler = None
                if metadata.get('use_scaler', False):
                    scaler_response = requests.get(repo_url + 'scaler.pkl', timeout=30)
                    scaler_response.raise_for_status()
                    scaler = joblib.load(BytesIO(scaler_response.content))
                
                return model, label_encoders, metadata, scaler
                
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"❌ Error al cargar el modelo: {str(e)}")
                st.info("Verifica que la URL del repositorio sea correcta y los archivos .pkl estén en la rama 'main'")
                return None, None, None, None
            time.sleep(2)

# ⚠️ CAMBIA ESTA URL POR LA DE TU REPOSITORIO
GITHUB_REPO_URL = 'https://github.com/andresdegante/titan'

# Cargar modelo
model, label_encoders, metadata, scaler = load_model_from_github(GITHUB_REPO_URL)

# ============================================
# INTERFAZ DE USUARIO
# ============================================

st.title("🚢 Predictor de Supervivencia del Titanic")
st.markdown("---")

if model is None:
    st.error("⚠️ No se pudo cargar el modelo. Verifica la URL del repositorio.")
    st.code(f"URL actual: {GITHUB_REPO_URL}")
    st.info("https://github.com/andresdegante/titan")
    st.stop()

# Mostrar información del modelo
with st.sidebar:
    st.header("ℹ️ Información del Modelo")
    st.metric("Modelo", metadata['model_name'])
    st.metric("Accuracy", f"{metadata['accuracy']:.2%}")
    st.metric("ROC-AUC", f"{metadata['roc_auc']:.4f}")
    st.markdown("---")
    st.markdown("**Variables del modelo:**")
    for feature in metadata['features']:
        st.text(f"• {feature}")

# Formulario de entrada
st.header("📝 Ingresa los datos del pasajero")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Información Personal")
    sex = st.selectbox("Sexo", options=['male', 'female'], index=0)
    age = st.number_input("Edad", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
    age_group = st.selectbox(
        "Grupo de Edad",
        options=['Infant', 'Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'],
        index=4
    )

with col2:
    st.subheader("Información del Ticket")
    pclass = st.selectbox("Clase del Ticket", options=[1, 2, 3], index=2)
    sibsp = st.number_input("Hermanos/Cónyuges a bordo", min_value=0, max_value=8, value=0)
    parch = st.number_input("Padres/Hijos a bordo", min_value=0, max_value=6, value=0)

with col3:
    st.subheader("Información Familiar")
    family_size = st.number_input(
        "Tamaño de Familia",
        min_value=1,
        max_value=11,
        value=sibsp + parch + 1
    )
    family_type = st.selectbox(
        "Tipo de Familia",
        options=['Alone', 'Medium', 'Large'],
        index=0 if family_size == 1 else (1 if family_size <= 4 else 2)
    )

# ============================================
# REALIZAR PREDICCIÓN
# ============================================

if st.button("🔮 Predecir Supervivencia", type="primary", use_container_width=True):
    
    # Crear dataframe con los datos de entrada
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'FamilySize': [family_size],
        'FamilyType': [family_type],
        'AgeGroup': [age_group]
    })
    
    # Codificar variables categóricas
    try:
        input_data['Sex'] = label_encoders['Sex'].transform(input_data['Sex'])
        input_data['FamilyType'] = label_encoders['FamilyType'].transform(input_data['FamilyType'])
        input_data['AgeGroup'] = label_encoders['AgeGroup'].transform(input_data['AgeGroup'])
        
        # Escalar si es necesario
        if scaler is not None:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
        else:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
        
        # Mostrar resultados
        st.markdown("---")
        st.header("📊 Resultados de la Predicción")
        
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            if prediction == 1:
                st.success("### ✅ SOBREVIVE")
                st.balloons()
            else:
                st.error("### ❌ NO SOBREVIVE")
        
        with col_result2:
            st.metric("Probabilidad de Supervivencia", f"{probability[1]:.2%}")
            st.metric("Probabilidad de Muerte", f"{probability[0]:.2%}")
        
        # Barra de probabilidad
        st.markdown("### Distribución de Probabilidad")
        prob_df = pd.DataFrame({
            'Resultado': ['Muere', 'Sobrevive'],
            'Probabilidad': [probability[0], probability[1]]
        })
        st.bar_chart(prob_df.set_index('Resultado'))
        
    except Exception as e:
        st.error(f"Error durante la predicción: {str(e)}")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Desarrollado con ❤️ usando Streamlit y scikit-learn</p>
    <p>Dataset: Titanic - Machine Learning from Disaster</p>
</div>
""", unsafe_allow_html=True)
