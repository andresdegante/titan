import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO

# ============================================
# CONFIGURACIÓN
# ============================================

st.set_page_config(
    page_title="Predictor Titanic",
    page_icon="🚢",
    layout="wide"  # Cambié a 'wide' para mejor uso del espacio horizontal
)

# ============================================
# CARGAR MODELO
# ============================================

@st.cache_resource(show_spinner=True)
def load_model():
    base_url = 'https://raw.githubusercontent.com/andresdegante/titan/main/'
    try:
        model = joblib.load(BytesIO(requests.get(base_url + 'titanic_model.pkl').content))
        encoders = joblib.load(BytesIO(requests.get(base_url + 'label_encoders.pkl').content))
        metadata = joblib.load(BytesIO(requests.get(base_url + 'model_metadata.pkl').content))
        scaler = None
        if metadata.get('use_scaler', False):
            scaler = joblib.load(BytesIO(requests.get(base_url + 'scaler.pkl').content))
        return model, encoders, metadata, scaler
    except Exception as e:
        st.error(f"Error al cargar modelo: {str(e)}")
        st.stop()

model, encoders, metadata, scaler = load_model()

# ============================================
# HEADER CON LOGO GRANDE Y TÍTULO CENTRADO
# ============================================

st.markdown(
    """
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
        <img src="https://uev.uadeo.mx/pluginfile.php/1/theme_adaptable/logo/1745964853/logoBlanco.png" width="300" style="margin-bottom: 10px;"/>
        <h1 style="font-size: 2.5em; font-weight: bold; margin-bottom: 0;">🚢 Predictor de Supervivencia Titanic</h1>
        <p style="font-size: 1.1em; margin-bottom: 0;">Ingresa los datos del pasajero para predecir si sobreviviría</p>
    </div>
    <hr style='margin-top: 1em; margin-bottom: 1em;'>
    """,
    unsafe_allow_html=True
)

# ============================================
# FORMULARIO Y RESULTADOS EN DOS COLUMNAS
# ============================================

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("📝 Datos del Pasajero")
    pclass = st.selectbox("🎫 Clase", [1, 2, 3], index=2, help="1=Primera, 2=Segunda, 3=Tercera")
    sex = st.radio("👤 Sexo", ['male', 'female'], horizontal=True)
    age = st.slider("🎂 Edad", 0, 80, 30)
    sibsp = st.number_input("👫 Hermanos/Cónyuges", 0, 8, 0)
    parch = st.number_input("👨‍👩‍👧 Padres/Hijos", 0, 6, 0)
    family_size = sibsp + parch + 1
    st.metric("👨‍👩‍👧‍👦 Tamaño Familia", family_size)

# Calcular categorías automáticamente
family_type = 'Alone' if family_size == 1 else 'Medium' if family_size <= 4 else 'Large'
if age < 2:
    age_group = 'Infant'
elif age < 12:
    age_group = 'Child'
elif age < 18:
    age_group = 'Teen'
elif age < 30:
    age_group = 'YoungAdult'
elif age < 60:
    age_group = 'Adult'
else:
    age_group = 'Senior'

with right_col:
    st.subheader("📊 Resultados de la Predicción")

    if st.button("🔮 Predecir Supervivencia", type="primary", use_container_width=True):
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
        # Codificar
        input_data['Sex'] = encoders['Sex'].transform(input_data['Sex'])
        input_data['FamilyType'] = encoders['FamilyType'].transform(input_data['FamilyType'])
        input_data['AgeGroup'] = encoders['AgeGroup'].transform(input_data['AgeGroup'])
        if scaler:
            input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        if prediction == 1:
            st.success("✅ EL PASAJERO SOBREVIVE")
        else:
            st.error("❌ EL PASAJERO NO SOBREVIVE")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("💀 Probabilidad de Muerte", f"{probability[0]:.1%}")
        with col_b:
            st.metric("💚 Probabilidad de Supervivencia", f"{probability[1]:.1%}")

        st.markdown("#### Confianza del Modelo")
        st.progress(probability[1], text=f"Supervivencia: {probability[1]:.1%}")

        st.markdown("##### 💡 Interpretación")
        if probability[1] > 0.7:
            st.info("Alta probabilidad de supervivencia. Factores como clase alta, sexo femenino y familia pequeña favorecen la supervivencia.")
        elif probability[1] > 0.4:
            st.warning("Probabilidad moderada. Los factores están balanceados.")
        else:
            st.error("Baja probabilidad de supervivencia. Factores como clase baja, sexo masculino y familia grande reducen las posibilidades.")

# ============================================
# MÉTRICAS DEL MODELO (PEQUEÑAS, INFERIOR)
# ============================================

st.divider()
st.markdown(
    f"""
    <div style='text-align: center; font-size: 0.8em; color: #aaa; margin-bottom: 0.8em;'>
        <b>Algoritmo:</b> {metadata['model_name']} &nbsp;|&nbsp;
        <b>Precisión:</b> {metadata['accuracy']:.1%} &nbsp;|&nbsp;
        <b>ROC-AUC:</b> {metadata['roc_auc']:.3f}
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================
# CRÉDITOS Y DESCRIPCIÓN (CENTRADOS, INFERIOR)
# ============================================

st.markdown(
    """
    <div style='text-align: center; margin-top: 0.5em; font-size: 1em;'>
        <hr style='margin-bottom: 0.5em;'>
        <span style='font-size:1.2em; font-weight: bold;'>Universidad Autónoma de Occidente</span><br>
        <b>Alumno:</b> Psi. Andrés Cruz Degante<br>
        <b>Profesora:</b> Dra. Alma Montserrat Romero Serrano<br>
        <b>Materia:</b> Estadística Aplicada a la Toma de Decisiones<br>
        <br>
        <span style='font-size: 0.85em; color: #aaa;'>Desarrollado con Streamlit • Dataset: Titanic - Machine Learning from Disaster (Kaggle)</span>
    </div>
    """,
    unsafe_allow_html=True,
)
