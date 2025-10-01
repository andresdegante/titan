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
    layout="centered"
)

# ============================================
# CARGAR MODELO
# ============================================

@st.cache_resource(show_spinner=True)
def load_model():
    """Carga modelo desde GitHub"""
    base_url = 'https://raw.githubusercontent.com/andresdegante/titan/main/'
    
    try:
        # Cargar archivos
        model = joblib.load(BytesIO(requests.get(base_url + 'titanic_model.pkl').content))
        encoders = joblib.load(BytesIO(requests.get(base_url + 'label_encoders.pkl').content))
        metadata = joblib.load(BytesIO(requests.get(base_url + 'model_metadata.pkl').content))
        
        # Cargar scaler si existe
        scaler = None
        if metadata.get('use_scaler', False):
            scaler = joblib.load(BytesIO(requests.get(base_url + 'scaler.pkl').content))
        
        return model, encoders, metadata, scaler
    
    except Exception as e:
        st.error(f"Error al cargar modelo: {str(e)}")
        st.stop()

model, encoders, metadata, scaler = load_model()

# ============================================
# HEADER CON LOGO
# ============================================

col_logo, col_title = st.columns([1, 3])

with col_logo:
    st.image(
        "https://uev.uadeo.mx/pluginfile.php/1/theme_adaptable/logo/1745964853/logoBlanco.png",
        width=120
    )

with col_title:
    st.title("🚢 Predictor de Supervivencia Titanic")
    st.markdown("Ingresa los datos del pasajero para predecir si sobreviviría")

st.divider()

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.header("📊 Información del Modelo")
    
    # Métricas del modelo
    st.metric("Algoritmo", metadata['model_name'])
    st.metric("Precisión", f"{metadata['accuracy']:.1%}")
    st.metric("ROC-AUC", f"{metadata['roc_auc']:.3f}")
    
    st.divider()
    
    # Descripción del modelo
    st.subheader("📝 Descripción")
    st.markdown("""
    Este modelo utiliza **Machine Learning** para predecir la probabilidad 
    de supervivencia de pasajeros del Titanic basándose en características 
    demográficas y sociales.
    
    **Variables consideradas:**
    - Clase del ticket (1ª, 2ª, 3ª)
    - Sexo y edad del pasajero
    - Composición familiar a bordo
    - Tamaño de la familia
    """)
    
    st.divider()
    
    # Créditos institucionales
    st.subheader("🎓 Créditos")
    st.markdown("""
    **Universidad Autónoma de Occidente**
    
    **Alumno:**  
    Psi. Andrés Cruz Degante
    
    **Profesora:**  
    Dra. Alma Montserrat Romero Serrano
    
    **Materia:**  
    Estadística Aplicada a la Toma de Decisiones
    """)
    
    st.divider()
    st.caption("Octubre 2025 • UADEO")

# ============================================
# FORMULARIO
# ============================================

st.subheader("📝 Datos del Pasajero")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("🎫 Clase", [1, 2, 3], index=2, help="1=Primera, 2=Segunda, 3=Tercera")
    sex = st.radio("👤 Sexo", ['male', 'female'], horizontal=True)
    age = st.slider("🎂 Edad", 0, 80, 30)

with col2:
    sibsp = st.number_input("👫 Hermanos/Cónyuges", 0, 8, 0)
    parch = st.number_input("👨‍👩‍👧 Padres/Hijos", 0, 6, 0)
    family_size = sibsp + parch + 1
    st.metric("👨‍👩‍👧‍👦 Tamaño Familia", family_size)

# Calcular categorías automáticamente
if family_size == 1:
    family_type = 'Alone'
elif family_size <= 4:
    family_type = 'Medium'
else:
    family_type = 'Large'

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

# ============================================
# PREDICCIÓN
# ============================================

if st.button("🔮 Predecir Supervivencia", type="primary", use_container_width=True):
    
    # Preparar datos
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
    
    # Predecir
    if scaler:
        input_data = scaler.transform(input_data)
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # Mostrar resultado
    st.divider()
    st.subheader("📊 Resultados de la Predicción")
    
    if prediction == 1:
        st.success("### ✅ EL PASAJERO SOBREVIVE")
    else:
        st.error("### ❌ EL PASAJERO NO SOBREVIVE")
    
    # Probabilidades
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("💀 Probabilidad de Muerte", f"{probability[0]:.1%}")
    with col_b:
        st.metric("💚 Probabilidad de Supervivencia", f"{probability[1]:.1%}")
    
    # Barra de progreso
    st.markdown("#### Confianza del Modelo")
    st.progress(probability[1], text=f"Supervivencia: {probability[1]:.1%}")
    
    # Interpretación
    st.divider()
    st.markdown("##### 💡 Interpretación")
    if probability[1] > 0.7:
        st.info("Alta probabilidad de supervivencia. Factores como clase alta, sexo femenino y familia pequeña favorecen la supervivencia.")
    elif probability[1] > 0.4:
        st.warning("Probabilidad moderada. Los factores están balanceados.")
    else:
        st.error("Baja probabilidad de supervivencia. Factores como clase baja, sexo masculino y familia grande reducen las posibilidades.")

# ============================================
# FOOTER
# ============================================

st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Dataset:</strong> Titanic - Machine Learning from Disaster (Kaggle)</p>
    <p>Desarrollado con 💙 usando Streamlit y scikit-learn</p>
</div>
""", unsafe_allow_html=True)
