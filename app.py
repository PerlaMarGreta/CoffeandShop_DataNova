import streamlit as st
from presentacion import pagina_presentacion
from mapa import mostrar_mapa
from analisis_sentimientos2 import analizar_sentimientos

# Configuración inicial
st.set_page_config(page_title="Koko Head Cafe - Dashboard", layout="wide")

# Título principal
st.title("🌟 Dashboard de Análisis para Koko Head Cafe 🌟")

# Crear pestañas
tab1, tab2, tab3 = st.tabs(["🎉 Presentación", "📍 Mapa Interactivo", "📊 Análisis de Sentimientos"])

# Pestaña 1: Presentación
with tab1:
    pagina_presentacion()

# Pestaña 2: Mapa Interactivo
with tab2:
    mostrar_mapa()

# Pestaña 3: Análisis de Sentimientos
with tab3:
    analizar_sentimientos()
