import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import joblib

def mostrar_mapa():
        # Título
        st.title("Predicción de Éxito de Negocios")

        # Cargar datos preprocesados y modelo entrenado
        @st.cache_data
        def cargar_datos():
            return pd.read_csv("data/Hawaii_predicciones.csv")

        @st.cache_resource
        def cargar_modelo():
            return joblib.load("data/modelo_hawaii.pkl")

        Hawaii = cargar_datos()
        model = cargar_modelo()

        # Generar el mapa interactivo
        st.subheader("Mapa Interactivo con Predicciones")
        mapa_hawaii = folium.Map(location=[Hawaii['latitude'].mean(), Hawaii['longitude'].mean()], zoom_start=7)
        marker_cluster = MarkerCluster().add_to(mapa_hawaii)

        for _, row in Hawaii.iterrows():
            popup_text = f"""
            <b>{row['name_x']}</b><br>
            Dirección: {row['address']}<br>
            Rating: {row['rating']}<br>
            Reseñas: {row['num_of_reviews']}<br>
            Probabilidad de éxito: {row['prob_success']:.2f}%
            """
            color = "green" if row['prediction'] == 1 else "red"
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color=color, icon="info-sign")
            ).add_to(marker_cluster)

        folium_static(mapa_hawaii)
