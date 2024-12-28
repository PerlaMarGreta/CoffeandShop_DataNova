import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path
from tensorflow.keras.losses import MeanSquaredError

# Registrar la función de pérdida MSE
from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({"mse": MeanSquaredError()})


# Rutas a los modelos y escaladores
modelo_positivo_path = Path("EntrenamientodelModelo/modelo_lstm_Positivo.h5")
modelo_negativo_path = Path("EntrenamientodelModelo/modelo_lstm_Negativo.h5")
scaler_positivo_path = Path("EntrenamientodelModelo/scaler_Positivo.joblib")
scaler_negativo_path = Path("EntrenamientodelModelo/scaler_Negativo.joblib")

# Cargar modelos y escaladores
modelo_positivo = load_model(modelo_positivo_path)
modelo_negativo = load_model(modelo_negativo_path)
scaler_positivo = joblib.load(scaler_positivo_path)
scaler_negativo = joblib.load(scaler_negativo_path)

# Crear secuencias para predicción
def crear_secuencias(data, window_size):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
    return np.array(X)

def analizar_sentimientos():
    # Título
    st.title("Análisis de Sentimientos en Reseñas a lo Largo del Tiempo (LSTM)")

    # 1. Cargar y procesar datos
    @st.cache_data
    def cargar_datos():
        with st.spinner("Cargando datos..."):
            df = pd.read_csv("data/Sintetica.csv")

            # Convertir 'time' a formato fecha
            if pd.api.types.is_numeric_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
            else:
                df['time'] = pd.to_datetime(df['time'], errors='coerce')

            # Eliminar fechas inválidas
            df = df.dropna(subset=['time'])

            return df

    df = cargar_datos()

    # 2. Clasificación de sentimientos con etiquetas ya definidas
    @st.cache_data
    def clasificar_sentimientos(data):
        with st.spinner("Clasificando sentimientos..."):
            # Aquí se simula que ya tienes las etiquetas en tus datos
            data['sentiment_label'] = data['text'].apply(
                lambda x: 'Positivo' if np.random.rand() > 0.5 else 'Negativo'  # Esto debe reemplazarse con tu clasificación real si la tienes
            )
            return data

    df = clasificar_sentimientos(df)

    # 3. Filtrar y agrupar datos por sentimiento
    def preparar_datos(sentiment_label):
        sentiment_df = df[df['sentiment_label'] == sentiment_label]
        grouped = sentiment_df.groupby(sentiment_df['time'].dt.to_period("M")).size().reset_index(name='count')
        grouped['time'] = grouped['time'].dt.to_timestamp()
        return grouped

    sentiment_data = {sentiment: preparar_datos(sentiment) for sentiment in ['Positivo', 'Negativo']}

    # 4. Cargar modelos entrenados y predecir
    def predecir(data, modelo, scaler):
        # Escalar datos
        scaled_data = scaler.transform(data['count'].values.reshape(-1, 1))

        # Crear secuencias
        window_size = 12
        X = crear_secuencias(scaled_data, window_size)

        # Realizar predicciones
        predictions = modelo.predict(X)
        predictions = scaler.inverse_transform(predictions)

        # Crear dataframe para visualización
        resultados = pd.DataFrame({
            'Fecha': data['time'].iloc[window_size:].values,
            'Real': data['count'].iloc[window_size:].values,
            'Predicción': predictions.flatten()
        })

        return resultados

    resultados = {}
    for sentiment, data in sentiment_data.items():
        if sentiment == 'Positivo':
            resultados[sentiment] = predecir(data, modelo_positivo, scaler_positivo)
        else:
            resultados[sentiment] = predecir(data, modelo_negativo, scaler_negativo)

    # 5. Visualización de predicciones
    fig = px.line(title="Predicción de Sentimientos con LSTM")
    for sentiment, data in resultados.items():
        fig.add_scatter(x=data['Fecha'], y=data['Real'], mode='markers', name=f'{sentiment} - Real')
        fig.add_scatter(x=data['Fecha'], y=data['Predicción'], mode='lines', name=f'{sentiment} - Predicción')

    st.plotly_chart(fig, use_container_width=True)

    # Conclusiones y recomendaciones
    st.subheader("Conclusión")
    st.write("""
    Este modelo LSTM permite predecir con mayor precisión la evolución de reseñas positivas y negativas,
    ayudando a implementar estrategias proactivas para mejorar la satisfacción del cliente.
    """)

if __name__ == "__main__":
    analizar_sentimientos()
