# Utilizar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos necesarios al contenedor
COPY . ./

# Instalar las dependencias desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto predeterminado de Streamlit
EXPOSE 8080

# Comando para ejecutar la aplicación Streamlit
ENTRYPOINT  ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
