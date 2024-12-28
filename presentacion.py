import streamlit as st

def pagina_presentacion():
    # Aplicar estilo al fondo y tipografía
    st.markdown(
        """
        <style>
        /* Fondo general */
        .stApp {
            background-color: #f5f7fa;
            font-family: Arial, Helvetica, sans-serif;
        }
        /* Caja de bienvenida */
        .bienvenida {
            background-color: #ffeedb;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            box-shadow: 2px 2px 10px #aaa;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Crear dos columnas: logo y texto
    col1, col2 = st.columns([1, 3])  # Tamaños proporcionados 1:3 para espacio adecuado

    # Columna 1: Logo
    with col1:
        st.image("img/kokoheadcafe_logo.png", width=200)  # Logo con tamaño reducido

    # Columna 2: Título
    with col2:
        st.markdown(
            '<h1 style="color: #ff6f61; font-size: 50px;">✨ Bienvenidos, Koko Head Cafe ✨</h1>',
            unsafe_allow_html=True
        )

    # Sección de bienvenida con una caja estilizada
    st.markdown(
        '<div class="bienvenida"><h4>Tu Dashboard de Análisis está listo para transformar tu negocio.</h4></div>',
        unsafe_allow_html=True
    )

    # Objetivos del Dashboard
    st.markdown("### 🎯 **Objetivos del Dashboard**")
    st.write("""
    - Analizar **opiniones y sentimientos** de los clientes.  
    - Identificar patrones y tendencias clave a lo largo del tiempo.  
    - Predecir el comportamiento de las reseñas y sugerir estrategias de mejora.
    """)

    # Imágenes en fila
    st.markdown("### 📸 Un vistazo al interior de Koko Head Cafe")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("img/cafe.jpg", caption="Vista Principal", use_container_width=True)
    with col2:
        st.image("img/bar.jpg", caption="Un Menú Especial", use_container_width=True)
    with col3:
        st.image("img/logoGrupo.png", caption="Nuestro equipo", use_container_width=True)

    # Sobre el equipo de DataNova
    st.markdown("### 👩‍🔬 **Sobre Estudio DataNova**")
    st.write("""
    Somos un equipo apasionado de **Científicos de Datos** dedicados a transformar información en oportunidades.  
    Trabajamos con las últimas tecnologías para ofrecer análisis precisos y visualizaciones impactantes.  
    """)
    st.image("img/portada1.png", caption="Estudio DataNova - Transformando Datos en Decisiones")

    # Mensaje final
    st.markdown("**¡Gracias por confiar en nosotros!** 😊")

