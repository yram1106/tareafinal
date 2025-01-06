import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Análisis Interactivo de Datos",
    page_icon="📊",
    layout="wide"
)

# Menú de navegación
menu = ["Inicio 🏠", "EDA 📊", "Regresiones 📈", "Informe 📄"]
choice = st.sidebar.selectbox("Navegación", menu)

# Manejo de las opciones del menú
if choice == "Inicio 🏠":
    st.title("Bienvenido a la Aplicación de Análisis Interactivo de Datos")
    st.write("Usa el menú de la izquierda para navegar entre los módulos.")

elif choice == "EDA 📊":
    st.title("Análisis Exploratorio de Datos 📊")
    st.write("Carga un dataset y realiza análisis exploratorios.")

elif choice == "Regresiones 📈":
    st.title("Módulo de Regresiones 📈")
    st.write("Aplica modelos de regresión a tus datos.")

elif choice == "Informe 📄":
    st.title("Generación de Informes 📄")
    st.write("Crea y descarga informes ejecutivos de tu análisis.")
