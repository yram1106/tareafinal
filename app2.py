import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pdfkit
import os

# Título de la Aplicación
st.title("Aplicación Interactiva de Análisis de Datos")

# 1. Carga de Datasets
st.header("1. Carga de Dataset")
uploaded_file = st.file_uploader("Sube tu archivo CSV o descarga desde Kaggle", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset cargado exitosamente!")
    st.write("Vista previa del Dataset:")
    st.dataframe(df.head())
else:
    st.warning("Por favor, sube un archivo para continuar.")

# 2. Módulo de EDA
if uploaded_file:
    st.header("2. Análisis Exploratorio de Datos (EDA)")

    # Información general
    st.subheader("Estadísticas Descriptivas")
    st.write(df.describe())

    # Visualizaciones
    st.subheader("Visualizaciones")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    selected_column = st.selectbox("Selecciona una columna para el histograma:", numeric_columns)
    if selected_column:
        fig, ax = plt.subplots()
        sns.histplot(df[selected_column], ax=ax, kde=True)
        st.pyplot(fig)

    st.subheader("Gráfico de Dispersión")
    scatter_x = st.selectbox("Selecciona la columna del eje X:", numeric_columns)
    scatter_y = st.selectbox("Selecciona la columna del eje Y:", numeric_columns)
    if scatter_x and scatter_y:
        fig = px.scatter(df, x=scatter_x, y=scatter_y, title="Gráfico de Dispersión")
        st.plotly_chart(fig)

# 3. Módulo de Regresiones
if uploaded_file:
    st.header("3. Modelos de Regresión")
    target_variable = st.selectbox("Selecciona la variable objetivo (Y):", numeric_columns)
    if target_variable:
        feature_variables = st.multiselect("Selecciona las variables predictoras (X):", numeric_columns)

        if st.button("Ejecutar Regresión Lineal"):
            X = df[feature_variables]
            y = df[target_variable]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predicciones
            y_pred = model.predict(X_test)

            # Métricas
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
            st.write(f"**R² Score:** {r2:.2f}")

            # Visualización
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Valores Reales")
            ax.set_ylabel("Predicciones")
            st.pyplot(fig)

# 4. Generación de Informes
if uploaded_file:
    st.header("4. Generación de Informes")
    if st.button("Generar Informe PDF"):
        report_html = """
        <html>
        <body>
        <h1>Informe de Análisis</h1>
        <p>Resumen de Estadísticas:</p>
        """ + df.describe().to_html() + """
        </body>
        </html>
        """
        pdfkit.from_string(report_html, "informe.pdf")
        st.success("Informe generado: informe.pdf")
        with open("informe.pdf", "rb") as pdf_file:
            st.download_button("Descargar Informe PDF", pdf_file, "informe.pdf")
