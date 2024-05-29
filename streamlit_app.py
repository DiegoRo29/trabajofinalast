import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

# Cargar dataset
@st.cache_data
def load_data():
    data = pd.read_csv("atus_valor_06.csv")
    return data

df = load_data()

# Mostrar nombres de las columnas para debug
st.write("Nombres de las columnas del DataFrame sin filtrar:")
st.write(df.columns)

# Mostrar algunas filas del DataFrame para verificar valores
st.write("Algunas filas del DataFrame sin filtrar:")
st.write(df.head(20))

# Filtros interactivos para seleccionar el periodo de tiempo, el municipio y la variable
years = df['año'].unique()
municipios = df['desc_municipio'].unique()
variables = {
    'Accidentes': 'Accidentes de tránsito terrestre en zonas urbanas y suburbanas',
    'Heridos': 'Víctimas heridas en los accidentes de tránsito',
    'Muertos': 'Víctimas muertas en los accidentes de tránsito'
}

selected_municipio = st.sidebar.selectbox('Selecciona el municipio', municipios)
selected_variable = st.sidebar.selectbox('Selecciona la variable', list(variables.keys()))

# Herramientas de interacción para seleccionar diferentes periodos de tiempo
st.sidebar.header("Herramientas de Interacción")
start_year = st.sidebar.slider("Selecciona el año de inicio", int(df['año'].min()), int(df['año'].max()), int(df['año'].min()))
end_year = st.sidebar.slider("Selecciona el año de fin", int(df['año'].min()), int(df['año'].max()), int(df['año'].max()))

# Verificar valores seleccionados
st.write(f"Municipio seleccionado: {selected_municipio}")
st.write(f"Variable seleccionada: {selected_variable}")
st.write(f"Descripción de la variable seleccionada: {variables[selected_variable]}")
st.write(f"Año de inicio seleccionado: {start_year}")
st.write(f"Año de fin seleccionado: {end_year}")

if selected_variable == 'Accidentes':
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
elif selected_variable == 'Heridos':
    order = (2, 2, 2)
    seasonal_order = (2, 2, 2, 9)
elif selected_variable == 'Muertos':
    order = (3, 0, 2)
    seasonal_order = (3, 0, 2, 12)

# Filtrar datos por años, municipio y variable seleccionados
filtered_df = df[(df['año'] >= start_year) & (df['año'] <= end_year) & (df['desc_municipio'] == selected_municipio) & (df['indicador'] == variables[selected_variable])]

# Verificar si el filtrado se realizó correctamente
st.write("DataFrame filtrado:")
st.write(filtered_df.head())

# Renombrar la columna 'valor' según la variable seleccionada
filtered_df = filtered_df.rename(columns={'valor': selected_variable.lower()})

# Convertir la columna año a tipo datetime
filtered_df['año'] = pd.to_datetime(filtered_df['año'], format='%Y')

# Establecer la columna año como índice
filtered_df = filtered_df.set_index('año')

# Verificar si el DataFrame no está vacío antes de continuar
if not filtered_df.empty:
    # Gráfico de línea de la serie temporal original
    st.subheader(f"Serie Temporal Original: {selected_variable}")
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_df.index, filtered_df[selected_variable.lower()], marker='o', linestyle='-', color='b')
    plt.title(f'Tendencia de la cantidad de {selected_variable.lower()} en accidentes de tránsito')
    plt.xlabel('Año')
    plt.ylabel(f'Número de {selected_variable.lower()}')
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Cambia el formato aquí
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

    # Descomposición de la serie temporal si hay suficientes datos
    if len(filtered_df) >= 24:
        st.subheader("Descomposición de la Serie Temporal")
        result = seasonal_decompose(filtered_df[selected_variable.lower()], model='additive', period=7)
        
        plt.figure(figsize=(14, 10))

        plt.subplot(4, 1, 1)
        result.observed.plot(label='Original')
        plt.title(f'Original - Número de {selected_variable.lower()}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.subplot(4, 1, 2)
        result.trend.plot(label='Tendencia')
        plt.title(f'Tendencia - Número de {selected_variable.lower()}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.subplot(4, 1, 3)
        result.seasonal.plot(label='Estacionalidad')
        plt.title(f'Estacionalidad - Número de {selected_variable.lower()}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.subplot(4, 1, 4)
        plt.scatter(result.resid.index, result.resid, label='Residual')
        plt.axhline(0, color='black')  # Agrega una línea horizontal en el valor cero
        plt.title(f'Residual - Número de {selected_variable.lower()}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        st.pyplot(plt)

    else:
        st.write("No hay suficientes datos para la descomposición de la serie temporal (se requieren al menos 24 observaciones).")

    #Diferenciar variable seleccionada
    filtered_df_diferenciada = filtered_df.copy()
    filtered_df_diferenciada[selected_variable.lower()] = filtered_df[selected_variable.lower()].diff().dropna()
    # Gráficos de autocorrelación y autocorrelación parcial
    st.subheader("Gráficos de Autocorrelación (ACF) y Autocorrelación Parcial (PACF)")
    max_lags = min(20, len(filtered_df) // 2 - 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_acf(filtered_df[selected_variable.lower()], ax=ax1, lags=max_lags, alpha=0.05)
    plot_pacf(filtered_df[selected_variable.lower()], ax=ax2, lags=max_lags, alpha=0.05)
    plt.tight_layout()
    st.pyplot(plt)

    # Aplicación de modelos ARIMA y SARIMA
    st.subheader("Modelos ARIMA y SARIMA")
    # Modelo ARIMA
    model_arima = ARIMA(filtered_df_diferenciada[selected_variable.lower()], order=order)
    result_arima = model_arima.fit()
    st.write(result_arima.summary())

    inicio = pd.to_datetime(start_year, format='%Y')
    fin = pd.to_datetime(end_year, format='%Y')


    predicciones_arima = result_arima.predict(start=fin, end=fin + pd.DateOffset(years=3), freq='YS')

    # Modelo SARIMA
    model_sarima = SARIMAX(filtered_df_diferenciada[selected_variable.lower()], order=order, seasonal_order=seasonal_order)
    result_sarima = model_sarima.fit(disp=False)
    st.write(result_sarima.summary())

    predicciones_sarima = result_sarima.predict(start=fin, end=fin + pd.DateOffset(years=3), freq='YS')

    # Gráfico de predicciones ARIMA
    st.subheader("Modelo ARIMA")
    plt.figure(figsize=(14, 6))
    plt.plot(filtered_df_diferenciada.index, filtered_df_diferenciada[selected_variable.lower()], label='Histórico', marker='o', color='blue')
    plt.plot(filtered_df_diferenciada.index, result_arima.fittedvalues, label='Modelo ARIMA', color='green')
    plt.plot(predicciones_arima.index, predicciones_arima, label='Predicciones', color='red', linestyle='--', marker='o')
    plt.legend()
    plt.title(f'Predicciones del modelo ARIMA para el numero de {selected_variable.lower()}')
    plt.xlabel('Año')
    plt.ylabel(f'Número de {selected_variable.lower()}')
    plt.grid(True)
    st.pyplot(plt)

    # Gráfico de predicciones SARIMA
    st.subheader("Modelo SARIMA")
    plt.figure(figsize=(14, 6))
    plt.plot(filtered_df_diferenciada.index, filtered_df_diferenciada[selected_variable.lower()], label='Histórico', marker='o', color='blue')
    plt.plot(filtered_df_diferenciada.index, result_sarima.fittedvalues, label='Modelo SARIMA', color='green')
    plt.plot(predicciones_sarima.index, predicciones_sarima, label='Predicciones', color='red', linestyle='--', marker='o')
    plt.legend()
    plt.title(f'Predicciones del modelo SARIMA para el numero de {selected_variable.lower()}')
    plt.xlabel('Año')
    plt.ylabel(f'Número de {selected_variable.lower()}')
    plt.grid(True)
    st.pyplot(plt)

    # Gráfico combinado de predicciones ARIMA y SARIMA
    st.subheader("Graficas del modelo ARIMA y SARIMA")
    plt.figure(figsize=(14, 8))
    plt.plot(filtered_df_diferenciada.index, filtered_df_diferenciada[selected_variable.lower()], label='Original', marker='o', color='blue')
    plt.plot(predicciones_arima.index, predicciones_arima, label='ARIMA', color='red', linestyle='--')
    plt.plot(predicciones_sarima.index, predicciones_sarima, label='SARIMA', color='green', linestyle='--')
    plt.legend()
    plt.title('Predicciones ARIMA y SARIMA')
    plt.xlabel('Año')
    plt.ylabel(f'Número de {selected_variable.lower()}')
    plt.grid(True)
    st.pyplot(plt)


    from sklearn.linear_model import LinearRegression

    # Crear un rango de fechas para las predicciones futuras
    dates = pd.date_range(start=inicio, end=fin, freq='YS')
    future_dates = pd.date_range(start=fin, periods=3, freq='YS')

    # Crear y entrenar el modelo de regresión lineal
    linear_model = LinearRegression()
    linear_model.fit(filtered_df.index.to_julian_date().values.reshape(-1, 1), filtered_df[selected_variable.lower()])

    linear_predictions = linear_model.predict(future_dates.to_julian_date().values.reshape(-1, 1))
    linear_predictions2 = linear_model.predict(dates.to_julian_date().values.reshape(-1, 1))

    # Graficar la regresion lineal, la original y prediccion de datos del modelo de regresion lineal
    st.subheader("Regresion lineal y predicciones del modelo de regresión lineal")
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_df.index, filtered_df[selected_variable.lower()], label='Original', marker='o', color='blue')
    plt.plot(future_dates, linear_predictions, label='Predicción Lineal', color='red')
    plt.plot(dates, linear_predictions2, label='Modelo de Regresión Lineal', color='green', linestyle='--')
    plt.title(f'Predicciones del modelo de regresión lineal para el numero de {selected_variable.lower()}')
    plt.xlabel('Año')
    plt.ylabel(f'Número de {selected_variable.lower()}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

    from sklearn.svm import SVR
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    y = filtered_df[selected_variable.lower()].values
    y= scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Crear y entrenar el modelo SVR
    svr_model = SVR()
    svr_model.fit(filtered_df.index.to_julian_date().values.reshape(-1, 1), y)

    svr_predictions = svr_model.predict(future_dates.to_julian_date().values.reshape(-1, 1))
    svr_predictions2 = svr_model.predict(dates.to_julian_date().values.reshape(-1, 1))

    # Graficar la máquina de soporte vectorial, la original y la predicción de datos del modelo de regresión lineal
    st.subheader("Máquina de soporte vectorial y predicciones")
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_df.index, y, label='Original', marker='o', color='blue')
    plt.plot(future_dates, svr_predictions, label='Predicción', color='red')
    plt.plot(dates, svr_predictions2, label='Modelo de SVR', color='green', linestyle='--')
    plt.title(f'Predicciones del modelo de SVR para el numero de {selected_variable.lower()}')
    plt.xlabel('Año')
    plt.ylabel(f'Número de {selected_variable.lower()}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(filtered_df.index.to_julian_date().values.reshape(-1, 1), y, test_size=0.2, random_state=43)

    # Crear y entrenar el modelo de árbol de decisión
    tree_model = DecisionTreeRegressor()
    tree_model.fit(filtered_df.index.to_julian_date().values.reshape(-1, 1), y)
    # tree_model2 = DecisionTreeRegressor()
    # tree_model2.fit(X_train, y_train)

    future_dates = pd.date_range(start=fin- pd.DateOffset(years=1), end=fin + pd.DateOffset(years=1), freq='YS')
    dates = pd.date_range(start=inicio, end=fin, freq='YS')
    tree_predictions = tree_model.predict(future_dates.to_julian_date().values.reshape(-1, 1))
    tree_predictions2 = tree_model.predict(dates.to_julian_date().values.reshape(-1, 1))


    X_train = pd.to_datetime(pd.Series(X_train.flatten()), origin='julian', unit='D')
    X_train, y_train = zip(*sorted(zip(X_train, y_train)))

    # Graficar el árbol de decisión, la original y la predicción de datos del modelo de regresión lineal
    st.subheader("Árbol de decisión y predicciones")
    plt.figure(figsize=(12, 6))
    plt.plot(dates, tree_predictions2, label='Modelo de árbol de decisión', marker='o', color='green')
    plt.plot(future_dates, tree_predictions, label='Predicción', color='red', marker='o')
    plt.plot(X_train, y_train, label='Original', color='b', linestyle='-')
    plt.title(f'Predicciones del modelo de arboles de decisión para el numero de {selected_variable.lower()}')
    plt.xlabel('Año')
    plt.ylabel(f'Número de {selected_variable.lower()}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

else:
    st.write(f"No hay datos disponibles para la selección actual: {selected_variable}, {selected_municipio}, {start_year}-{end_year}")
