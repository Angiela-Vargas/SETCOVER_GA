import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt

# 1) Título y descripción
st.title("Dashboard GA para Set Covering")
st.markdown("""
Visualización de la evolución del Algoritmo Genético y de los parámetros
optimados con *Bayesian Optimization*.
""")

# 2) Carga de datos
@st.cache
def load_stats():
    df = pd.read_csv('ga_stats.csv')
    return df

@st.cache
def load_params():
    with open('best_params.json','r') as f:
        return json.load(f)

df_stats  = load_stats()
best_params = load_params()

# 3) Mostrar parámetros óptimos
st.subheader("Parámetros óptimos encontrados")
st.json(best_params)

# 4) Gráfica evolución fitness
st.subheader("Evolución de la aptitud (fitness) por generación")
fig, ax = plt.subplots()
ax.plot(df_stats['Generation'], df_stats['BestFitness'],
        label='Mejor fitness', marker='o')
ax.plot(df_stats['Generation'], df_stats['MeanFitness'],
        label='Fitness medio', marker='o')
ax.set_xlabel("Generación")
ax.set_ylabel("Fitness")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# 5) Tabla de estadísticas
st.subheader("Tabla de estadísticas por generación")
st.dataframe(df_stats.style.format({
    'Generation':'{:.0f}',
    'BestFitness':'{:.0f}',
    'MeanFitness':'{:.0f}'
}))

# 6) Insights adicionales
st.markdown("""
**Observaciones**:

- La **pendiente** se aplana alrededor de la generación 30–40,  
  indicando convergencia.
- Puedes ajustar el rango de generaciones o parámetros desde MATLAB y  
  volver a exportar los CSV/JSON para actualizar este dashboard.
""")
