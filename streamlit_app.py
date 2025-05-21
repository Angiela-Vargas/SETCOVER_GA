import streamlit as st
import pandas as pd
import json
import altair as alt

st.title("Dashboard GA para Set Covering")

@st.cache_data
def load_stats():
    return pd.read_csv('ga_stats.csv')

@st.cache_data
def load_params():
    with open('best_params.json','r') as f:
        return json.load(f)

# Cargar datos
df_stats    = load_stats()
best_params = load_params()

# Mostrar parámetros óptimos
st.subheader("Parámetros óptimos encontrados")
st.json(best_params)

# Primera gráfica: line chart
st.subheader("Evolución del fitness (líneas)")
st.line_chart(
    df_stats.set_index('Generation')[['BestFitness','MeanFitness']]
)

# Segunda gráfica: líneas + puntos con Altair
st.subheader("Evolución del fitness (líneas y puntos)")
base = alt.Chart(df_stats).encode(x='Generation')
line_best   = base.mark_line(color='blue').encode(y='BestFitness')
points_best = base.mark_point(color='blue').encode(y='BestFitness')
line_mean   = base.mark_line(color='red').encode(y='MeanFitness')
points_mean = base.mark_point(color='red').encode(y='MeanFitness')

chart = (line_best + points_best + line_mean + points_mean).properties(
    width=700, height=400,
    title="Best (azul) y Mean (rojo) Fitness por generación"
).interactive()

st.altair_chart(chart, use_container_width=True)

# Tabla de datos
st.subheader("Tabla de estadísticas por generación")
st.dataframe(df_stats.style.format({
    'Generation':'{:.0f}',
    'BestFitness':'{:.0f}',
    'MeanFitness':'{:.0f}'
}))

