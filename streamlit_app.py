import streamlit as st
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random

# --- 1) Función Python que corre el GA (DEAP) y devuelve un DataFrame con stats ---
def run_ga_python(pop_size, max_gen, cx_frac, mut_rate):
    # Carga de datos
    cov   = np.loadtxt('set_cover_500x500.csv', delimiter=',')
    costs = pd.read_excel('Costo_S.xlsx', header=None).iloc[1,1:501].values

    # DEAP setup
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list,  fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                    toolbox.attr_bool, len(costs))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalSC(ind):
        sel  = np.array(ind)
        cost = costs.dot(sel)
        return (cost if np.all(cov.dot(sel)>=1) else cost+1e6,)

    toolbox.register("evaluate", evalSC)
    toolbox.register("mate",    tools.cxTwoPoint)
    toolbox.register("mutate",  tools.mutFlipBit, indpb=mut_rate)
    toolbox.register("select",  tools.selTournament, tournsize=3)

    # Estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    pop = toolbox.population(n=pop_size)
    log = tools.Logbook()

    # Ejecutar GA
    pop, log = algorithms.eaSimple(pop, toolbox,
                                   cxpb=cx_frac, mutpb=mut_rate,
                                   ngen=max_gen, stats=stats,
                                   verbose=False)

    # Construir DataFrame
    df = pd.DataFrame({
        'Generation': log.select("gen"),
        'BestFitness': log.select("min"),
        'MeanFitness': log.select("avg"),
    })
    return df

# --- 2) Streamlit app ---
st.title("Interfaz Interactiva del GA para Set Covering")

# --- 3) Sidebar: sliders para hiperparámetros ---
st.sidebar.header("Ajuste de parámetros del GA")
pop_size = st.sidebar.slider("Population size",  50, 200, 100, step=10)
max_gen  = st.sidebar.slider("Max generations", 50, 300, 100, step=10)
cx_frac  = st.sidebar.slider("Crossover frac.", 0.5, 0.9, 0.7, step=0.05)
mut_rate = st.sidebar.slider("Mutation rate",  0.005, 0.05, 0.01, step=0.005)

if st.sidebar.button("Ejecutar GA"):
    with st.spinner("Corriendo GA, espera unos segundos..."):
        df_stats = run_ga_python(pop_size, max_gen, cx_frac, mut_rate)

    # --- 4) Mostrar resultados ---
    st.subheader("Evolución del fitness")
    st.line_chart(df_stats.set_index('Generation'))

    st.subheader("Estadísticas por generación")
    st.dataframe(df_stats)

    # Datos finales
    best_cost = int(df_stats['BestFitness'].iloc[-1])
    st.markdown(f"**Mejor coste final:** {best_cost}")
    # Antenas seleccionadas aproximadas (asumiendo fitness sin penalización)
    st.markdown(f"**Antenas seleccionadas aprox.:** "
                f"{int(best_cost/np.mean(pd.read_excel('Costo_S.xlsx', header=None).iloc[1,1:]))}")

else:
    st.info("Ajusta los parámetros en la barra lateral y pulsa **Ejecutar GA**.")
