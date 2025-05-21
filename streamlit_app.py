import streamlit as st
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random

# --- GA con devolución explícita del mejor individuo y stats ---
def run_ga_python(pop_size, max_gen, cx_frac, mut_rate):
    # 1) Carga de datos (hazlo fuera si quieres optimizar)
    cov   = np.loadtxt('set_cover_500x500.csv', delimiter=',')
    costs = pd.read_excel('Costo_S.xlsx', header=None).iloc[1,1:501].values

    # 2) Setup DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
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

    # 3) Estadísticas y HallOfFame
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    hof = tools.HallOfFame(1)

    # 4) Ejecuta el GA
    pop = toolbox.population(n=pop_size)
    pop, log = algorithms.eaSimple(pop, toolbox,
                                   cxpb=cx_frac, mutpb=mut_rate,
                                   ngen=max_gen, stats=stats,
                                   halloffame=hof, verbose=False)

    # 5) Extraer mejor individuo y su coste
    best_ind = np.array(hof[0])
    best_cost = costs.dot(best_ind)
    num_antennas = int(best_ind.sum())

    # 6) Construir DataFrame de stats
    df = pd.DataFrame({
        'Generation': log.select("gen"),
        'BestFitness': log.select("min"),
        'MeanFitness': log.select("avg"),
    })

    return df, best_cost, num_antennas

# --- Streamlit app ---
st.title("Interfaz Interactiva del GA para Set Covering")

# SideBar de parámetros
st.sidebar.header("Ajuste de parámetros")
pop_size = st.sidebar.slider("Population size",  50, 200, 100, step=10)
max_gen  = st.sidebar.slider("Max generations", 50, 300, 100, step=10)
cx_frac  = st.sidebar.slider("Crossover fraction", 0.5, 0.9, 0.7, step=0.05)
mut_rate = st.sidebar.slider("Mutation rate",  0.005, 0.05, 0.01, step=0.005)

if st.sidebar.button("Ejecutar GA"):
    with st.spinner("Corriendo GA..."):
        df_stats, best_cost, num_ant = run_ga_python(pop_size, max_gen, cx_frac, mut_rate)

    st.subheader("Evolución del fitness")
    st.line_chart(df_stats.set_index('Generation'))

    st.subheader("Resultados finales")
    st.markdown(f"- **Mejor coste:** {best_cost:.0f}")
    st.markdown(f"- **Antenas seleccionadas:** {num_ant}")

    st.subheader("Tabla de estadísticas")
    st.dataframe(df_stats)

else:
    st.info("Ajusta los parámetros y pulsa **Ejecutar GA**.")
