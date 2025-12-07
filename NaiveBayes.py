import math

import pandas as pd
import streamlit as st
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# základné nastavenie Streamlit aplikácie
st.set_page_config(page_title="Naive Bayes", layout="centered")
st.title("Naive Bayesova inferenčná sieť")

st.subheader("Načítanie dát")

# nahratie vlastného CSV súboru alebo použitie predvoleného
uploaded_file = st.file_uploader(
    "Nahraj CSV súbor (voliteľné):",
    type=["csv"],
    help="Ak nič nenahráš, použije sa 'weather_forecast.csv'.",
)

# načítanie dát do DataFrame
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.caption("Používa sa nahratý súbor.")
else:
    try:
        df = pd.read_csv("weather_forecast.csv")
        st.caption("Používa sa predvolený súbor 'weather_forecast.csv'.")
    except FileNotFoundError:
        st.error("Súbor 'weather_forecast.csv' sa nenašiel a žiadny CSV nebol nahratý.")
        st.stop()

# rýchla kontrola dát a ich náhľad
st.subheader("Náhľad dát")
st.write(f"Počet riadkov: {df.shape[0]}, počet stĺpcov: {df.shape[1]}")
st.dataframe(df.head())

# výber cieľovej premennej (target)
target_col = st.selectbox("Cieľová premenná (target):", df.columns, index=len(df.columns) - 1)

# ostatné stĺpce budú vstupné premenné (features)
feature_cols = [c for c in df.columns if c != target_col]
st.caption(f"Features: {', '.join(feature_cols)}")

# inicializácia objektov v session_state 
if "model" not in st.session_state:
    st.session_state["model"] = None
    st.session_state["target_col"] = None
    st.session_state["feature_cols"] = None
    st.session_state["df_clean"] = None

st.subheader("Tréning Naive Bayes modelu")
st.write("Pred tréningom odstránim NaN a prevediem stĺpce na string (diskrétne premenné).")

if st.button("Natrénovať model"):
    # kópia pôvodných dát
    df_clean = df.copy()

    # vyhodenie riadkov s chýbajúcimi hodnotami
    df_clean = df_clean.dropna()

    # všetky hodnoty prevediem na text, diskrétne kategórie
    df_clean = df_clean.astype(str)

    # definícia štruktúry siete: všetky features smerujú do targetu
    edges = [(f, target_col) for f in feature_cols]
    model = DiscreteBayesianNetwork(edges)

    # odhad podmienených pravdepodobností z dát 
    model.fit(
        df_clean,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=5.0,
    )

    # kontrola, či model má korektné CPD
    model.check_model()

    # uloženie modelu a nastavení do session_state
    st.session_state["model"] = model
    st.session_state["target_col"] = target_col
    st.session_state["feature_cols"] = feature_cols
    st.session_state["df_clean"] = df_clean

    st.success("Model bol natrénovaný.")
    st.write("Hrany (feature → target):", edges)

    # zobrazenie grafu štruktúry siete (target v strede, features okolo)
    with st.expander("Štruktúra inferenčnej siete"):
        G = nx.DiGraph()
        G.add_nodes_from(model.nodes())
        G.add_edges_from(model.edges())

        # pozície uzlov v grafe
        pos = {}
        pos[target_col] = (0.0, 0.0)
        n_feats = len(feature_cols)
        radius = 2.0

        for i, feat in enumerate(feature_cols):
            angle = 2 * math.pi * i / max(n_feats, 1)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            pos[feat] = (x, y)

        fig, ax = plt.subplots(figsize=(5, 5))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=2000,
            node_color="#a3c4f3",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=15,
            ax=ax,
        )
        ax.set_axis_off()
        st.pyplot(fig)

    # výpočet a priori rozdelenia cieľa (bez evidencie)
    inference = VariableElimination(model)
    prior_distribution = inference.query(variables=[target_col])

    st.subheader("A priori rozdelenie cieľa")
    st.write(prior_distribution)

# inferencia sa robí len vtedy, keď už existuje natrénovaný model
if st.session_state.get("model") is not None:
    model = st.session_state["model"]
    target_col = st.session_state["target_col"]
    feature_cols = st.session_state["feature_cols"]
    df_clean = st.session_state["df_clean"]

    st.subheader("Inferencia s evidenciou")
    st.write("Vyber premenné a hodnoty, ktoré pôjdu ako evidencia.")

    # výber vstupných premenných, ktoré použijeme ako evidenciu
    selected_features = st.multiselect(
        "Premenné ako evidencia:",
        feature_cols,
    )

    # slovník evidencie {premenná: hodnota}
    evidence = {}
    for feat in selected_features:
        possible_vals = sorted(df_clean[feat].unique())
        val = st.selectbox(
            f"Hodnota pre {feat}:",
            possible_vals,
            key=f"evid_{feat}",
        )
        evidence[feat] = val

    if st.button("Vypočítať posterior"):
        inference = VariableElimination(model)

        # ak nie je evidencia, dostaneme prior; inak posterior podmienený na evidenciu
        if evidence:
            posterior = inference.query(variables=[target_col], evidence=evidence)
        else:
            posterior = inference.query(variables=[target_col])

        st.write("Evidencia:", evidence)
        st.subheader("Posterior cieľovej premennej")
        st.write(posterior)

        try:
            # stavy cieľovej premennej a ich pravdepodobnosti
            states = list(posterior.state_names[target_col])
            probs = posterior.values

            # najpravdepodobnejší stav (maximum z posterioru)
            best_idx = probs.argmax()
            best_state = states[best_idx]
            best_prob = probs[best_idx]
            st.write(f"Najpravdepodobnejší stav: {best_state} (p = {best_prob:.3f})")

            # jednoduchý bar graf posterioru
            fig, ax = plt.subplots()
            ax.bar(states, probs)
            ax.set_ylabel("Pravdepodobnosť")
            ax.set_title(f"Posterior pre {target_col}")
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"Nepodarilo sa zobraziť graf posterioru: {e}")