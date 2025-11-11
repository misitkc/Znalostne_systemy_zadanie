import pandas as pd
import streamlit as st
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(page_title="Naive Bayes", layout="centered")
st.title("Naive Bayesova inferenčná sieť")

try:
    df = pd.read_csv("weather_forecast.csv")
except FileNotFoundError:
    st.error("Súbor 'weather_forecast.csv' sa nenašiel v tomto priečinku.")
    st.stop()

st.subheader("Náhľad dát")
st.dataframe(df.head())

target_col = st.selectbox("Cieľ (target):", df.columns, index=len(df.columns)-1)
feature_cols = [c for c in df.columns if c != target_col]
st.caption(f"Features: {', '.join(feature_cols)}")

if st.button("Natrénovať model"):
    df_enc = df.copy()
    for c in df_enc.columns:
        df_enc[c] = pd.Categorical(df_enc[c]).codes

    edges = [(f, target_col) for f in feature_cols]
    model = DiscreteBayesianNetwork(edges)

    model.fit(
        df_enc,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=5.0,
    )
    model.check_model()

    st.success("Model natrénovaný.")
    st.write("Hrany:", edges)

    with st.expander("Zobraziť inferenčnú sieť"):
        G = nx.DiGraph()
        G.add_nodes_from(model.nodes())
        G.add_edges_from(model.edges())

        fig, ax = plt.subplots(figsize=(5, 4))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(G, pos, with_labels=True, node_size=2200, node_color="#89c2ff", ax=ax, arrows=False)
        ax.set_axis_off()
        ax.set_title("Štruktúra Bayesovej siete")
        st.pyplot(fig)

    