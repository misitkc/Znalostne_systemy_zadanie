import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

# -----------------------------------------
# ZÁKLADNÉ NASTAVENIE
# -----------------------------------------
st.set_page_config(page_title="Naive Bayes", page_icon="🌤️", layout="wide")

st.title("Naive Bayesova inferenčná sieť")
st.write(
    "Načítaš dáta, vyberieš cieľovú premennú, natrénuješ Naive Bayes a potom skúšaš rôznu evidenciu."
)

# -----------------------------------------
# NAHRANIE / NAČÍTANIE DÁT
# -----------------------------------------
st.sidebar.header("Dáta")

uploaded_file = st.sidebar.file_uploader(
    "Nahraj CSV (voliteľné):",
    type=["csv"],
    help="Ak nič nenahráš, použije sa 'weather_forecast.csv' z priečinka.",
)

DEFAULT_PATH = "weather_forecast.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_data_from_upload(file) -> pd.DataFrame:
    return pd.read_csv(file)


if uploaded_file is not None:
    df = load_data_from_upload(uploaded_file)
    data_source = "Nahraný súbor"
else:
    try:
        df = load_data(DEFAULT_PATH)
        data_source = f"Lokálny súbor: {DEFAULT_PATH}"
    except FileNotFoundError:
        st.error("Súbor 'weather_forecast.csv' sa nenašiel a žiadne CSV nebolo nahraté.")
        st.stop()

# -----------------------------------------
# PREHĽAD DÁT + VÝBER CIEĽA
# -----------------------------------------
c1, c2 = st.columns([1.4, 1])

with c1:
    st.subheader("Dáta")
    st.caption(f"Zdroj dát: {data_source}")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Počet riadkov", len(df))
    with m2:
        st.metric("Počet stĺpcov", df.shape[1])
    with m3:
        st.metric("Typ dát (očakávané)", "Kategórie")

    with st.expander("Náhľad datasetu"):
        st.dataframe(df.head(), use_container_width=True)

with c2:
    st.subheader("Cieľ a features")
    target_col = st.selectbox(
        "Cieľová premenná (target):",
        df.columns,
        index=len(df.columns) - 1,
    )
    feature_cols = [c for c in df.columns if c != target_col]
    st.write("Features:", ", ".join(feature_cols))

st.divider()

# -----------------------------------------
# SESSION STATE PRE MODEL
# -----------------------------------------
if "model" not in st.session_state:
    st.session_state["model"] = None
    st.session_state["target_col"] = None
    st.session_state["feature_cols"] = None

# -----------------------------------------
# TRÉNING MODELU + PRIOR
# -----------------------------------------
c_train, c_prior = st.columns([1, 1.2])

with c_train:
    st.subheader("Tréning Naive Bayes modelu")
    st.write(
        "Predpoklad: všetky features sú podmienečne nezávislé, "
        "ak poznáme cieľovú premennú."
    )

    if st.button("Natrénovať model"):
        edges = [(f, target_col) for f in feature_cols]
        model = DiscreteBayesianNetwork(edges)

        model.fit(
            df,
            estimator=BayesianEstimator,
            prior_type="BDeu",
            equivalent_sample_size=5.0,
        )
        model.check_model()

        st.session_state["model"] = model
        st.session_state["target_col"] = target_col
        st.session_state["feature_cols"] = feature_cols

        st.success("Model bol úspešne natrénovaný.")
        with st.expander("Zobraziť štruktúru (hrany)"):
            st.code(edges, language="python")

with c_prior:
    st.subheader("A priori rozdelenie cieľa")
    if st.session_state["model"] is None:
        st.info("Najprv natrénuj model.")
    else:
        model = st.session_state["model"]
        infer = VariableElimination(model)
        prior = infer.query(variables=[target_col])

        st.write(prior)

        states = list(prior.state_names[target_col])
        probs = prior.values

        fig, ax = plt.subplots()
        ax.bar(states, probs)
        ax.set_ylabel("Pravdepodobnosť")
        ax.set_title(f"A priori rozdelenie pre {target_col}")
        st.pyplot(fig)

# -----------------------------------------
# ŠTRUKTÚRA SIETE
# -----------------------------------------
if st.session_state["model"] is not None:
    st.subheader("Štruktúra Bayesovej siete")
    with st.expander("Zobraziť graf"):
        model = st.session_state["model"]

        G = nx.DiGraph()
        G.add_nodes_from(model.nodes())
        G.add_edges_from(model.edges())

        fig, ax = plt.subplots(figsize=(5, 4))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(G, pos, with_labels=True, node_size=2000, ax=ax, arrows=False)
        ax.set_axis_off()
        st.pyplot(fig)

st.divider()

# -----------------------------------------
# INFERENCIA S EVIDENCIOU
# -----------------------------------------
st.subheader("Inferencia s evidenciou")

if st.session_state["model"] is None:
    st.info("Najprv natrénuj model.")
else:
    model = st.session_state["model"]
    target_col = st.session_state["target_col"]
    feature_cols = st.session_state["feature_cols"]

    col_evid, col_res = st.columns([1, 1.3])

    with col_evid:
        st.write("1. Zadaj evidenciu")

        selected_features = st.multiselect(
            "Premenné ako evidencia:",
            feature_cols,
        )

        evidence = {}
        for feat in selected_features:
            values = sorted(df[feat].unique())
            val = st.selectbox(
                f"Hodnota pre {feat}:",
                values,
                key=f"evid_{feat}",
            )
            evidence[feat] = val

        compute = st.button("Vypočítať posterior")

    with col_res:
        if compute:
            infer = VariableElimination(model)

            if evidence:
                posterior = infer.query(variables=[target_col], evidence=evidence)
            else:
                posterior = infer.query(variables=[target_col])

            st.write("Evidencia:", evidence if evidence else "žiadna (len prior)")
            st.subheader("Posterior cieľovej premennej")
            st.write(posterior)

            # porovnanie prior vs posterior
            try:
                prior = infer.query(variables=[target_col])

                prior_states = list(prior.state_names[target_col])
                prior_probs = prior.values

                post_states = list(posterior.state_names[target_col])
                post_probs = posterior.values

                post_dict = {s: p for s, p in zip(post_states, post_probs)}
                post_probs_aligned = [post_dict.get(s, 0.0) for s in prior_states]

                comp_df = pd.DataFrame(
                    {
                        "Hodnota": prior_states,
                        "Prior": prior_probs,
                        "Posterior": post_probs_aligned,
                    }
                )
                st.write("Porovnanie prior vs posterior")
                st.table(comp_df)

                fig, ax = plt.subplots()
                x = range(len(prior_states))
                width = 0.35

                ax.bar([i - width / 2 for i in x], prior_probs, width, label="Prior")
                ax.bar([i + width / 2 for i in x], post_probs_aligned, width, label="Posterior")
                ax.set_xticks(list(x))
                ax.set_xticklabels(prior_states)
                ax.set_ylabel("Pravdepodobnosť")
                ax.set_title(f"Porovnanie pre {target_col}")
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.warning(f"Nepodarilo sa zobraziť porovnanie: {e}")