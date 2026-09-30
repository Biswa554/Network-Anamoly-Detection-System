"""Beginner-friendly Streamlit interface for the KDD Cup 99 project."""
from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

try:
    from src.data_preprocessing import clean_data, load_sampled_data
    from src.train import run as train_models
except ImportError:  # Also support launching from inside the src directory.
    from data_preprocessing import clean_data, load_sampled_data
    from train import run as train_models


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#e7edf7; --muted:#9aa9bd; --blue:#79a6ff; --line:#263449;
      --panel:#111c2d; --panel-raised:#162338; --canvas:#0a1220; }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
    .stApp { background:var(--canvas); color:var(--ink); }
    [data-testid="stAppViewContainer"] { background:
      radial-gradient(ellipse at 82% 0%, rgba(45,91,161,.13), transparent 36%), var(--canvas); }
    h1,h2,h3 { font-family:'Manrope', sans-serif !important; color:var(--ink); }
    .block-container { padding-top:2.2rem; max-width:1380px; }
    .hero { background:linear-gradient(115deg,#101a2a 0%,#172b49 58%,#244b83 100%);
      border-radius:22px; padding:34px 38px; color:white; margin-bottom:22px;
      border:1px solid #30496c; box-shadow:0 18px 48px rgba(0,0,0,.28); }
    .hero h1 { color:white !important; font-size:2.15rem; margin:0 0 8px 0; }
    .hero p { color:#d9e5fb; margin:0; font-size:1.02rem; max-width:780px; }
    .eyebrow { color:#98b6ff; text-transform:uppercase; letter-spacing:.13em;
      font-weight:700; font-size:.72rem; margin-bottom:10px; }
    .section-kicker { color:#8295b1; text-transform:uppercase; letter-spacing:.12em;
      font-size:.73rem; font-weight:700; margin:8px 0 5px; }
    div[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line);
      border-radius:15px; padding:17px 19px; box-shadow:0 8px 22px rgba(0,0,0,.14); }
    div[data-testid="stMetricLabel"] { color:#a6b4c8; }
    div[data-testid="stMetricValue"] { color:#f2f6fc; }
    .step-card { background:var(--panel); border:1px solid var(--line); border-radius:15px;
      padding:18px 20px; min-height:120px; }
    .step-number { color:var(--blue); font-weight:800; font-size:.78rem; letter-spacing:.1em; }
    .step-card h3 { font-size:1.04rem; margin:.35rem 0; }
    .step-card p { color:var(--muted); font-size:.9rem; margin:0; }
    [data-testid="stSidebar"] { background:#0d1726; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { background:#0d1726; }
    [data-testid="stFileUploaderDropzone"] { background:#111c2d; border:1px dashed #405674; }
    [data-testid="stFileUploaderDropzone"] * { color:#d7e1ef; }
    [data-testid="stTabs"] button { color:#9aa9bd; }
    [data-testid="stTabs"] button[aria-selected="true"] { color:#a8c5ff; }
    [data-testid="stAlert"] { background:#111c2d; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
    .stButton > button, [data-testid="stDownloadButton"] button,
    [data-testid="stFormSubmitButton"] button { border-radius:10px; font-weight:650; }
    .small-note { color:#9aa9bd; font-size:.83rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _clear_run() -> None:
    st.session_state.pop("metrics", None)
    st.session_state.pop("models_ready", None)
    st.session_state.pop("trained_fingerprint", None)


st.sidebar.markdown("## 🛡️ Intrusion Lab")
st.sidebar.caption("A guided workspace for exploring KDD Cup 99 traffic.")
st.sidebar.markdown("### 1 · Add your dataset")
uploaded = st.sidebar.file_uploader(
    "KDD Cup 99 connection data",
    type=["gz", "csv", "corrected", "data"],
    help="Use the 42-column connection-level KDD Cup 99 file. A dataset catalog CSV is not the traffic data.",
)
st.sidebar.markdown(
    '<p class="small-note">Accepted: the headerless KDD connection file or a standard 42-column CSV. '
    "The app takes a reproducible 12,000-row sample (seed 42).</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("### 2 · Run the experiment")
st.sidebar.caption("Training compares Random Forest, SVM, Isolation Forest, and One-Class SVM. Full tuning can take a few minutes.")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Network security · Machine learning</div>
      <h1>See the signal in network traffic.</h1>
      <p>Explore a reproducible intrusion-detection experiment, compare four models, and try a sample prediction—without needing to edit code.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

step_cols = st.columns(3)
steps = [
    ("01 · DATA", "Upload a dataset", "Choose the KDD Cup 99 connection data file in the sidebar."),
    ("02 · TRAIN", "Run the comparison", "Preprocessing stays inside the model pipelines to prevent train/test leakage."),
    ("03 · REVIEW", "Explore the results", "Review class balance, held-out metrics, confusion matrices, ROC curves, and feature importance."),
]
for col, (number, title, description) in zip(step_cols, steps):
    with col:
        st.markdown(
            f'<div class="step-card"><div class="step-number">{number}</div>'
            f'<h3>{title}</h3><p>{description}</p></div>',
            unsafe_allow_html=True,
        )

st.markdown("<div class='section-kicker'>Your workspace</div>", unsafe_allow_html=True)
overview_tab, results_tab, predict_tab = st.tabs(["① Data overview", "② Model results", "③ Try a prediction"])

if "raw_sample" not in st.session_state:
    st.session_state.raw_sample = None

if uploaded is not None:
    fingerprint = (uploaded.name, uploaded.size)
    if st.session_state.get("upload_fingerprint") != fingerprint:
        suffix = ".gz" if uploaded.name.lower().endswith(".gz") else ".csv"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded.getvalue())
                temp_path = Path(temp_file.name)
            with st.spinner("Checking the file and selecting the reproducible 12,000-row sample…"):
                sample = load_sampled_data(temp_path)
            st.session_state.raw_sample = sample
            st.session_state.upload_fingerprint = fingerprint
            _clear_run()
        except Exception as exc:
            st.session_state.raw_sample = None
            st.session_state.upload_fingerprint = None
            st.sidebar.error(f"Could not load this file: {exc}")
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()

with overview_tab:
    if st.session_state.raw_sample is None:
        st.info("Start by uploading the 42-column KDD Cup 99 network-connection dataset in the sidebar.")
        st.markdown(
            """
            **New here?** The file named `KDD Cup 99.csv` that lists many datasets is a catalog, not the traffic records. 
            Use the KDD connection data file (for example, `kddcup.data.gz`).
            """
        )
        with st.expander("What happens when I upload a file?"):
            st.write("The app checks the KDD column count, takes exactly 12,000 rows with seed 42, normalizes the labels, and prepares a preview. It does not upload your dataset anywhere.")
    else:
        sample = st.session_state.raw_sample
        cleaned = clean_data(sample)
        normal_count = int((cleaned["target"] == 0).sum())
        attack_count = int((cleaned["target"] == 1).sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Sampled connections", f"{len(sample):,}")
        m2.metric("Normal", f"{normal_count:,}", f"{normal_count / len(sample):.1%} of sample")
        m3.metric("Attack", f"{attack_count:,}", f"{attack_count / len(sample):.1%} of sample")
        left, right = st.columns([1.1, 1])
        with left:
            st.markdown("#### Sample preview")
            st.dataframe(sample.head(8), use_container_width=True, hide_index=True)
        with right:
            st.markdown("#### Class balance")
            balance = cleaned["target"].map({0: "Normal", 1: "Attack"}).value_counts().rename_axis("Traffic").reset_index(name="Connections")
            fig, ax = plt.subplots(figsize=(6, 3.7))
            color_map = {"Normal": "#42b883", "Attack": "#f07469"}
            ax.bar(balance["Traffic"], balance["Connections"],
                   color=[color_map[name] for name in balance["Traffic"]], width=.58)
            ax.set_xlabel("")
            ax.set_ylabel("Connections")
            ax.grid(axis="y", alpha=.18)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with st.expander("Data checks and feature types"):
            st.write("Categorical fields: `protocol_type`, `service`, and `flag`. The remaining 38 fields are numeric.")
            st.write(f"Missing values in uploaded sample: **{int(sample.isna().sum().sum())}**. Invalid numeric values are imputed within each training pipeline.")
            st.dataframe(sample.dtypes.rename("Data type").to_frame(), use_container_width=True)

    st.markdown("#### Train and evaluate")
    if st.session_state.raw_sample is None:
        st.button("Train models", disabled=True, use_container_width=True)
    else:
        if st.button("Train all four models", type="primary", use_container_width=True):
            try:
                with st.spinner("Training and tuning models… This can take several minutes."):
                    metrics = train_models(
                        output_dir=str(ROOT),
                        sampled_df=st.session_state.raw_sample,
                    )
                st.session_state.metrics = metrics
                st.session_state.models_ready = True
                st.session_state.trained_fingerprint = st.session_state.get("upload_fingerprint")
                st.success("Training is complete. Open **Model results** to review the held-out scores and charts.")
            except Exception as exc:
                st.error(f"Training could not complete: {exc}")

with results_tab:
    metrics = st.session_state.get("metrics")
    if metrics is None:
        st.info("Upload the dataset and run training to see your model comparison here.")
    else:
        st.markdown("#### Held-out test results")
        st.caption("Scores are calculated from the sampled experiment. KDD Cup 99 is a historical benchmark; high random-split scores do not guarantee performance on current network traffic.")
        show = metrics.set_index("Model") if "Model" in metrics.columns else metrics.copy()
        st.dataframe(show.style.format("{:.4f}"), use_container_width=True)
        st.download_button(
            "Download metrics CSV",
            data=show.to_csv().encode("utf-8"),
            file_name="kdd99_model_comparison.csv",
            mime="text/csv",
        )
        chart_col, cm_col = st.columns([1.1, .9])
        comparison_path = RESULTS_DIR / "model_comparison.png"
        with chart_col:
            st.markdown("#### Model comparison")
            if comparison_path.exists():
                st.image(str(comparison_path), use_container_width=True)
        with cm_col:
            st.markdown("#### Confusion matrix")
            model_files = {
                "Random Forest": "confusion_matrix_random_forest.png",
                "SVM": "confusion_matrix_svm.png",
                "Isolation Forest": "confusion_matrix_isolation_forest.png",
                "One-Class SVM": "confusion_matrix_one_class_svm.png",
            }
            selected_model = st.selectbox("Choose a model", list(model_files))
            matrix_path = RESULTS_DIR / model_files[selected_model]
            if matrix_path.exists():
                st.image(str(matrix_path), use_container_width=True)
        roc_col, importance_col = st.columns(2)
        with roc_col:
            st.markdown("#### ROC curves")
            if (RESULTS_DIR / "roc_curve.png").exists():
                st.image(str(RESULTS_DIR / "roc_curve.png"), use_container_width=True)
        with importance_col:
            st.markdown("#### Random Forest feature importance")
            if (RESULTS_DIR / "feature_importance.png").exists():
                st.image(str(RESULTS_DIR / "feature_importance.png"), use_container_width=True)

with predict_tab:
    st.markdown("#### Classify one connection")
    st.caption("After training, enter a few known connection fields. Missing fields are imputed by the selected fitted pipeline.")
    available_models = [
        (name, file_name) for name, file_name in [
            ("Random Forest", "random_forest.pkl"),
            ("SVM", "svm.pkl"),
            ("Isolation Forest", "isolation_forest.pkl"),
            ("One-Class SVM", "one_class_svm.pkl"),
        ] if (MODELS_DIR / file_name).exists()
    ]
    ready_for_current_upload = (
        st.session_state.get("trained_fingerprint") is not None
        and st.session_state.get("trained_fingerprint") == st.session_state.get("upload_fingerprint")
    )
    if not available_models or not ready_for_current_upload:
        st.info("Train the models first. The prediction form will appear once model pipelines are saved.")
    else:
        with st.form("prediction_form"):
            chosen_name = st.selectbox("Model", [x[0] for x in available_models])
            c1, c2, c3 = st.columns(3)
            duration = c1.number_input("Duration (seconds)", min_value=0, value=0, step=1)
            protocol = c2.selectbox("Protocol", ["tcp", "udp", "icmp"])
            service = c3.text_input("Service", value="http", help="Examples: http, private, domain_u")
            flag = c1.text_input("Connection flag", value="sf", help="Examples: sf, s0, rst_o")
            src_bytes = c2.number_input("Source bytes", min_value=0, value=181, step=1)
            dst_bytes = c3.number_input("Destination bytes", min_value=0, value=5450, step=1)
            submitted = st.form_submit_button("Predict traffic class", type="primary", use_container_width=True)
        if submitted:
            _, model_file = next(x for x in available_models if x[0] == chosen_name)
            model = joblib.load(MODELS_DIR / model_file)
            record = {
                "duration": duration,
                "protocol_type": protocol,
                "service": service.strip().lower(),
                "flag": flag.strip().lower(),
                "src_bytes": src_bytes,
                "dst_bytes": dst_bytes,
            }
            frame = pd.DataFrame([record])
            prediction = int(model.predict(frame)[0])
            is_anomaly_detector = chosen_name in {"Isolation Forest", "One-Class SVM"}
            malicious = prediction == -1 if is_anomaly_detector else prediction == 1
            if malicious:
                st.error("🚨 Malicious Traffic")
            else:
                st.success("✅ Normal Traffic")
            st.caption("This example fills unspecified features using training-learned imputers. Validate predictions against complete, representative traffic before operational use.")

st.divider()
st.caption("Network Intrusion Detection Using Machine Learning · KDD Cup 99 · Reproducible sample seed 42")
