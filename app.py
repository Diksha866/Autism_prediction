from typing import TYPE_CHECKING

# Help static analyzers/IDEs without affecting runtime: when type-checking, import streamlit
if TYPE_CHECKING:
    import streamlit as st  # type: ignore

try:
    import streamlit as st  # type: ignore[import]
except ImportError as e:
    raise ImportError("The 'streamlit' package is required. Install it with 'pip install streamlit'.") from e

from pathlib import Path

import pandas as pd
import pickle

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
ENCODERS_PATH = BASE_DIR / "encoders.pkl"

st.set_page_config(
    page_title="Autism Screening Predictor",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(55, 111, 126, 0.16), transparent 34rem),
                linear-gradient(180deg, #f8fafc 0%, #eef4f8 100%);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        .hero {
            background:
                linear-gradient(135deg, rgba(18, 48, 71, 0.96), rgba(37, 99, 111, 0.93)),
                linear-gradient(45deg, #123047, #5f8d6d);
            color: white;
            padding: 2.2rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            box-shadow: 0 18px 42px rgba(18, 48, 71, 0.18);
        }

        .hero h1 {
            margin: 0 0 0.5rem 0;
            font-size: 2.3rem;
            line-height: 1.12;
            letter-spacing: 0;
        }

        .hero p {
            margin: 0;
            max-width: 760px;
            color: rgba(255, 255, 255, 0.86);
            font-size: 1.02rem;
            line-height: 1.65;
        }

        .top-note {
            display: flex;
            gap: 0.75rem;
            align-items: flex-start;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid #dce6ef;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin-bottom: 1rem;
            color: #334155;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }

        .top-note strong {
            color: #17344a;
        }

        .section-title {
            margin: 0.25rem 0 0.75rem 0;
            color: #17344a;
            font-size: 1.05rem;
            font-weight: 700;
        }

        .helper-text {
            color: #64748b;
            font-size: 0.92rem;
            line-height: 1.55;
            margin: -0.25rem 0 1rem 0;
        }

        div[data-testid="stForm"] {
            background: transparent;
            border: 0;
            padding: 0;
            box-shadow: none;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid #dce6ef;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }

        .stButton > button,
        .stFormSubmitButton > button {
            width: 100%;
            border-radius: 6px;
            min-height: 3rem;
            font-weight: 700;
            background: #174b63;
            color: #ffffff;
            border: 0;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            background: #12394b;
            color: #ffffff;
            border: 0;
        }

        .stSlider {
            padding-right: 0.35rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.9);
            border-color: #dce6ef;
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.07);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not ENCODERS_PATH.exists():
        raise FileNotFoundError(f"Encoders file not found: {ENCODERS_PATH}")

    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)

    with ENCODERS_PATH.open("rb") as f:
        encoders = pickle.load(f)

    return model, encoders


model, encoders = load_artifacts()

st.markdown(
    """
    <div class="hero">
        <h1>Autism Screening Predictor</h1>
        <p>Enter screening scores and profile details to generate a quick, model-based autism risk estimate.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="top-note">
        <span><strong>Note:</strong> This educational tool supports screening only. It does not replace a formal assessment by a qualified healthcare professional.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    left_col, right_col = st.columns([1.25, 1], gap="large")

    with left_col:
        st.markdown('<div class="section-title">Questionnaire Scores</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="helper-text">Adjust each screening item from 1 to 5.</p>',
            unsafe_allow_html=True,
        )

        score_cols = st.columns(5)
        score_values = {}
        for index in range(1, 11):
            with score_cols[(index - 1) % 5]:
                score_values[f"A{index}_Score"] = st.slider(
                    f"A{index}",
                    min_value=1,
                    max_value=5,
                    value=3,
                    step=1,
                )

        total_score = sum(score_values.values())
        average_score = total_score / len(score_values)

        metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
        metric_col_1.metric("Total questionnaire score", total_score)
        metric_col_2.metric("Average score", f"{average_score:.1f}")
        metric_col_3.metric("Items completed", "10/10")

    with right_col:
        st.markdown('<div class="section-title">Profile Details</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="helper-text">Use the same categories that were available in the training data.</p>',
            unsafe_allow_html=True,
        )

        age = st.number_input("Age", min_value=1, max_value=100, value=25)
        gender_label = st.radio("Gender", ["Female", "Male"], horizontal=True)
        gender = "f" if gender_label == "Female" else "m"

        ethnicity = st.selectbox("Ethnicity", encoders["ethnicity"].classes_.tolist())
        contry_of_res = st.selectbox("Country of Residence", encoders["contry_of_res"].classes_.tolist())
        relation = st.selectbox("Relation", encoders["relation"].classes_.tolist())

        st.markdown('<div class="section-title">Clinical History</div>', unsafe_allow_html=True)
        jaundice = st.selectbox("Jaundice", encoders["jaundice"].classes_.tolist())
        austim = st.selectbox("Previous Autism Diagnosis", encoders["austim"].classes_.tolist())
        used_app_before = st.selectbox("Used Screening App Before", encoders["used_app_before"].classes_.tolist())
        result = st.number_input("Previous Screening Result", min_value=0.0, max_value=100.0, value=0.0, step=0.1)

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    gender_encoded = encoders["gender"].transform([gender])[0]
    ethnicity_encoded = encoders["ethnicity"].transform([ethnicity])[0]
    jaundice_encoded = encoders["jaundice"].transform([jaundice])[0]
    austim_encoded = encoders["austim"].transform([austim])[0]
    contry_of_res_encoded = encoders["contry_of_res"].transform([contry_of_res])[0]
    used_app_before_encoded = encoders["used_app_before"].transform([used_app_before])[0]
    relation_encoded = encoders["relation"].transform([relation])[0]

    data = pd.DataFrame({
        **{key: [value] for key, value in score_values.items()},
        "age": [age],
        "gender": [gender_encoded],
        "ethnicity": [ethnicity_encoded],
        "jaundice": [jaundice_encoded],
        "austim": [austim_encoded],
        "contry_of_res": [contry_of_res_encoded],
        "used_app_before": [used_app_before_encoded],
        "result": [result],
        "relation": [relation_encoded],
    })

    prediction = model.predict(data)

    result_container = st.container(border=True)

    with result_container:
        st.markdown("### Prediction Result")

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(data)[0][1]
            st.progress(float(probability))
            st.caption(f"Model confidence for detected class: {probability:.1%}")

        if prediction[0] == 1:
            st.error("Autism risk detected. Please consult a qualified healthcare professional for a formal assessment.")
        else:
            st.success("No autism risk detected by the model.")

        st.caption("This result is generated by a machine-learning model and should not replace professional clinical judgment.")
