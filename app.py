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

st.title("Autism Prediction App")

A1_Score = st.number_input("A1 Score", min_value=1, max_value=5, value=3)
A2_Score = st.number_input("A2 Score", min_value=1, max_value=5, value=3)
A3_Score = st.number_input("A3 Score", min_value=1, max_value=5, value=3)
A4_Score = st.number_input("A4 Score", min_value=1, max_value=5, value=3)
A5_Score = st.number_input("A5 Score", min_value=1, max_value=5, value=3)
A6_Score = st.number_input("A6 Score", min_value=1, max_value=5, value=3)
A7_Score = st.number_input("A7 Score", min_value=1, max_value=5, value=3)
A8_Score = st.number_input("A8 Score", min_value=1, max_value=5, value=3)
A9_Score = st.number_input("A9 Score", min_value=1, max_value=5, value=3)
A10_Score = st.number_input("A10 Score", min_value=1, max_value=5, value=3)

age = st.number_input("Age", min_value=1, max_value=100, value=25)

gender_label = st.selectbox("Gender", ["Female", "Male"])

gender = "f" if gender_label == "Female" else "m"

ethnicity = st.selectbox("Ethnicity", encoders["ethnicity"].classes_.tolist())

jaundice = st.selectbox("Jaundice", encoders["jaundice"].classes_.tolist())

austim = st.selectbox("Autism Diagnosis", encoders["austim"].classes_.tolist())

contry_of_res = st.selectbox("Country of Residence", encoders["contry_of_res"].classes_.tolist())

used_app_before = st.selectbox("Used App Before", encoders["used_app_before"].classes_.tolist())

result = st.number_input("Result", min_value=0.0, max_value=100.0, value=0.0, step=0.1)

relation = st.selectbox("Relation", encoders["relation"].classes_.tolist())

if st.button("Predict"):

    gender_encoded = encoders["gender"].transform([gender])[0]
    ethnicity_encoded = encoders["ethnicity"].transform([ethnicity])[0]
    jaundice_encoded = encoders["jaundice"].transform([jaundice])[0]
    austim_encoded = encoders["austim"].transform([austim])[0]
    contry_of_res_encoded = encoders["contry_of_res"].transform([contry_of_res])[0]
    used_app_before_encoded = encoders["used_app_before"].transform([used_app_before])[0]
    relation_encoded = encoders["relation"].transform([relation])[0]

    data = pd.DataFrame({
        "A1_Score": [A1_Score],
        "A2_Score": [A2_Score],
        "A3_Score": [A3_Score],
        "A4_Score": [A4_Score],
        "A5_Score": [A5_Score],
        "A6_Score": [A6_Score],
        "A7_Score": [A7_Score],
        "A8_Score": [A8_Score],
        "A9_Score": [A9_Score],
        "A10_Score": [A10_Score],
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

    if prediction[0] == 1:
        st.error("Autism Detected")
    else:
        st.success("No Autism Detected")