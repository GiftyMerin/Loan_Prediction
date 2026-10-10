import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered",
)

# -----------------------------
# Load model and dataset
# -----------------------------
MODEL_PATH = "models/random_forest.joblib"
DATA_PATH = "data/loan_data_new.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    return df

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Please run train_model.py first.")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found at data/loan_data_new.csv.")
    st.stop()

model = load_model()
df = load_data()

# Use exactly the input features used during training
feature_columns = list(model.feature_names_in_)
df_features = df[feature_columns]

# -----------------------------
# App interface
# -----------------------------
st.title("🏦 Loan Approval Prediction")
st.write(
    "Enter the applicant's details to estimate the loan approval status "
    "using a trained Random Forest classification model."
)

st.info(
    "This is an educational machine learning demonstration, "
    "not an actual bank decision or a guarantee of loan approval."
)

with st.form("loan_prediction_form"):
    st.subheader("Applicant Details")

    inputs = {}

    for column in feature_columns:
        series = df_features[column]

        if (
            pd.api.types.is_numeric_dtype(series)
            and series.nunique(dropna=True) > 10
        ):
            values = pd.to_numeric(series, errors="coerce").dropna()

            if values.empty:
                inputs[column] = st.number_input(
                    column,
                    value=0.0,
                )
                continue

            minimum = float(values.min())
            maximum = float(values.max())
            default = float(values.median())

            # Keep the widget range valid
            if minimum == maximum:
                inputs[column] = st.number_input(
                    column,
                    value=default,
                )
            else:
                inputs[column] = st.number_input(
                    column,
                    min_value=minimum,
                    max_value=maximum,
                    value=min(max(default, minimum), maximum),
                )

        else:
            options = series.dropna().unique().tolist()

            if options:
                options = sorted(options, key=lambda value: str(value))
                inputs[column] = st.selectbox(column, options)
            else:
                inputs[column] = st.text_input(column)

    submitted = st.form_submit_button(
        "Predict Loan Status",
        use_container_width=True,
    )

# -----------------------------
# Make prediction
# -----------------------------
if submitted:
    input_df = pd.DataFrame(
        [inputs],
        columns=feature_columns,
    )

    try:
        prediction = int(model.predict(input_df)[0])

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("✅ Predicted Status: LOAN APPROVED")
        else:
            st.error("❌ Predicted Status: LOAN REJECTED")

        st.caption(
            "The model predicts 1 = Approved and 0 = Rejected."
        )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            if 1 in classes:
                approval_probability = float(
                    probabilities[classes.index(1)]
                )

                st.metric(
                    "Model-estimated approval probability",
                    f"{approval_probability:.1%}",
                )

                st.progress(approval_probability)

                st.caption(
                    "This probability is a model estimate, not a "
                    "guarantee or a verified real-world lending probability."
                )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
