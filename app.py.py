import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ReCare AI",
    page_icon="🏥",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("models/recare_xgb_model.pkl")


model = load_model()

# =========================================================
# HEADER
# =========================================================

st.title("🏥 ReCare AI")

st.subheader(
    "Explainable & Fair Hospital Readmission Risk Prediction"
)

st.markdown(
    """
    **ReCare AI** estimates the risk of **30-day hospital readmission**
    to support discharge planning and follow-up prioritization.
    """
)

st.warning(
    "⚠️ Prototype decision-support tool. "
    "This risk estimate is not clinically validated and does not replace "
    "professional medical judgment."
)

# =========================================================
# 1. PATIENT INFORMATION
# =========================================================

st.header("1. Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    race = st.selectbox(
        "Race",
        ["Caucasian", "AfricanAmerican", "Asian",
         "Hispanic", "Other", "Unknown"]
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col3:
    age = st.selectbox(
        "Age Group",
        [
            "[0-10)", "[10-20)", "[20-30)", "[30-40)",
            "[40-50)", "[50-60)", "[60-70)", "[70-80)",
            "[80-90)", "[90-100)"
        ],
        index=5
    )

# =========================================================
# 2. HOSPITALIZATION
# =========================================================

st.header("2. Hospitalization Information")

col1, col2, col3 = st.columns(3)

with col1:
    time_in_hospital = st.number_input(
        "Time in Hospital (days)", 1, 14, 4
    )

with col2:
    num_lab_procedures = st.number_input(
        "Number of Lab Procedures", 1, 132, 40
    )

with col3:
    num_procedures = st.number_input(
        "Number of Procedures", 0, 6, 1
    )

col1, col2, col3 = st.columns(3)

with col1:
    num_medications = st.number_input(
        "Number of Medications", 1, 81, 16
    )

with col2:
    number_diagnoses = st.number_input(
        "Number of Diagnoses", 1, 16, 7
    )

with col3:
    admission_type_id = st.selectbox(
        "Admission Type",
        ["1", "2", "3", "4", "5", "6", "7", "8"]
    )

# =========================================================
# 3. PREVIOUS HEALTHCARE UTILIZATION
# =========================================================

st.header("3. Previous Healthcare Utilization")

col1, col2, col3 = st.columns(3)

with col1:
    number_outpatient = st.number_input(
        "Previous Outpatient Visits", 0, 42, 0
    )

with col2:
    number_emergency = st.number_input(
        "Previous Emergency Visits", 0, 76, 0
    )

with col3:
    number_inpatient = st.number_input(
        "Previous Inpatient Visits", 0, 21, 0
    )

# =========================================================
# 4. DIABETES MANAGEMENT
# =========================================================

st.header("4. Diabetes Management")

col1, col2, col3 = st.columns(3)

with col1:
    diabetes_med = st.selectbox(
        "Diabetes Medication Prescribed",
        ["Yes", "No"]
    )

with col2:
    insulin = st.selectbox(
        "Insulin Treatment",
        ["No", "Up", "Down", "Steady"]
    )

with col3:
    metformin = st.selectbox(
        "Metformin Treatment",
        ["No", "Steady", "Up", "Down"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    A1Cresult = st.selectbox(
        "HbA1c Test Result",
        ["None", "Norm", ">7", ">8"]
    )

with col2:
    max_glu_serum = st.selectbox(
        "Blood Glucose Test Result",
        ["None", "Norm", ">200", ">300"]
    )

# =========================================================
# SHAP HELPERS
# =========================================================

def get_shap_explanation(model, input_df):
    """
    Explain one prediction for a tree model or a pipeline ending in a tree model.
    Returns an explanation and display values, or raises an informative error.
    """
    # For a scikit-learn Pipeline, transform using its fitted preprocessing.
    if hasattr(model, "steps") and hasattr(model, "named_steps") and len(model.steps) >= 2:
        tree_model = model.steps[-1][1]
        preprocessing = model[:-1]
        model_input = preprocessing.transform(input_df)

        if hasattr(model_input, "toarray"):
            model_input = model_input.toarray()
        model_input = np.asarray(model_input)

        try:
            feature_names = list(preprocessing.get_feature_names_out())
        except Exception:
            feature_names = [f"feature_{i+1}" for i in range(model_input.shape[1])]
    else:
        # Standalone model: this assumes the model can explain the same input
        # representation that was passed to predict_proba().
        tree_model = model
        model_input = input_df
        feature_names = list(input_df.columns)

    explainer = shap.TreeExplainer(tree_model)
    shap_result = explainer(model_input)

    # Normalize SHAP versions / binary-class output shapes.
    if isinstance(shap_result, list):
        selected = shap_result[1] if len(shap_result) > 1 else shap_result[0]
        values = np.asarray(selected.values)
        base_values = np.asarray(selected.base_values)
    else:
        values = np.asarray(shap_result.values)
        base_values = np.asarray(shap_result.base_values)

        # Some SHAP versions/models return rows x features x classes.
        if values.ndim == 3:
            class_index = 1 if values.shape[-1] > 1 else 0
            values = values[:, :, class_index]
            if base_values.ndim >= 2:
                base_values = base_values[..., class_index]

    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] != 1:
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")

    if len(feature_names) != values.shape[1]:
        feature_names = [f"feature_{i+1}" for i in range(values.shape[1])]

    display_input = np.asarray(model_input)
    if display_input.ndim == 1:
        display_input = display_input.reshape(1, -1)
    display_input = display_input[0].reshape(-1)

    base_value = float(np.asarray(base_values).reshape(-1)[0])
    explanation = shap.Explanation(
        values=values[0],
        base_values=base_value,
        data=display_input,
        feature_names=feature_names,
    )
    return explanation, feature_names, values[0], display_input


# =========================================================
# 5. PREDICTION
# =========================================================

st.header("5. Readmission Risk")

if st.button("🔍 Predict Readmission Risk", type="primary"):

    expected_columns = list(model.feature_names_in_)

    data = {}
    numeric_columns = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses"
    ]

    for column in expected_columns:
        if column in numeric_columns:
            data[column] = 0
        else:
            data[column] = "No"

    data["race"] = race
    data["gender"] = gender
    data["age"] = age

    data["admission_type_id"] = admission_type_id
    data["admission_source_id"] = "7"

    data["time_in_hospital"] = time_in_hospital
    data["num_lab_procedures"] = num_lab_procedures
    data["num_procedures"] = num_procedures
    data["num_medications"] = num_medications

    data["number_outpatient"] = number_outpatient
    data["number_emergency"] = number_emergency
    data["number_inpatient"] = number_inpatient
    data["number_diagnoses"] = number_diagnoses

    data["diag_1"] = "428"
    data["diag_2"] = "250"
    data["diag_3"] = "250"

    data["diabetesMed"] = diabetes_med
    data["insulin"] = insulin
    data["metformin"] = metformin
    data["A1Cresult"] = A1Cresult
    data["max_glu_serum"] = max_glu_serum

    input_df = pd.DataFrame([data])
    input_df = input_df[expected_columns]

    if list(input_df.columns) != expected_columns:
        st.error("Input columns do not match the trained model.")
        st.stop()

    probability = model.predict_proba(input_df)[0][1]
    risk_percentage = probability * 100

    if probability < 0.30:
        risk_category = "LOW RISK"
    elif probability < 0.60:
        risk_category = "MODERATE RISK"
    else:
        risk_category = "HIGH RISK"

    st.success("Prediction completed successfully!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Estimated 30-Day Readmission Risk",
            f"{risk_percentage:.1f}%"
        )

    with col2:
        st.metric("Risk Category", risk_category)

    # =====================================================
    # SHAP: WHY THIS PATIENT RECEIVED THIS PREDICTION
    # =====================================================

    st.subheader("Why did the model make this prediction?")
    st.write(
        "The SHAP explanation shows which model inputs pushed the model output "
        "higher or lower for this patient. These are model contributions, not "
        "proof that a feature caused readmission."
    )

    try:
        explanation, feature_names, shap_values, feature_values = (
            get_shap_explanation(model, input_df)
        )

        st.markdown("**Individual patient explanation**")
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=12, show=False)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        contributions = pd.DataFrame({
            "Feature": feature_names,
            "SHAP contribution": shap_values,
            "Model input value": [str(v) for v in feature_values],
        })
        contributions["Absolute contribution"] = contributions["SHAP contribution"].abs()
        contributions = contributions.sort_values(
            "Absolute contribution", ascending=False
        ).head(10)
        contributions["Direction"] = np.where(
            contributions["SHAP contribution"] >= 0,
            "Pushes model output higher",
            "Pushes model output lower",
        )

        st.markdown("**Top contributing features**")
        st.dataframe(
            contributions[
                ["Feature", "Model input value", "SHAP contribution", "Direction"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "SHAP explains the model's output. Depending on the model and explainer, "
            "the plot's scale may be log-odds or another model-output scale rather "
            "than the displayed probability percentage."
        )

    except Exception as e:
        st.warning(
            "Prediction succeeded, but the SHAP explanation could not be generated. "
            "The saved model may use an unsupported model type or input preprocessing."
        )
        st.caption(f"Technical detail: {type(e).__name__}: {e}")

    # =====================================================
    # FOLLOW-UP PRIORITY
    # =====================================================

    st.subheader("Follow-up Priority")

    if risk_category == "HIGH RISK":
        st.error(
            "🔴 HIGH PRIORITY — Consider enhanced follow-up "
            "and discharge-planning review."
        )
    elif risk_category == "MODERATE RISK":
        st.warning(
            "🟠 MODERATE PRIORITY — Consider additional "
            "follow-up and discharge-planning review."
        )
    else:
        st.success(
            "🟢 LOWER PRIORITY — Standard follow-up may be appropriate."
        )

    st.caption(
        "Prototype thresholds only. This system has not been clinically validated."
    )
