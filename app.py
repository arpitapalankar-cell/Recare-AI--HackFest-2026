import streamlit as st
import pandas as pd
import joblib

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
# PATIENT INFORMATION
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
# HOSPITALIZATION
# =========================================================

st.header("2. Hospitalization Information")

col1, col2, col3 = st.columns(3)

with col1:
    time_in_hospital = st.number_input(
        "Time in Hospital (days)",
        1, 14, 4
    )

with col2:
    num_lab_procedures = st.number_input(
        "Number of Lab Procedures",
        1, 132, 40
    )

with col3:
    num_procedures = st.number_input(
        "Number of Procedures",
        0, 6, 1
    )

col1, col2, col3 = st.columns(3)

with col1:
    num_medications = st.number_input(
        "Number of Medications",
        1, 81, 16
    )

with col2:
    number_diagnoses = st.number_input(
        "Number of Diagnoses",
        1, 16, 7
    )

with col3:
    admission_type_id = st.selectbox(
        "Admission Type",
        ["1", "2", "3", "4", "5", "6", "7", "8"]
    )

# =========================================================
# PREVIOUS HEALTHCARE UTILIZATION
# =========================================================

st.header("3. Previous Healthcare Utilization")

col1, col2, col3 = st.columns(3)

with col1:
    number_outpatient = st.number_input(
        "Previous Outpatient Visits",
        0, 42, 0
    )

with col2:
    number_emergency = st.number_input(
        "Previous Emergency Visits",
        0, 76, 0
    )

with col3:
    number_inpatient = st.number_input(
        "Previous Inpatient Visits",
        0, 21, 0
    )

# =========================================================
# DIABETES MANAGEMENT
# =========================================================

st.header("4. Diabetes Management")

col1, col2, col3 = st.columns(3)

with col1:
    diabetes_med = st.selectbox(
        "Diabetes Medication",
        ["Yes", "No"]
    )

with col2:
    insulin = st.selectbox(
        "Insulin",
        ["No", "Up", "Down", "Steady"]
    )

with col3:
    A1Cresult = st.selectbox(
        "A1C Result",
        ["None", "Norm", ">7", ">8", "Unknown"]
    )

# =========================================================
# PREDICTION
# =========================================================

st.header("5. Readmission Risk")

if st.button("🔍 Predict Readmission Risk", type="primary"):

    # -----------------------------------------------------
    # GET EXACT MODEL COLUMNS
    # -----------------------------------------------------

    expected_columns = list(model.feature_names_in_)

    # -----------------------------------------------------
    # CREATE DEFAULT VALUES
    # -----------------------------------------------------

    data = {}

    for column in expected_columns:

        if column in [
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses"
        ]:
            data[column] = 0

        else:
            data[column] = "No"

    # -----------------------------------------------------
    # OVERWRITE WITH USER INPUT
    # -----------------------------------------------------

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

    data["max_glu_serum"] = "Unknown"
    data["A1Cresult"] = A1Cresult

    data["insulin"] = insulin
    data["diabetesMed"] = diabetes_med

    # -----------------------------------------------------
    # CREATE DATAFRAME
    # -----------------------------------------------------

    input_df = pd.DataFrame([data])

    # Force exact model order
    input_df = input_df[expected_columns]

    # -----------------------------------------------------
    # SAFETY CHECK
    # -----------------------------------------------------

    if list(input_df.columns) != expected_columns:

        st.error("Input columns do not match the trained model.")

        st.stop()

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    probability = model.predict_proba(input_df)[0][1]

    risk_percentage = probability * 100

    # -----------------------------------------------------
    # RISK CATEGORY
    # -----------------------------------------------------

    if probability < 0.30:
        risk_category = "LOW RISK"

    elif probability < 0.60:
        risk_category = "MODERATE RISK"

    else:
        risk_category = "HIGH RISK"

    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    st.success("Prediction completed successfully!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Estimated 30-Day Readmission Risk",
            f"{risk_percentage:.1f}%"
        )

    with col2:
        st.metric(
            "Risk Category",
            risk_category
        )

    # -----------------------------------------------------
    # FOLLOW-UP
    # -----------------------------------------------------

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
        "Prototype thresholds only. This system has not been "
        "clinically validated."
    )