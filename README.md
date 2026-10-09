# Recare-AI--HackFest-2026
# ReCare AI 🏥

### Explainable and Fair 30-Day Hospital Readmission Risk Prediction

## KLE TECH HACKFEST 2026 
Problem Statement 6(B): Hospital Readmission Risk Prediction

 Project Overview
ReCare AI is a machine-learning-based decision-support prototype designed to predict the risk of hospital readmission within 30 days of discharge. It uses historical hospital encounter data to compare predictive models, explain model predictions, and assess performance disparities across demographic groups.

The project aims to support healthcare professionals and discharge planners in identifying patients who may benefit from targeted follow-up and additional care planning.

# Objectives

* Predict the likelihood of hospital readmission within 30 days.
* Compare Logistic Regression, Random Forest, and XGBoost models.
* Evaluate model performance using accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC.
* Investigate class imbalance and improve minority-class detection.
* Use SHAP to explain model predictions.
* Assess fairness across age, gender, and race.
* Develop an interactive dashboard for risk visualization and follow-up prioritization.

##  Dataset

**Dataset:** [UCI Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

* **Records:** 101,766 hospital encounters
* **Original features:** 50 variables
* **Data period:** 1999–2008
* **Prediction task:** Binary classification of 30-day readmission

### Target Variable

The original readmission variable is transformed into a binary target:

* `<30` → 1: Readmitted within 30 days
* `>30` or `NO` → 0: Not readmitted within 30 days

The positive class represents approximately 11.16% of encounters, making class imbalance an important modelling challenge.

##  Methodology

### 1. Data Preprocessing
* Inspect missing values, duplicate records, and data quality.
* Handle missing values and outliers using documented rationale.
* Encode categorical variables and process numerical features.
* Investigate repeated patient encounters and potential data leakage.
* Document feature selection and exclusions.

### 2. Predictive Modelling

Three model types are included in the initial comparison:

* **Logistic Regression:** Baseline classification model.
* **Random Forest:** Tree-based ensemble model.
* **XGBoost:** Gradient-boosting model.

### 3. Model Evaluation

Evaluate and compare models using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Precision-Recall AUC (PR-AUC)

Model selection will consider the clinical use case, particularly the balance between missed readmissions and unnecessary follow-up alerts, rather than accuracy alone.

### 4. Explainable AI

SHAP (SHapley Additive exPlanations) is used to examine how individual features contribute to model predictions and to improve model interpretability.

### 5. Fairness Assessment

Assess model performance across age, gender, and race groups using appropriate subgroup metrics. Investigate disparities and evaluate mitigation methods where feasible.

### 6. Risk Visualization

The planned Streamlit dashboard will display model-generated risk scores, risk categories, and prediction explanations to demonstrate potential follow-up prioritization.

##  Initial Model Results

The initial implementation produced the following results:

| Metric    | Logistic Regression | Random Forest | XGBoost |
| --------- | ------------------: | ------------: | ------: |
| Accuracy  |              64.16% |        88.80% |  66.83% |
| Precision |              16.43% |        47.78% |  17.52% |
| Recall    |              54.12% |         3.79% |  53.24% |
| F1-score  |              25.21% |         7.02% |  26.37% |
| ROC-AUC   |               0.641 |         0.645 |   0.657 |

**Initial observation:** XGBoost achieved the highest ROC-AUC and F1-score among the three tested models. However, its precision and overall discrimination remain limited. Random Forest's high accuracy alongside very low recall demonstrates why accuracy alone is insufficient for this imbalanced dataset.

*These are initial results and will be reassessed using a documented, leakage-aware evaluation procedure. They are not claims of clinically validated performance.*

## Technology Stack

* **Programming language:** Python
* **Data processing:** Pandas, NumPy
* **Machine learning:** Scikit-learn, XGBoost
* **Explainable AI:** SHAP
* **Visualization:** Matplotlib, Seaborn
* **Dashboard:** Streamlit
* **Development environment:** Google Colab / Jupyter Notebook

##  Development Roadmap

* [x] Initial data preprocessing
* [x] Baseline comparison of three machine-learning models
* [x] Initial SHAP explainability analysis
* [x] Initial fairness assessment
* [ ] Complete dataset quality and leakage audit
* [ ] Improve class-imbalance handling
* [ ] Tune model hyperparameters and classification thresholds
* [ ] Evaluate PR-AUC and probability calibration
* [ ] Reassess fairness across demographic groups
* [ ] Compare fairness-mitigation methods
* [ ] Improve and test the Streamlit dashboard
* [ ] Validate performance using independent data, if available

##  Expected Impact

ReCare AI explores how predictive analytics and explainable machine learning can support readmission risk assessment, transparent decision support, and more informed discharge follow-up planning.

The project's practical value will depend on further performance evaluation, fairness assessment, calibration, and independent validation.

##  Limitations and Disclaimer

* The dataset contains historical encounters from 1999–2008 and may not represent current healthcare practices.
* Class imbalance and limited predictive performance remain challenges.
* Model explanations do not establish causal relationships.
* Risk categories and thresholds require further validation.
* The system is a research and educational prototype, not a clinically validated medical tool. It must not replace professional clinical judgement.

##  Future Scope

* Evaluate the model on more recent and independent datasets.
* Improve predictive performance and probability calibration.
* Investigate fairness-aware learning approaches.
* Enhance dashboard usability and explainability.
* Explore clinical workflow integration subject to appropriate validation, privacy safeguards, and clinical oversight.

##  Team

**Team Name:** Proteome Prism

**Event:** KLE TECH HACKFEST 2026

## 📄 Data Source

UCI Machine Learning Repository: [Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

---

*ReCare AI — Predict earlier. Explain clearly. Follow up smarter.*

