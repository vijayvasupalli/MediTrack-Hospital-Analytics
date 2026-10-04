# MediTrack — Hospital Operations Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://meditrack-hospital-analytics-dav.streamlit.app/)

A data analytics, visualization, and predictive modeling mini-project built with **Python, Streamlit, Pandas, NumPy, Plotly, and Scikit-learn**.

**🚀 Live Application URL**: [https://meditrack-hospital-analytics-dav.streamlit.app/](https://meditrack-hospital-analytics-dav.streamlit.app/)

---

## 1. Project Overview

**MediTrack** is an academic hospital operations analytics application designed to optimize hospital patient stay management, evaluate operational resource utilization, and predict patient **length of stay (in days)** using machine learning regression models.

---

## 2. Synthetic Dataset (10,000 Records)

The project utilizes a reproducible synthetic hospital operations dataset containing **10,000 records**.

The dataset features:
- **Patient Information**: `patient_id`, `age`, `gender`, `blood_type`, `medical_condition`
- **Admission Information**: `admission_type`, `admission_date`, `discharge_date`, `department`, `hospital`, `doctor`, `insurance_provider`
- **Treatment Information**: `medication`, `test_result`, `previous_visits`
- **Financial Information**: `billing_amount`
- **Operational Target**: `length_of_stay`

The dataset is generated using a fixed random seed (`RANDOM_STATE = 42`), ensuring 100% reproducibility every time the application is run.

---

## 3. Data Cleaning & Preprocessing

The raw dataset intentionally contains missing values and duplicate records to demonstrate an end-to-end data cleaning workflow:

### Missing Value Handling
- **Numeric Features** (`age`, `billing_amount`): Imputed using **median** values.
- **Categorical Features** (`gender`, `medical_condition`, `insurance_provider`, `test_result`): Imputed using **mode (most frequent)** values.

### Duplicate Detection & Trimming
- Duplicate records are identified and removed.
- Downsampled to ensure the final analytical dataset contains **exactly 10,000 valid records**.

### Feature Engineering
- `length_of_stay`: Computed from discharge and admission dates `(discharge_date - admission_date).dt.days`.
- `admission_month` & `admission_weekday`: Extracted temporal features for seasonal volume analysis.
- `age_group`: Categorized into demographic groups (`0-18`, `19-35`, `36-50`, `51-65`, `65+`).

---

## 4. Exploratory Data Analysis (EDA)

The application provides interactive Plotly visualizations across 10 analytical perspectives:
1. **Patient Age Distribution** (Histogram & box plot)
2. **Medical Condition Distribution** (Bar chart)
3. **Admission Type Mix** (Emergency / Urgent / Elective donut chart)
4. **Department Workload** (Bar chart)
5. **Average Length of Stay by Medical Condition** (Bar chart)
6. **Billing Amount Distribution** (Histogram)
7. **Length of Stay vs. Billing Amount** (Scatter plot colored by urgency)
8. **Monthly Admissions Trend** (Line chart)
9. **Hospital-wise Patient Volume** (Stacked bar chart)
10. **Numeric Feature Correlation Matrix** (Heatmap)

---

## 5. Machine Learning Models

Predicts **hospital length of stay (days)** as a regression task.

### Feature Pipeline (No Target Leakage)
- **Numeric Features** (`age`, `previous_visits`, `billing_amount`): Median Imputer + StandardScaler
- **Categorical Features** (`gender`, `medical_condition`, `admission_type`, `department`, `hospital`, `insurance_provider`, `test_result`): Mode Imputer + OneHotEncoder
- Combined using Scikit-Learn `ColumnTransformer`.

### Evaluated Regression Models
1. **Linear Regression**
2. **Random Forest Regressor**

Both models are evaluated on an **80/20 train/test split** (`random_state = 42`) using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score** (Coefficient of Determination)

---

## 6. Streamlit Web Application Workflow

The application preserves a 5-step intuitive navigation structure:
1. **1 · Actual Dataset**: Raw dataset preview, shape metrics, and column guide.
2. **2 · Data Cleaning**: Missing value summary, duplicate removal audit, and clean preview.
3. **3 · Exploratory Analysis**: KPI cards and 10 interactive Plotly charts with operational insights.
4. **4 · Model Comparison**: Validation scorecard and bar chart comparing MAE, RMSE, and R² scores.
5. **5 · Predict Performance**: Interactive patient profile input form with real-time length-of-stay predictions and operational discharge recommendations.

---

## 7. Live Demo & Local Execution

### 🚀 Live Web Deployment
The application is deployed live on Streamlit Community Cloud:  
👉 **[https://meditrack-hospital-analytics-dav.streamlit.app/](https://meditrack-hospital-analytics-dav.streamlit.app/)**

### 💻 Run Locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Streamlit Application**:
   ```bash
   python -m streamlit run app.py
   ```
