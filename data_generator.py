import numpy as np
import pandas as pd

RANDOM_STATE = 42

MEDICAL_CONDITIONS = [
    "Emergency Trauma",
    "Heart Disease",
    "Respiratory Infection",
    "Diabetes Complication",
    "Appendicitis",
    "Stroke",
    "Routine Evaluation"
]

ADMISSION_TYPES = ["Emergency", "Urgent", "Elective"]

DEPARTMENTS = [
    "Emergency Care",
    "Cardiology",
    "Pulmonology",
    "Endocrinology",
    "General Surgery",
    "Neurology",
    "General Medicine"
]

HOSPITALS = [
    "City General Hospital",
    "St. Jude Medical Center",
    "Metro Health Institute",
    "Grace Memorial Hospital",
    "Valley Care Center"
]

DOCTORS = [
    "Dr. Smith", "Dr. Patel", "Dr. Johnson", "Dr. Garcia",
    "Dr. Lee", "Dr. Kim", "Dr. Davis", "Dr. Wilson"
]

INSURANCE_PROVIDERS = ["Medicare", "Blue Cross", "Aetna", "UnitedHealth", "Cigna", "Uninsured"]

MEDICATIONS = [
    "Antibiotics", "Beta Blockers", "Insulin",
    "Anticoagulants", "Analgesics", "Bronchodilators", "None"
]

TEST_RESULTS = ["Normal", "Abnormal", "Inconclusive", "Critical"]

BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]


def generate_raw_dataset(seed=RANDOM_STATE):
    """
    Generates a reproducible raw hospital operations dataset with intentional
    missing values and duplicates.
    Target clean size: 10,000 records.
    Raw size before cleaning: 10,250 records.
    """
    rng = np.random.default_rng(seed)
    n_base = 10000

    # 1. Primary Demographics
    age = np.clip(rng.normal(52, 18, n_base).astype(int), 1, 95)
    gender = rng.choice(["Male", "Female", "Other"], n_base, p=[0.49, 0.49, 0.02])
    blood_type = rng.choice(BLOOD_TYPES, n_base)
    condition = rng.choice(MEDICAL_CONDITIONS, n_base, p=[0.18, 0.20, 0.16, 0.14, 0.12, 0.10, 0.10])

    # 2. Admission & Operational Info
    admission_type = rng.choice(ADMISSION_TYPES, n_base, p=[0.42, 0.33, 0.25])
    department = rng.choice(DEPARTMENTS, n_base)
    hospital = rng.choice(HOSPITALS, n_base)
    doctor = rng.choice(DOCTORS, n_base)
    insurance = rng.choice(INSURANCE_PROVIDERS, n_base, p=[0.28, 0.24, 0.18, 0.15, 0.10, 0.05])

    # 3. Treatment & Clinical Info
    medication = rng.choice(MEDICATIONS, n_base)
    test_result = rng.choice(TEST_RESULTS, n_base, p=[0.45, 0.30, 0.15, 0.10])
    previous_visits = rng.poisson(2.4, n_base)
    previous_visits = np.clip(previous_visits, 0, 15)

    # 4. Target Modeling: length_of_stay (Realistic non-deterministic relationships)
    # Base stay duration
    base_stay = 3.0
    
    # Condition impact
    condition_stay = np.select(
        [
            condition == "Stroke",
            condition == "Emergency Trauma",
            condition == "Heart Disease",
            condition == "Appendicitis",
            condition == "Respiratory Infection",
            condition == "Diabetes Complication",
            condition == "Routine Evaluation"
        ],
        [5.2, 4.5, 3.8, 2.5, 2.2, 2.0, 0.8],
        default=2.0
    )

    # Admission type impact
    adm_stay = np.select(
        [admission_type == "Emergency", admission_type == "Urgent"],
        [2.8, 1.4],
        default=0.0
    )

    # Test result impact
    test_stay = np.select(
        [test_result == "Critical", test_result == "Abnormal", test_result == "Inconclusive"],
        [3.5, 1.8, 0.9],
        default=0.0
    )

    # Age impact (older patients stay somewhat longer)
    age_stay = (age / 25.0) * 0.9

    # Previous visits impact
    prev_stay = (previous_visits / 4.0) * 0.4

    # Combined with controlled noise
    noise_stay = rng.normal(0, 1.5, n_base)
    raw_los = base_stay + condition_stay + adm_stay + test_stay + age_stay + prev_stay + noise_stay
    length_of_stay = np.clip(np.round(raw_los), 1, 30).astype(int)

    # 5. Dates
    start_date = np.datetime64("2024-01-01")
    random_days = rng.integers(0, 365, n_base)
    admission_dates = pd.to_datetime(start_date + random_days)
    discharge_dates = admission_dates + pd.to_timedelta(length_of_stay, unit="D")

    # 6. Financial Info: billing_amount
    # Realistic billing formula: stay length + condition complexity + admission fee + noise
    daily_rate = 650.0
    base_fee = np.select(
        [admission_type == "Emergency", admission_type == "Urgent"],
        [1800.0, 1000.0],
        default=400.0
    )
    condition_fee = np.select(
        [
            condition == "Stroke",
            condition == "Emergency Trauma",
            condition == "Heart Disease",
            condition == "Appendicitis",
            condition == "Diabetes Complication",
            condition == "Respiratory Infection"
        ],
        [4500.0, 4000.0, 3800.0, 2200.0, 1800.0, 1500.0],
        default=800.0
    )
    billing_noise = rng.normal(0, 450.0, n_base)
    raw_billing = (length_of_stay * daily_rate) + base_fee + condition_fee + (age * 12.0) + billing_noise
    billing_amount = np.clip(raw_billing, 350.0, 45000.0).round(2)

    # Build initial dataframe
    df = pd.DataFrame({
        "patient_id": [f"PT_{i:05d}" for i in range(1, n_base + 1)],
        "age": age.astype(float),
        "gender": gender,
        "blood_type": blood_type,
        "medical_condition": condition,
        "admission_type": admission_type,
        "admission_date": admission_dates,
        "discharge_date": discharge_dates,
        "department": department,
        "hospital": hospital,
        "doctor": doctor,
        "insurance_provider": insurance,
        "medication": medication,
        "test_result": test_result,
        "previous_visits": previous_visits,
        "billing_amount": billing_amount,
        "length_of_stay": length_of_stay
    })

    # 7. Introduce Intentional Missing Values (~3% to 4%)
    df.loc[rng.choice(n_base, int(n_base * 0.035), replace=False), "age"] = np.nan
    df.loc[rng.choice(n_base, int(n_base * 0.030), replace=False), "gender"] = np.nan
    df.loc[rng.choice(n_base, int(n_base * 0.032), replace=False), "medical_condition"] = np.nan
    df.loc[rng.choice(n_base, int(n_base * 0.040), replace=False), "insurance_provider"] = np.nan
    df.loc[rng.choice(n_base, int(n_base * 0.030), replace=False), "billing_amount"] = np.nan
    df.loc[rng.choice(n_base, int(n_base * 0.028), replace=False), "test_result"] = np.nan

    # 8. Introduce Intentional Duplicate Rows (250 duplicate rows)
    duplicates = df.sample(n=250, random_state=seed, replace=False)
    raw_df = pd.concat([df, duplicates], ignore_index=True)

    return raw_df
