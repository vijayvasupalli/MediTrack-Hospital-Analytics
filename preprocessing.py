import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Feature definition for machine learning (NO TARGET LEAKAGE!)
NUMERIC_FEATURES = ["age", "previous_visits", "billing_amount"]
CATEGORICAL_FEATURES = [
    "gender",
    "medical_condition",
    "admission_type",
    "department",
    "hospital",
    "insurance_provider",
    "test_result"
]
ALL_INPUT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_FEATURE = "length_of_stay"


def clean_dataset(raw_df):
    """
    Cleans raw dataset, imputes missing values, removes duplicates,
    performs feature engineering, and ensures exactly 10,000 valid records.
    """
    df = raw_df.copy()

    # 1. Missing Values Analysis Before Cleaning
    missing_counts = df.isna().sum()
    missing_pcts = (missing_counts / len(df) * 100).round(2)
    
    missing_summary = pd.DataFrame({
        "Column": missing_counts.index,
        "Missing Values": missing_counts.values,
        "Missing %": missing_pcts.values
    })
    
    # Treatment column documentation
    treatments = []
    for col in missing_summary["Column"]:
        if col in ["age", "billing_amount"]:
            treatments.append("Median Imputation")
        elif col in ["gender", "medical_condition", "insurance_provider", "test_result"]:
            treatments.append("Mode (Most Frequent) Imputation")
        else:
            treatments.append("None required")
    missing_summary["Treatment Applied"] = treatments

    # 2. Duplicate Detection & Removal
    duplicates_detected = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # Downsample or trim to exactly 10,000 records if needed
    if len(df) > 10000:
        df = df.iloc[:10000].copy()

    # 3. Handle Missing Values Imputation
    # Numeric columns
    for col in ["age", "billing_amount"]:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns
    for col in ["gender", "medical_condition", "insurance_provider", "test_result"]:
        if df[col].isna().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    # 4. Date Type Conversions
    df["admission_date"] = pd.to_datetime(df["admission_date"])
    df["discharge_date"] = pd.to_datetime(df["discharge_date"])

    # 5. Feature Engineering
    # Target length of stay derived from dates (ensuring logical consistency)
    df["length_of_stay"] = (df["discharge_date"] - df["admission_date"]).dt.days
    df["length_of_stay"] = np.clip(df["length_of_stay"], 1, 30).astype(int)

    # Temporal features from admission_date
    df["admission_month"] = df["admission_date"].dt.strftime("%b")
    df["admission_day"] = df["admission_date"].dt.day
    df["admission_weekday"] = df["admission_date"].dt.strftime("%A")

    # Age Group Bins
    age_bins = [0, 18, 35, 50, 65, 120]
    age_labels = ["0-18 (Pediatric)", "19-35 (Young Adult)", "36-50 (Adult)", "51-65 (Senior)", "65+ (Geriatric)"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=True)

    final_count = len(df)

    return df, missing_summary, duplicates_detected, final_count


def make_preprocessor():
    """
    Creates a reproducible Scikit-Learn ColumnTransformer pipeline
    for numeric and categorical feature preprocessing.
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ])

    return preprocessor
