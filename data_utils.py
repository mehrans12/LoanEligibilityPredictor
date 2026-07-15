"""
data_utils.py
-------------
Handles loading the raw dataset and turning it into model-ready features.
Keeping this separate from app.py and model_utils.py means the data logic
can be tested or reused independently of the UI or the model.
"""

import pandas as pd

CATEGORICAL_COLS = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area"
]
NUMERIC_COLS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History"
]
TARGET_COL = "Loan_Status"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw loan dataset from a CSV file."""
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame):
    """
    Cleans the dataframe and converts it into a numeric feature matrix (X)
    and target vector (y) the models can train on.

    Returns:
        X (pd.DataFrame): one-hot encoded feature matrix
        y (pd.Series): binary target (1 = approved, 0 = rejected)
        feature_names (list): column names of X, in order
    """
    df = df.copy()

    # Drop rows with missing critical values rather than silently guessing
    df = df.dropna(subset=NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL])

    y = (df[TARGET_COL] == "Y").astype(int)

    X = pd.get_dummies(df[NUMERIC_COLS + CATEGORICAL_COLS], columns=CATEGORICAL_COLS)
    feature_names = list(X.columns)

    return X, y, feature_names


def encode_single_input(input_dict: dict, feature_names: list) -> pd.DataFrame:
    """
    Converts a single applicant's raw form input (from the UI) into a
    one-hot encoded row matching the training feature columns exactly,
    filling any missing dummy columns with 0.
    """
    raw_df = pd.DataFrame([input_dict])
    encoded = pd.get_dummies(raw_df, columns=CATEGORICAL_COLS)

    # Ensure every training column exists (fill missing dummies with 0)
    for col in feature_names:
        if col not in encoded.columns:
            encoded[col] = 0

    # Match column order exactly to what the model was trained on
    encoded = encoded[feature_names]
    return encoded


def validate_input(input_dict: dict) -> list:
    """
    Basic validation for user-entered applicant data.
    Returns a list of error messages (empty list means input is valid).
    """
    errors = []
    if input_dict.get("ApplicantIncome", 0) <= 0:
        errors.append("Applicant income must be greater than 0.")
    if input_dict.get("LoanAmount", 0) <= 0:
        errors.append("Loan amount must be greater than 0.")
    if input_dict.get("Loan_Amount_Term", 0) <= 0:
        errors.append("Loan term must be greater than 0.")
    if input_dict.get("CoapplicantIncome", 0) < 0:
        errors.append("Co-applicant income cannot be negative.")
    return errors
