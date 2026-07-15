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
