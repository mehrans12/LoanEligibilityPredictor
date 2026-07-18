"""
generate_data.py
-----------------
Creates a synthetic but realistic loan-eligibility dataset (mirrors the
structure of the well-known Kaggle "Loan Prediction" dataset) and saves it
to data/loan_data.csv.

Approval probability is driven by sensible, explainable rules (credit
history, income vs loan amount, employment) plus random noise, so the
trained models will have genuine, defensible patterns to learn and explain.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 600  # similar size to the classic Kaggle dataset

genders = np.random.choice(["Male", "Female"], N, p=[0.78, 0.22])
married = np.random.choice(["Yes", "No"], N, p=[0.65, 0.35])
dependents = np.random.choice(["0", "1", "2", "3+"], N, p=[0.55, 0.18, 0.17, 0.10])
education = np.random.choice(["Graduate", "Not Graduate"], N, p=[0.78, 0.22])
self_employed = np.random.choice(["Yes", "No"], N, p=[0.14, 0.86])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], N, p=[0.38, 0.38, 0.24])

applicant_income = np.random.gamma(shape=4.5, scale=1200, size=N).round(0)
coapplicant_income = np.where(
    married == "Yes",
    np.random.gamma(shape=2.5, scale=900, size=N).round(0),
    0
)
loan_amount = (
    (applicant_income + coapplicant_income) * np.random.uniform(0.08, 0.20, N)
).round(0) / 1000  # in thousands, like the original dataset
loan_amount = loan_amount.round(0)
loan_amount_term = np.random.choice([360, 180, 120, 84, 60], N, p=[0.72, 0.12, 0.08, 0.05, 0.03])
credit_history = np.random.choice([1.0, 0.0], N, p=[0.84, 0.16])

# --- Rule-driven approval probability (keeps the target explainable) ---
total_income = applicant_income + coapplicant_income
income_to_loan = total_income / (loan_amount * 1000 + 1)

logit = (
    -1.2
    + 2.6 * credit_history
    + 0.55 * (income_to_loan > 12).astype(int)
    + 0.35 * (education == "Graduate").astype(int)
    - 0.30 * (self_employed == "Yes").astype(int)
    + 0.25 * (property_area == "Semiurban").astype(int)
    - 0.20 * (dependents == "3+").astype(int)
    + np.random.normal(0, 0.6, N)
)
prob_approve = 1 / (1 + np.exp(-logit))
loan_status = np.where(np.random.uniform(0, 1, N) < prob_approve, "Y", "N")

df = pd.DataFrame({
    "Gender": genders,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income.astype(int),
    "CoapplicantIncome": coapplicant_income.astype(int),
    "LoanAmount": loan_amount.astype(int),
    "Loan_Amount_Term": loan_amount_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
    "Loan_Status": loan_status,
})

df.to_csv("data/loan_data.csv", index=False)
print(f"Saved {len(df)} rows to data/loan_data.csv")
print(df["Loan_Status"].value_counts(normalize=True))
