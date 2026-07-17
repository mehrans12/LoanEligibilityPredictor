# Loan Eligibility Predictor

An AI-powered web app that predicts whether a loan application would be
approved, and explains *why*, using a Decision Tree compared against a
Logistic Regression baseline.

## Problem

Given an applicant's income, credit history, employment status, and other
details, predict loan approval and clearly show which factors drove the
decision — not just a yes/no answer.

## AI Approach

- **Decision Tree Classifier** (primary model) — chosen because its
  decisions can be traced step-by-step, making it naturally explainable.
- **Logistic Regression** (baseline for comparison) — a standard,
  well-understood classifier used to benchmark the tree's performance.

## Project Structure

```
LoanEligibilityPredictor/
├── app.py              # Streamlit UI (entry point)
├── data_utils.py        # load_data(), preprocess_data(), input validation
├── model_utils.py       # train_models(), evaluate_models(), predict_single()
├── explain_utils.py     # generate_explanation(), get_feature_importance()
├── generate_data.py     # script used to create the sample dataset
├── requirements.txt
├── data/
│   └── loan_data.csv    # sample dataset (600 synthetic applicants)
└── screenshots/         # add your UI screenshots here before submission
```

## Setup

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Your browser will open automatically at `http://localhost:8501`.

## Using the App

1. Go to the **Predict & Explain** tab.
2. Fill in the applicant's details using the dropdowns and sliders.
3. Click **Run Prediction**.
4. View the decision, confidence score, plain-language explanation, and
   the exact decision path the model followed.
5. Switch to the **Model Comparison** tab to see accuracy/precision/recall
   for both models side by side, plus the full decision tree diagram.
6. The **Dataset** tab shows the raw data the models were trained on.

## Dataset

`data/loan_data.csv` is a synthetically generated dataset (600 rows) built
to mirror the structure of the well-known Kaggle "Loan Prediction"
dataset (Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area, Loan_Status). It was generated with
`generate_data.py` using explainable rules (credit history, income-to-loan
ratio, education, employment) plus random noise, so the patterns the
models discover are genuine and defensible in a viva.

## Regenerating the Dataset

If you'd like a fresh sample or want to change the size/rules:
```bash
python generate_data.py
```

## Notes for Submission

- Take 3-5 screenshots of the running app (prediction result, explanation
  panel, model comparison tab) and place them in `screenshots/`.
- See `report.md` for the short written report (problem, method, AI used,
  results).
