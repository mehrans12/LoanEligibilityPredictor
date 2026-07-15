# Loan Eligibility Predictor — Project Report

## 1. Problem

Financial institutions need to decide, quickly and consistently, whether a
loan application should be approved. This project builds a tool that takes
an applicant's details (income, credit history, employment, dependents,
etc.) and predicts approval — while also explaining *why*, so the decision
is transparent rather than a black box.

- **Input:** applicant demographic and financial details (12 fields)
- **Output:** Approved / Rejected, with a confidence score and explanation
- **Constraints:** the explanation must be understandable to a
  non-technical user, and the model must be evaluable against a baseline

## 2. Method

### Data
A 600-row synthetic dataset (`data/loan_data.csv`) was generated to mirror
the structure of the well-known Kaggle "Loan Prediction" dataset. Approval
outcomes were generated using explainable rules — credit history, income-
to-loan ratio, education, and employment status all influence the
probability of approval, with random noise added so the data isn't
perfectly separable (as with real-world data).

### Preprocessing
Categorical fields (Gender, Married, Dependents, Education,
Self_Employed, Property_Area) were one-hot encoded. Numeric fields
(income, loan amount, term, credit history) were used directly for the
Decision Tree and standardized (zero mean, unit variance) before being
fed to Logistic Regression.

### Models
- **Decision Tree Classifier** (max depth = 4) — the primary model. A
  shallow depth was deliberately chosen to keep the tree interpretable:
  every prediction can be traced through at most 4 yes/no splits.
- **Logistic Regression** — trained as a baseline for comparison, since
  it's a simpler, well-understood linear model.

### Explainability
For every prediction, the app walks the Decision Tree's actual decision
path for that specific applicant and converts each split into a plain
sentence (e.g. *"Credit_History is above 0.5"*). It also shows the top
features by importance across the whole tree, so the user can see both
the specific reasoning for their case and the general patterns the model
learned.

## 3. AI Used

This project uses classical, fully local **Machine Learning** (Option 3
from the lab guide) — no external APIs were called, so there are no
associated cost, privacy, or rate-limit concerns to report.

## 4. Results

| Metric | Decision Tree | Logistic Regression |
|---|---|---|
| Accuracy | 0.753 | 0.820 |
| Precision | 0.797 | 0.840 |
| Recall | 0.914 | 0.948 |
| F1 Score | 0.851 | 0.891 |

*(Exact numbers vary slightly with random seed/re-generated data — run
`generate_data.py` and the app to reproduce your own figures.)*

Logistic Regression scored slightly higher on all four metrics in this
run. This is a reasonable outcome: the synthetic approval rule is close
to linear (credit history and income-to-loan ratio combine additively),
which favors a linear model. The Decision Tree, however, was kept as the
primary model for predictions because its interpretability directly
satisfies the explainability requirement — a trade-off worth stating
explicitly in the viva.

Both models agree on the most influential factor: **Credit History**,
followed by **income levels** and **number of dependents**.

## 5. Limitations & Future Improvements

- The dataset is synthetic; a real-world dataset (e.g. actual bank
  records, with consent/anonymization) would validate the model further.
- The Decision Tree's shallow depth trades some accuracy for
  interpretability — an ensemble (Random Forest) could improve accuracy
  but would require additional work (e.g. SHAP values) to stay
  explainable.
- No fairness/bias audit was performed across Gender or Property_Area;
  this would be an important next step before any real deployment.
