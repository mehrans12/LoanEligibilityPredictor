"""
app.py
------
Loan Eligibility Predictor - main entry point.

Run with:  streamlit run app.py

This file only handles the UI/orchestration layer. All AI logic lives in
data_utils.py, model_utils.py, and explain_utils.py, kept separate as
required by the project's modular design guideline.
"""

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd
import streamlit as st

from data_utils import load_data, preprocess_data, encode_single_input, validate_input
from model_utils import train_models, evaluate_models, predict_single
from explain_utils import get_feature_importance, generate_explanation, plain_language_summary

DATA_PATH = "data/loan_data.csv"

st.set_page_config(page_title="Loan Eligibility Predictor", layout="wide")


# ----------------------------------------------------------------------
# Cache the expensive steps so retraining doesn't happen on every click
# ----------------------------------------------------------------------
@st.cache_data
def get_processed_data():
    df = load_data(DATA_PATH)
    X, y, feature_names = preprocess_data(df)
    return df, X, y, feature_names


@st.cache_resource
def get_trained_models(X, y):
    models, X_train, X_test, y_train, y_test = train_models(X, y)
    results = evaluate_models(models, X_test, y_test)
    return models, X_test, y_test, results


def create_visuals(results: dict, importance_df: pd.DataFrame, tree_model, feature_names):
    """Builds all chart objects used across the app. Kept separate from
    render_ui() so visuals can be reused or tested independently."""
    figs = {}

    # Feature importance bar chart
    fig1, ax1 = plt.subplots(figsize=(5, 3.2))
    ax1.barh(importance_df["feature"], importance_df["importance"], color="#4C72B0")
    ax1.set_xlabel("Importance")
    ax1.set_title("Top Features Driving Decisions (Decision Tree)")
    ax1.invert_yaxis()
    fig1.tight_layout()
    figs["importance"] = fig1

    # Decision tree diagram
    fig2, ax2 = plt.subplots(figsize=(16, 8))
    plot_tree(
        tree_model, feature_names=feature_names, class_names=["Rejected", "Approved"],
        filled=True, rounded=True, fontsize=8, ax=ax2
    )
    ax2.set_title("Decision Tree Structure (max depth = 4)")
    figs["tree"] = fig2

    # Confusion matrices for both models
    fig3, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, (name, res) in zip(axes, results.items()):
        cm = res["confusion_matrix"]
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Rejected", "Approved"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Rejected", "Approved"], fontsize=8)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10)
    fig3.tight_layout()
    figs["confusion"] = fig3

    return figs


def render_ui():
    st.title("🏦 Loan Eligibility Predictor")
    st.caption(
        "AI Lab Project — compares a Decision Tree and Logistic Regression model "
        "to predict loan approval, and explains every decision."
    )

    df, X, y, feature_names = get_processed_data()
    models, X_test, y_test, results = get_trained_models(X, y)
    tree_model = models["Decision Tree"]

    tab_predict, tab_compare, tab_data = st.tabs(
        ["🔍 Predict & Explain", "📊 Model Comparison", "📁 Dataset"]
    )

    # ------------------------------------------------------------------
    # TAB 1: Problem Setup Module + prediction + explainability
    # ------------------------------------------------------------------
    with tab_predict:
        st.subheader("Applicant Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])

        with col2:
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            credit_history = st.selectbox("Credit History", [1.0, 0.0],
                                           format_func=lambda x: "Good (1)" if x == 1.0 else "Poor (0)")
            loan_term = st.select_slider("Loan Term (months)", [60, 84, 120, 180, 360], value=360)

        with col3:
            applicant_income = st.slider("Applicant Monthly Income ($)", 0, 20000, 5000, step=100)
            coapplicant_income = st.slider("Co-applicant Monthly Income ($)", 0, 10000, 0, step=100)
            loan_amount = st.slider("Loan Amount (in $1000s)", 0, 700, 120, step=5)

        input_dict = {
            "Gender": gender, "Married": married, "Dependents": dependents,
            "Education": education, "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history, "Property_Area": property_area,
        }

        run = st.button("Run Prediction", type="primary")

        if run:
            errors = validate_input(input_dict)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                with st.spinner("Running model..."):
                    encoded_row = encode_single_input(input_dict, feature_names)
                    label, confidence = predict_single(tree_model, encoded_row)
                    reasons = generate_explanation(tree_model, encoded_row, feature_names)
                    summary = plain_language_summary(label, confidence, reasons)

                st.success("Prediction complete.")

                result_col, explain_col = st.columns([1, 1.4])
                with result_col:
                    st.metric("Decision", label, f"{confidence*100:.1f}% confidence")
                    if label == "Approved":
                        st.markdown("✅ **This application would likely be approved.**")
                    else:
                        st.markdown("❌ **This application would likely be rejected.**")

                with explain_col:
                    st.markdown("**Why this decision?**")
                    st.write(summary)
                    with st.expander("See full decision path"):
                        for r in reasons:
                            st.write("• " + r)

                st.markdown("---")
                importance_df = get_feature_importance(tree_model, feature_names)
                figs = create_visuals(results, importance_df, tree_model, feature_names)
                st.pyplot(figs["importance"])

    # ------------------------------------------------------------------
    # TAB 2: Evaluation Module (compare Decision Tree vs Logistic Regression)
    # ------------------------------------------------------------------
    with tab_compare:
        st.subheader("Model Comparison")
        metrics_df = pd.DataFrame({
            name: {
                "Accuracy": f"{res['accuracy']:.3f}",
                "Precision": f"{res['precision']:.3f}",
                "Recall": f"{res['recall']:.3f}",
                "F1 Score": f"{res['f1']:.3f}",
            }
            for name, res in results.items()
        }).T
        st.table(metrics_df)

        importance_df = get_feature_importance(tree_model, feature_names)
        figs = create_visuals(results, importance_df, tree_model, feature_names)

        st.markdown("**Confusion Matrices**")
        st.pyplot(figs["confusion"])

        st.markdown("**Decision Tree Structure**")
        st.pyplot(figs["tree"])

    # ------------------------------------------------------------------
    # TAB 3: Raw dataset viewer
    # ------------------------------------------------------------------
    with tab_data:
        st.subheader("Sample Dataset")
        st.write(f"{len(df)} rows loaded from `{DATA_PATH}`")
        st.dataframe(df.head(50))
        st.markdown("**Loan Status Distribution**")
        st.bar_chart(df["Loan_Status"].value_counts())


if __name__ == "__main__":
    render_ui()
