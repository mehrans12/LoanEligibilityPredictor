"""
model_utils.py
--------------
Core AI logic: trains two models (Decision Tree + Logistic Regression),
runs predictions, and computes evaluation metrics for comparison.
Kept separate from the UI so the "AI" part of the project is self-contained
and easy to explain in a viva.
"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)


def train_models(X, y, test_size=0.25, random_state=42):
    """
    Splits the data and trains both models.

    Returns:
        models (dict): {"Decision Tree": fitted model, "Logistic Regression": fitted model}
        X_train, X_test, y_train, y_test: the split data (needed for evaluation)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # max_depth is capped so the tree stays small enough to visualize and explain
    tree_model = DecisionTreeClassifier(max_depth=4, random_state=random_state)
    tree_model.fit(X_train, y_train)

    # Wrapped in a scaler so income-scale features don't dominate convergence
    logreg_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logreg_model.fit(X_train, y_train)

    models = {
        "Decision Tree": tree_model,
        "Logistic Regression": logreg_model,
    }
    return models, X_train, X_test, y_train, y_test


def evaluate_models(models: dict, X_test, y_test) -> dict:
    """
    Computes accuracy, precision, recall, F1, and confusion matrix
    for every model, so they can be compared side by side (Module E).
    """
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, preds),
        }
    return results


def predict_single(model, encoded_row):
    """
    Runs a prediction for one applicant.

    Returns:
        label (str): "Approved" or "Rejected"
        confidence (float): probability of the predicted class (0-1)
    """
    pred = model.predict(encoded_row)[0]
    proba = model.predict_proba(encoded_row)[0]
    confidence = proba[pred]
    label = "Approved" if pred == 1 else "Rejected"
    return label, float(confidence)
