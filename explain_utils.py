"""
explain_utils.py
-----------------
Turns the Decision Tree's internal logic into a human-readable explanation
for a single prediction, and extracts feature importances. This is what
satisfies the "Explainability Module" requirement.
"""

import numpy as np
import pandas as pd


def get_feature_importance(model, feature_names, top_n=6) -> pd.DataFrame:
    """Returns the top N most influential features for a tree-based model."""
    importances = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n)
    return df


def generate_explanation(model, encoded_row: pd.DataFrame, feature_names: list, currency_config: dict = None) -> list:
    """
    Walks the Decision Tree's actual decision path for this specific
    applicant and converts each split into a plain-language sentence.

    Returns:
        list[str]: ordered list of human-readable reasons, e.g.
                    "Credit_History <= 0.50 -> True (this applicant has poor credit history)"
    """
    tree = model.tree_
    feature = tree.feature
    threshold = tree.threshold

    node_indicator = model.decision_path(encoded_row)
    leaf_id = model.apply(encoded_row)
    sample_id = 0

    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
    ]

    reasons = []
    for node_id in node_index:
        if leaf_id[sample_id] == node_id:
            continue  # leaf node has no split rule to explain

        feat_name = feature_names[feature[node_id]]
        value = float(encoded_row.iloc[sample_id, feature[node_id]])
        thresh = float(threshold[node_id])

        # Apply currency scaling/formatting (defaults to PKR)
        config = currency_config if currency_config is not None else {"factor": 278.0, "symbol": "₨"}
        factor = config.get("factor", 1.0)
        symbol = config.get("symbol", "$")

        if feat_name == "ApplicantIncome":
            display_value = value * factor
            display_thresh = thresh * factor
            val_str = f"{symbol} {display_value:,.2f}" if factor != 1.0 else f"{symbol}{display_value:,.2f}"
            thr_str = f"{symbol} {display_thresh:,.2f}" if factor != 1.0 else f"{symbol}{display_thresh:,.2f}"
        elif feat_name == "CoapplicantIncome":
            display_value = value * factor
            display_thresh = thresh * factor
            val_str = f"{symbol} {display_value:,.2f}" if factor != 1.0 else f"{symbol}{display_value:,.2f}"
            thr_str = f"{symbol} {display_thresh:,.2f}" if factor != 1.0 else f"{symbol}{display_thresh:,.2f}"
        elif feat_name == "LoanAmount":
            if factor != 1.0:
                display_value = value * 1000.0 * factor
                display_thresh = thresh * 1000.0 * factor
                val_str = f"{symbol} {display_value:,.2f}"
                thr_str = f"{symbol} {display_thresh:,.2f}"
            else:
                display_value = value
                display_thresh = thresh
                val_str = f"{symbol}{display_value:,.2f}k"
                thr_str = f"{symbol}{display_thresh:,.2f}k"
        else:
            val_str = f"{round(value, 2)}"
            thr_str = f"{round(thresh, 2)}"

        if value <= thresh:
            direction = "at or below"
        else:
            direction = "above"

        reasons.append(
            f"'{feat_name}' is {direction} {thr_str} "
            f"(applicant's value: {val_str})"
        )

    return reasons


def plain_language_summary(label: str, confidence: float, top_reasons: list) -> str:
    """Builds a short natural-language summary for the result panel."""
    reason_text = "; ".join(top_reasons[:3]) if top_reasons else "the overall profile"
    return (
        f"The model predicts this application would be **{label}** "
        f"with {confidence*100:.1f}% confidence, mainly because {reason_text}."
    )
