import pandas as pd
import shap


def calculate_shap_importance(model, X, feature_names, sample_size=10000):
    """
    Calculate mean absolute SHAP importance for a sample of observations.
    """

    if len(X) > sample_size:
        X_sample = X[:sample_size]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": abs(shap_values).mean(axis=0),
    })

    return importance.sort_values(
        "mean_abs_shap",
        ascending=False
    ).reset_index(drop=True)


def aggregate_shap_importance(
    importance,
    feature_mapping
):
    """
    Aggregate encoded SHAP features back to their original
    feature groups.
    """

    aggregated = importance.copy()

    aggregated["original_feature"] = (
        aggregated["feature"]
        .map(feature_mapping)
        .fillna(aggregated["feature"])
    )

    return (
        aggregated
        .groupby("original_feature")["mean_abs_shap"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )