import pandas as pd


RISK_BINS = [0, 0.10, 0.20, 0.30, 0.40, 1.0]

RISK_LABELS = [
    "Very Low",
    "Low",
    "Medium",
    "High",
    "Very High",
]


def assign_risk_bands(probabilities):
    """
    Convert calibrated probabilities into risk bands.
    """

    return pd.cut(
        probabilities,
        bins=RISK_BINS,
        labels=RISK_LABELS,
        include_lowest=True,
    )


def summarize_risk_bands(y_true, probabilities):
    """
    Compare predicted PD with observed default rates
    across risk bands.
    """

    summary = pd.DataFrame({
        "actual_default": y_true,
        "predicted_pd": probabilities,
    })

    summary["risk_band"] = assign_risk_bands(
        summary["predicted_pd"]
    )

    band_summary = (
        summary
        .groupby("risk_band", observed=True)
        .agg(
            borrowers=("actual_default", "size"),
            predicted_pd=("predicted_pd", "mean"),
            observed_default_rate=("actual_default", "mean"),
        )
    )

    return band_summary