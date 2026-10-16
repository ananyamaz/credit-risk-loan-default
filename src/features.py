import numpy as np
import pandas as pd


def engineer_features(df):
    """
    Create the features used for credit-risk modeling.
    """

    df = df.copy()

    # Convert issue date into datetime
    df["issue_date"] = pd.to_datetime(
        df["issue_d"],
        format="%b-%y"
    )

    # Time features
    df["issue_year"] = df["issue_date"].dt.year
    df["issue_month"] = df["issue_date"].dt.month

    # Loan-to-income ratio
    df["loan_to_income"] = (
        df["loan_amnt"] / df["revenue"]
    )

    # Log transformations
    df["log_revenue"] = np.log1p(df["revenue"])
    df["log_loan_amnt"] = np.log1p(df["loan_amnt"])

    # Convert employment length to numeric years
    employment_mapping = {
        "< 1 year": 0,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9,
        "10+ years": 10,
        "NI": np.nan,
    }

    df["employment_years"] = df["emp_length"].map(
        employment_mapping
    )

    return df