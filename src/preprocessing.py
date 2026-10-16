import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "revenue",
    "dti_n",
    "loan_amnt",
    "fico_n",
    "issue_year",
    "loan_to_income",
    "log_revenue",
    "log_loan_amnt",
    "issue_month",
    "employment_years",
]

CATEGORICAL_FEATURES = [
    "purpose",
    "home_ownership_n",
    "addr_state",
]


def create_preprocessor():
    """
    Create the preprocessing pipeline used for model training.

    Numeric features are standardized and categorical features
    are one-hot encoded.
    """

    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def clean_features(df):
    """
    Prepare the modeling features from the engineered dataframe.
    """

    df = df.copy()

    # Treat the dataset's anomalous DTI value as missing.
    df["dti_n"] = df["dti_n"].replace(999, np.nan)

    return df[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ]