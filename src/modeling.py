import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
)
from xgboost import XGBClassifier


def train_xgboost(X_train, y_train):
    """
    Train the XGBoost credit-risk classifier.
    """

    scale_pos_weight = (
        (y_train == 0).sum() / (y_train == 1).sum()
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(y_true, probabilities, threshold=0.5):
    """
    Calculate the main classification metrics.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),
        "f1": f1_score(y_true, predictions),
        "brier_score": brier_score_loss(
            y_true, probabilities
        ),
    }