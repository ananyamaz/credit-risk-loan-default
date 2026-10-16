from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


def fit_isotonic_calibrator(probabilities, y_true):
    """
    Fit an isotonic regression calibrator using validation predictions.
    """

    calibrator = IsotonicRegression(
        y_min=0,
        y_max=1,
        out_of_bounds="clip",
    )

    calibrator.fit(probabilities, y_true)

    return calibrator


def calibrate_probabilities(calibrator, probabilities):
    """
    Convert raw model probabilities into calibrated probabilities.
    """

    return calibrator.predict(probabilities)


def calibration_brier_score(y_true, probabilities):
    """
    Calculate the Brier score for predicted probabilities.
    """

    return brier_score_loss(y_true, probabilities)