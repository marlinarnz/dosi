"""Reusable prediction function for the log-fit quality classifier.

    from log_fit_classifier.predict import predict
    predict({"log_r2": 0.92, "n_data_points": 14, ...})
    # -> {"prediction": 1, "probability_good_fit": 0.87}

Any subset of log_fit_classifier.features.FEATURE_COLUMNS may be passed;
missing features are imputed the same way as at training time.
"""

from pathlib import Path

import joblib
import pandas as pd

try:
    from .features import FEATURE_COLUMNS
except ImportError:  # running/imported without package context
    from features import FEATURE_COLUMNS

MODEL_PATH = (
    Path(__file__).resolve().parent / "models" / "log_fit_classifier.joblib"
)

_bundle = None


def _get_pipeline():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run "
                "`python log_fit_classifier/train.py` first."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle["pipeline"]


def predict(features: dict) -> dict:
    """Predict whether a log fit is good.

    features: dict of {column_name: value}, any subset of FEATURE_COLUMNS.
    Returns {"prediction": 0 or 1, "probability_good_fit": float}.
    """
    pipeline = _get_pipeline()
    row = pd.DataFrame([features]).reindex(columns=FEATURE_COLUMNS)
    probability = pipeline.predict_proba(row)[0, 1]
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "probability_good_fit": float(probability),
    }
