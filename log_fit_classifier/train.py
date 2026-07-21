"""Train the log-fit quality classifier.

Usage:
    python log_fit_classifier/train.py

Loads data/summary_table_v24_21Mar.xlsx, cross-validates a RandomForest
classifier against the human-adjudicated `use_logfit_FIN` label, prints an
evaluation report, then refits on the full dataset and persists the model.
"""

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from features import FEATURE_COLUMNS, TARGET_COLUMN, build_feature_frame

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "summary_table_v24_21Mar.xlsx"
SHEET_NAME = "summary_table_v24_CW"
MODEL_PATH = (
    Path(__file__).resolve().parent / "models" / "log_fit_classifier.joblib"
)

SCORING = ["accuracy", "roc_auc", "precision", "recall", "f1"]


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    X = build_feature_frame(df)
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def print_evaluation_report(X: pd.DataFrame, y: pd.Series) -> None:
    baseline_accuracy = y.value_counts(normalize=True).max()
    print(f"Majority-class baseline accuracy: {baseline_accuracy:.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(build_pipeline(), X, y, cv=cv, scoring=SCORING)

    print(f"\n5-fold stratified CV over {len(y)} rows "
          f"({y.mean():.0%} positive / {1 - y.mean():.0%} negative):")
    for metric in SCORING:
        scores = results[f"test_{metric}"]
        print(f"  {metric:10s}: {scores.mean():.3f} (+/- {scores.std():.3f})")


def train_and_save() -> None:
    X, y = load_training_data()
    print_evaluation_report(X, y)

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": FEATURE_COLUMNS,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        MODEL_PATH,
    )
    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"\nSaved model to {MODEL_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    train_and_save()
