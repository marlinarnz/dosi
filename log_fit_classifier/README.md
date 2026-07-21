# log_fit_classifier

Predicts whether a logistic fit is "good" — i.e. `use_logfit_FIN == 1` in
`data/summary_table_v24_21Mar.xlsx` (sheet `summary_table_v24_CW`). This is a
human-adjudicated label (reviewer scoring -> agreement -> final), not a
rule-derived one, so it's a reasonable ML target.

Kept standalone from `streamlit/` on purpose — see "Using this from the
Streamlit app" below for how to wire it up later.

## Files

- `features.py` — the feature list (`FEATURE_COLUMNS`) and target column
  (`TARGET_COLUMN`), shared by training and prediction so they can't drift
  apart. Deliberately excludes every column derived from the human review
  itself (`use_logfit_CW`/`GN`, `"use_logfit_AGREE?"`, `"FINAL AGREED"`,
  `select_*`, `comments_*`) and the `C1_R2`...`C7_lin_r2` / `use_log` /
  `use_lin` columns, which are rule-based thresholds computed from the same
  raw stats (see `dosi_graphs_2.py` ~L797-860) and would leak the label.
- `train.py` — loads the xlsx, runs 5-fold stratified cross-validation,
  prints a metrics report, refits on the full dataset, and saves the model.
- `predict.py` — loads the saved model once and exposes `predict(features)`.
- `models/log_fit_classifier.joblib` — the trained model, committed to the
  repo (~5MB) so `predict()` works out of the box without retraining.

## Model

`RandomForestClassifier(n_estimators=150, class_weight="balanced")` in a
pipeline with median imputation for the ~0-4% missing values per feature.
Chosen over logistic regression because several of the strongest signals
(e.g. `log_r2`, `autocorr_l1`) behave like thresholded criteria in this
dataset's own review rules (e.g. "good" roughly means `r2_log > 0.7`), which
a random forest captures without manual feature engineering or scaling.

## Current performance (5-fold stratified CV, 1768 rows, 70/30 class split)

| metric    | score |
|-----------|-------|
| accuracy  | 0.837 (majority-class baseline: 0.700) |
| roc_auc   | 0.909 |
| precision | 0.865 |
| recall    | 0.910 |
| f1        | 0.887 |

Re-run `python log_fit_classifier/train.py` to reproduce/update this report.

## Retraining

```bash
python log_fit_classifier/train.py
```

Prints the CV report to stdout and overwrites
`models/log_fit_classifier.joblib`. Only needed if the source data changes
or the model/features are adjusted — commit the updated `.joblib` alongside
code changes if it stays under a few MB.

## Using `predict()`

```python
from log_fit_classifier.predict import predict

predict({"log_r2": 0.92, "n_data_points": 14, "autocorr_l1": 0.3})
# -> {"prediction": 1, "probability_good_fit": 0.87}
```

Any subset of `features.FEATURE_COLUMNS` may be passed; missing keys are
imputed the same way as at training time. `prediction` is thresholded at
probability >= 0.5; callers wanting a different cutoff (e.g. a Streamlit
slider) can threshold `probability_good_fit` themselves.

## Using this from the Streamlit app (future step, not wired up yet)

The app currently reads pre-processed CSVs, not this xlsx, and has no
dependency on this directory. To call `predict()` from a page under
`streamlit/pages/`, add the repo root to `sys.path` before importing, e.g.:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from log_fit_classifier.predict import predict
```

This mirrors the `Path(__file__).parent...`-based relative-path convention
already used elsewhere in `streamlit/pages/*.py`.
