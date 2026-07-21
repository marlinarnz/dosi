"""Feature/target definitions for the log-fit quality classifier.

Single source of truth shared by train.py and predict.py so the two can't
drift apart on what a "feature" is.
"""

import pandas as pd

# Raw, non-leaky diagnostics of the logistic (and competing exponential /
# linear) curve fits. Deliberately excludes every human-review column
# (use_logfit_CW/GN, "use_logfit_AGREE?", "FINAL AGREED", select_*,
# comments_*, old scoring columns) and the C1_R2...C7_lin_r2 / use_log /
# use_lin columns, which are rule-based thresholds computed from these same
# raw stats (see dosi_graphs_2.py ~L797-860) and would leak the label rather
# than teach the model anything new.
FEATURE_COLUMNS = [
    "log_r2",
    "log_r2adj",
    "log_rmse",
    "log_mae",
    "slope_log",
    "log_t0",
    "log_Dt",
    "log_K",
    "n_data_points",
    "n_non_zero_data_points",
    "length_trimmed_series_years",
    "n_data_points_beyond_max",
    "n_data_points_beyond_min",
    "max_over_K",
    "min_over_K",
    "range_over_k",
    "autocorr_l1",
    "suspected_reversal_up2down",
    "suspected_reversal_down2up",
    "at_least_one_big_jump",
    "all_values_less_than_or_equal_to_1",
    "all_values_less_than_or_equal_to_100",
    "exp_r2",
    "lin_r2",
]

TARGET_COLUMN = "use_logfit_FIN"


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select and coerce the model's feature columns from a raw df."""
    return df.reindex(columns=FEATURE_COLUMNS).apply(
        pd.to_numeric, errors="coerce"
    )
