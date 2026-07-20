import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

import plotly.graph_objects as go
from scipy.optimize import curve_fit, differential_evolution, minimize
from scipy.stats import linregress

CURRENT_DIR = Path(__file__).parent.parent
PATH = "../data"
YEAR_PADDING_FOR_PLOTTING = 10
GROUP_VARS = ["Innovation Name", "Spatial Scale", "Indicator Number", "Description", "Metric"]

st.set_page_config(layout="wide")
st.title("Single Series Explorer")
st.markdown(
    """
    Pick exactly one time series, optionally exclude individual points or year ranges,
    then fit it with any of the curve-fitting approaches used across this repository.
    """
)

# ──────────────────────────────────────────────────────────────
# 1. Data loading (mirrors the Dashboard page)
# ──────────────────────────────────────────────────────────────

source = st.radio("Choose data source", ["Repo file", "Upload local file"], horizontal=True)

dosi_df = None


@st.cache_data
def load_data(version_data):
    return pd.read_csv(
        CURRENT_DIR / PATH / f"adjusted_datasets_{version_data}.csv",
        converters={"Indicator Number": str},
    )


if source == "Upload local file":
    uploaded_file = st.file_uploader("Upload local DoSI data file", type=["csv"])
    if uploaded_file is not None:
        dosi_df = pd.read_csv(uploaded_file, converters={"Indicator Number": str})
else:
    files = [
        entry
        for entry in os.listdir(CURRENT_DIR / PATH)
        if entry.startswith("adjusted_datasets_v")
        and entry.endswith(".csv")
        and os.path.isfile(CURRENT_DIR / PATH / entry)
    ]
    version_data = max([int(file[-6:-4]) for file in files if len(file.split("_")[-1]) == 7])
    try:
        dosi_df = load_data("v" + str(version_data))
    except FileNotFoundError:
        st.error(f"⚠️ Data version '{version_data}' not found. Using default 'v30'.")
        dosi_df = load_data("v30")

if dosi_df is None:
    st.stop()

# Optional: stored logfit summary, used only for reference comparison
summary_df = None
summary_files = [
    entry
    for entry in os.listdir(CURRENT_DIR / PATH)
    if entry.startswith("summary_table_v")
    and entry.endswith(".csv")
    and os.path.isfile(CURRENT_DIR / PATH / entry)
]
if summary_files:
    version_summary = max(
        [int(file[-6:-4]) for file in summary_files if len(file.split("_")[-1]) == 7]
    )
    try:
        summary_df = pd.read_csv(
            CURRENT_DIR / PATH / f"summary_table_v{version_summary}.csv",
            converters={"Indicator Number": str},
        )
        summary_df["Spatial Scale"] = summary_df["Spatial Scale"].str.rstrip()
        summary_df["Innovation Name"] = summary_df["Innovation Name"].str.rstrip()
    except FileNotFoundError:
        summary_df = None

dosi_df["Value"] = pd.to_numeric(dosi_df["Value"], errors="coerce")
dosi_df = dosi_df.dropna(subset=["Value"])
dosi_df["Spatial Scale"] = dosi_df["Spatial Scale"].str.rstrip()
dosi_df["Innovation Name"] = dosi_df["Innovation Name"].str.rstrip()


# ──────────────────────────────────────────────────────────────
# 2. Fitting functions (the different approaches used in this repo)
# ──────────────────────────────────────────────────────────────

def FPLogValue_with_scaling(x, t0, Dt, K):
    x = np.asarray(x, dtype=float)
    return K / (1 + np.exp(-np.log(81) * (x - t0) / Dt))


def logistic_3p(x, A, M, T):
    return A / (1.0 + np.exp(-M * (x - T)))


def exponential_func(x, a, b, c):
    return a * np.exp(b * (x - c))


def calc_metrics(y_obs, y_pred, n_params):
    y_obs = np.asarray(y_obs, dtype=float)
    if y_pred is None or np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        return {"r2": np.nan, "r2_adj": np.nan, "rmse": np.nan, "mae": np.nan}
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n = len(y_obs)
    denom = n - n_params - 1
    r2_adj = 1 - ((1 - r2) * (n - 1) / denom) if denom > 0 and not np.isnan(r2) else np.nan
    rmse = np.sqrt(np.mean((y_obs - y_pred) ** 2))
    mae = np.mean(np.abs(y_obs - y_pred))
    return {"r2": r2, "r2_adj": r2_adj, "rmse": rmse, "mae": mae}


def fit_logistic_curvefit(x, y, A_guess, M_guess, T_guess, maxfev):
    if len(x) < 3:
        return None
    try:
        popt, _ = curve_fit(logistic_3p, x, y, p0=[A_guess, M_guess, T_guess], maxfev=maxfev)
        A_fit, M_fit, T_fit = popt
        if M_fit == 0:
            return None
        return {"t0": T_fit, "Dt": np.log(81) / M_fit, "K": A_fit}
    except (RuntimeError, ValueError):
        return None


def fit_logistic_logit_linear(x, y, threshold, thresholdup):
    mask = (y > threshold) & (y < 1 - thresholdup) & (y < 1)
    x_f, y_f = x[mask], y[mask]
    if len(x_f) < 2:
        return None
    logits = np.log(y_f / (1 - y_f))
    slope, intercept = np.polyfit(x_f, logits, 1)
    if slope == 0:
        return None
    # K is fixed to 1 by construction: this method assumes y is already normalised
    return {"t0": -intercept / slope, "Dt": np.log(81) / slope, "K": 1.0}


def _logistic_sse(params, x, y):
    t0, Dt, K = params
    if Dt == 0:
        return np.inf
    return np.sum((y - FPLogValue_with_scaling(x, t0, Dt, K)) ** 2)


def fit_logistic_diffevo(x, y, t0_bounds, Dt_bounds, K_bounds, maxiter, seed):
    if len(x) < 3:
        return None
    result = differential_evolution(
        _logistic_sse,
        bounds=[t0_bounds, Dt_bounds, K_bounds],
        args=(x, y),
        maxiter=maxiter,
        seed=seed,
    )
    t0, Dt, K = result.x
    return {"t0": t0, "Dt": Dt, "K": K}


def fit_logistic_multistart(x, y, t0_bounds, Dt_bounds, K_bounds, n_starts, method, seed):
    if len(x) < 3:
        return None
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        init = [rng.uniform(*t0_bounds), rng.uniform(*Dt_bounds), rng.uniform(*K_bounds)]
        try:
            res = minimize(_logistic_sse, init, args=(x, y), method=method)
        except Exception:
            continue
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return None
    t0, Dt, K = best.x
    return {"t0": t0, "Dt": Dt, "K": K}


def fit_exponential(x, y, a0, b0, c0, maxfev):
    if len(x) < 3:
        return None
    try:
        popt, _ = curve_fit(exponential_func, x, y, p0=[a0, b0, c0], maxfev=maxfev)
        return {"a": popt[0], "b": popt[1], "c": popt[2]}
    except (RuntimeError, ValueError):
        return None


def fit_linear(x, y):
    if len(x) < 2:
        return None
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return {"slope": slope, "intercept": intercept, "r_value": r_value, "p_value": p_value, "std_err": std_err}


# ──────────────────────────────────────────────────────────────
# 3. Series selection (cascading dropdowns, like the Dashboard page)
# ──────────────────────────────────────────────────────────────

st.subheader("1. Select a single time series")

sorted_inno_index = dosi_df.sort_values("Innovation Name", key=lambda col: col.str.lower()).index
innovation_names = dosi_df.loc[sorted_inno_index, "Innovation Name"].drop_duplicates().tolist()

col1, col2 = st.columns(2)
with col1:
    sel_innovation = st.selectbox("Innovation", innovation_names, index=0)
df1 = dosi_df[dosi_df["Innovation Name"] == sel_innovation]

with col2:
    spatial_options = sorted(df1["Spatial Scale"].unique())
    sel_spatial = st.selectbox("Spatial scale", spatial_options, index=0)
df2 = df1[df1["Spatial Scale"] == sel_spatial]

col3, col4, col5 = st.columns(3)
with col3:
    indicator_codes = sorted(df2["Indicator Number"].unique())
    label_to_code = {
        f"{code} - {df2.loc[df2['Indicator Number'] == code, 'Indicator Name'].iloc[0]}": code
        for code in indicator_codes
    }
    sel_indicator_label = st.selectbox("Indicator", list(label_to_code.keys()), index=0)
    sel_indicator = label_to_code[sel_indicator_label]
df3 = df2[df2["Indicator Number"] == sel_indicator]

with col4:
    description_options = sorted(df3["Description"].unique())
    sel_description = st.selectbox("Description", description_options, index=0)
df4 = df3[df3["Description"] == sel_description]

with col5:
    metric_options = sorted(df4["Metric"].unique())
    sel_metric = st.selectbox("Metric", metric_options, index=0)

series_df = (
    df4[df4["Metric"] == sel_metric]
    .sort_values("Year")
    .drop_duplicates(subset="Year", keep="last")
    .reset_index(drop=True)
)

if len(series_df) == 0:
    st.warning("No data available for this combination.")
    st.stop()

series_key = " | ".join([sel_innovation, sel_spatial, sel_indicator, sel_description, sel_metric])
st.caption(
    f"**{series_key}** — {len(series_df)} data points "
    f"({int(series_df['Year'].min())}–{int(series_df['Year'].max())})"
)

years_all = series_df["Year"].values.astype(float)
values_all = series_df["Value"].values.astype(float)


# ──────────────────────────────────────────────────────────────
# 4. Exclusions: individual points and year ranges
# ──────────────────────────────────────────────────────────────

st.subheader("2. Exclude points or ranges (optional)")

if "excluded_years" not in st.session_state:
    st.session_state["excluded_years"] = {}
if "excluded_ranges" not in st.session_state:
    st.session_state["excluded_ranges"] = {}

if st.button("Reset exclusions for this series"):
    st.session_state["excluded_years"].pop(series_key, None)
    st.session_state["excluded_ranges"].pop(series_key, None)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Exclude individual points**")
    prev_excluded = st.session_state["excluded_years"].get(series_key, set())
    points_editor_df = series_df[["Year", "Value"]].copy()
    points_editor_df["Exclude"] = points_editor_df["Year"].isin(prev_excluded)
    edited_points = st.data_editor(
        points_editor_df,
        column_config={
            "Year": st.column_config.NumberColumn(disabled=True),
            "Value": st.column_config.NumberColumn(disabled=True, format="%.4g"),
            "Exclude": st.column_config.CheckboxColumn(),
        },
        hide_index=True,
        key=f"points_editor_{series_key}",
    )
    excluded_years = set(edited_points.loc[edited_points["Exclude"], "Year"])
    st.session_state["excluded_years"][series_key] = excluded_years

with col_b:
    st.markdown("**Exclude year ranges**")
    st.caption("Add rows to drop all points within a start–end year range (inclusive).")
    default_ranges = st.session_state["excluded_ranges"].get(
        series_key, pd.DataFrame({"Start Year": pd.Series(dtype="float"), "End Year": pd.Series(dtype="float")})
    )
    edited_ranges = st.data_editor(
        default_ranges,
        column_config={
            "Start Year": st.column_config.NumberColumn(),
            "End Year": st.column_config.NumberColumn(),
        },
        num_rows="dynamic",
        hide_index=True,
        key=f"ranges_editor_{series_key}",
    )
    st.session_state["excluded_ranges"][series_key] = edited_ranges

range_mask = np.zeros(len(years_all), dtype=bool)
for _, row in edited_ranges.dropna().iterrows():
    lo, hi = sorted([float(row["Start Year"]), float(row["End Year"])])
    range_mask |= (years_all >= lo) & (years_all <= hi)

point_mask = np.isin(years_all, list(excluded_years)) if excluded_years else np.zeros(len(years_all), dtype=bool)
excluded_mask = point_mask | range_mask
included_mask = ~excluded_mask

x_fit = years_all[included_mask]
y_fit = values_all[included_mask]

st.caption(f"{included_mask.sum()} of {len(years_all)} points included in the fit.")


# ──────────────────────────────────────────────────────────────
# 5. Fitting method and its tunable parameters
# ──────────────────────────────────────────────────────────────

st.subheader("3. Choose a fitting method")

FIT_METHODS = [
    "Logistic — 3-parameter (curve_fit)",
    "Logistic — logit-linearization (fast, requires 0<y<1)",
    "Logistic — bounded global search (differential evolution)",
    "Logistic — multi-start local search",
    "Exponential",
    "Linear",
]

method = st.selectbox(
    "Fitting method",
    FIT_METHODS,
    index=0,
    help=(
        "'Logistic — 3-parameter (curve_fit)' is the approach used for almost all logistic "
        "fits stored in this repo's summary tables (dosi_graphs_2.py, coevolution_logistic.py); "
        "it is the default here too. The other logistic variants were used in this repo for "
        "hard-to-fit / bounded cases (e.g. market shares)."
    ),
)

fit_kwargs = {}

with st.expander("Advanced fitting parameters", expanded=False):
    if len(x_fit) >= 2:
        year_min, year_max = float(x_fit.min()), float(x_fit.max())
    else:
        year_min, year_max = float(years_all.min()), float(years_all.max())
    if year_min == year_max:
        year_min, year_max = year_min - 5, year_max + 5

    if method == "Logistic — 3-parameter (curve_fit)":
        A_guess_auto = float(np.max(y_fit)) if len(y_fit) and np.max(y_fit) > 0 else 1.0
        half = A_guess_auto / 2
        T_guess_auto = (
            float(x_fit[np.argmin(np.abs(y_fit - half))]) if len(x_fit) else year_min
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            A_guess = st.number_input("Initial guess A (asymptote/K)", value=A_guess_auto)
        with c2:
            M_guess = st.number_input("Initial guess M (growth rate)", value=1.0, format="%.4f")
        with c3:
            T_guess = st.number_input("Initial guess T (midpoint year)", value=T_guess_auto)
        with c4:
            maxfev = st.number_input("Max function evaluations", value=10000, step=1000)
        fit_kwargs = dict(A_guess=A_guess, M_guess=M_guess, T_guess=T_guess, maxfev=int(maxfev))

    elif method == "Logistic — logit-linearization (fast, requires 0<y<1)":
        st.caption(
            "Fits log(y/(1-y)) linearly against year. Assumes the series is already "
            "normalised (K=1), e.g. a market share expressed as a fraction."
        )
        c1, c2 = st.columns(2)
        with c1:
            threshold = st.number_input("Lower threshold", value=0.0, min_value=0.0, max_value=0.99, step=0.01)
        with c2:
            thresholdup = st.number_input("Upper threshold", value=0.0, min_value=0.0, max_value=0.99, step=0.01)
        fit_kwargs = dict(threshold=threshold, thresholdup=thresholdup)

    elif method == "Logistic — bounded global search (differential evolution)":
        K_max_default = float(np.max(y_fit) * 2) if len(y_fit) and np.max(y_fit) > 0 else 1.0
        c1, c2, c3 = st.columns(3)
        with c1:
            t0_bounds = st.slider(
                "t0 bounds (year)", year_min - 300, year_max + 300, (year_min - 50, year_max + 50)
            )
        with c2:
            Dt_bounds = st.slider("Dt bounds", -500.0, 500.0, (0.1, 200.0))
        with c3:
            K_bounds = st.slider("K bounds", 0.0, max(K_max_default, 1.0), (0.0, K_max_default))
        c4, c5 = st.columns(2)
        with c4:
            maxiter = st.number_input("Max iterations", value=1000, step=100)
        with c5:
            seed = st.number_input("Random seed", value=42, step=1)
        fit_kwargs = dict(
            t0_bounds=t0_bounds, Dt_bounds=Dt_bounds, K_bounds=K_bounds,
            maxiter=int(maxiter), seed=int(seed),
        )

    elif method == "Logistic — multi-start local search":
        K_max_default = float(np.max(y_fit) * 2) if len(y_fit) and np.max(y_fit) > 0 else 1.0
        c1, c2, c3 = st.columns(3)
        with c1:
            t0_bounds = st.slider(
                "t0 search range (year)", year_min - 300, year_max + 300,
                (year_min - 50, year_max + 50), key="ms_t0",
            )
        with c2:
            Dt_bounds = st.slider("Dt search range", -500.0, 500.0, (0.1, 200.0), key="ms_dt")
        with c3:
            K_bounds = st.slider("K search range", 0.0, max(K_max_default, 1.0), (0.0, K_max_default), key="ms_k")
        c4, c5, c6 = st.columns(3)
        with c4:
            n_starts = st.number_input("Number of random starts", value=20, min_value=1, step=5)
        with c5:
            opt_method = st.selectbox("Optimizer", ["BFGS", "Nelder-Mead", "Powell"])
        with c6:
            seed = st.number_input("Random seed", value=42, step=1, key="ms_seed")
        fit_kwargs = dict(
            t0_bounds=t0_bounds, Dt_bounds=Dt_bounds, K_bounds=K_bounds,
            n_starts=int(n_starts), method=opt_method, seed=int(seed),
        )

    elif method == "Exponential":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            a0 = st.number_input("Initial guess a", value=10.0)
        with c2:
            b0 = st.number_input("Initial guess b", value=0.001, format="%.5f")
        with c3:
            c0 = st.number_input("Initial guess c", value=float(np.median(x_fit)) if len(x_fit) else year_min)
        with c4:
            maxfev = st.number_input("Max function evaluations", value=2000, step=500, key="exp_maxfev")
        fit_kwargs = dict(a0=a0, b0=b0, c0=c0, maxfev=int(maxfev))

    else:  # Linear
        st.caption("No tunable parameters for ordinary least-squares linear regression.")


# ──────────────────────────────────────────────────────────────
# 6. Run the fit
# ──────────────────────────────────────────────────────────────

fit_result = None
predict_fn = None
n_params = 0

if included_mask.sum() < 2:
    st.warning("At least 2 included points are required to fit a curve.")
else:
    if method == "Logistic — 3-parameter (curve_fit)":
        fit_result = fit_logistic_curvefit(x_fit, y_fit, **fit_kwargs)
        n_params = 3
    elif method == "Logistic — logit-linearization (fast, requires 0<y<1)":
        fit_result = fit_logistic_logit_linear(x_fit, y_fit, **fit_kwargs)
        n_params = 2
    elif method == "Logistic — bounded global search (differential evolution)":
        fit_result = fit_logistic_diffevo(x_fit, y_fit, **fit_kwargs)
        n_params = 3
    elif method == "Logistic — multi-start local search":
        fit_result = fit_logistic_multistart(x_fit, y_fit, **fit_kwargs)
        n_params = 3
    elif method == "Exponential":
        fit_result = fit_exponential(x_fit, y_fit, **fit_kwargs)
        n_params = 3
    else:
        fit_result = fit_linear(x_fit, y_fit)
        n_params = 2

    if fit_result is not None:
        if method.startswith("Logistic"):
            predict_fn = lambda xx, r=fit_result: FPLogValue_with_scaling(xx, r["t0"], r["Dt"], r["K"])
        elif method == "Exponential":
            predict_fn = lambda xx, r=fit_result: exponential_func(xx, r["a"], r["b"], r["c"])
        else:
            predict_fn = lambda xx, r=fit_result: r["slope"] * xx + r["intercept"]


# ──────────────────────────────────────────────────────────────
# 7. Results: metrics, parameters, plot
# ──────────────────────────────────────────────────────────────

st.subheader("4. Fit results")

if fit_result is None:
    st.error(
        "Fit failed (or too few points included). Try adjusting initial guesses/bounds, "
        "or include more points."
    )
else:
    y_pred_fit = predict_fn(x_fit)
    metrics = calc_metrics(y_fit, y_pred_fit, n_params)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R²", f"{metrics['r2']:.3f}" if not np.isnan(metrics["r2"]) else "n/a")
    m2.metric("Adj. R²", f"{metrics['r2_adj']:.3f}" if not np.isnan(metrics["r2_adj"]) else "n/a")
    m3.metric("RMSE", f"{metrics['rmse']:.3g}" if not np.isnan(metrics["rmse"]) else "n/a")
    m4.metric("MAE", f"{metrics['mae']:.3g}" if not np.isnan(metrics["mae"]) else "n/a")

    st.write("**Fitted parameters**")
    param_display = {
        k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v)
        for k, v in fit_result.items()
    }
    st.json(param_display)

    export_row = {
        "series": series_key,
        "method": method,
        **param_display,
        **{k: (round(float(v), 4) if not np.isnan(v) else None) for k, v in metrics.items()},
        "n_points_included": int(included_mask.sum()),
        "n_points_total": int(len(years_all)),
    }
    st.download_button(
        "Download fit parameters (CSV)",
        pd.DataFrame([export_row]).to_csv(index=False),
        file_name="single_series_fit.csv",
        mime="text/csv",
    )

st.markdown("**Compare against fits already stored in the summary table**")
reference_choices = st.multiselect(
    "Overlay stored pipeline fit(s)",
    ["Logistic", "Exponential", "Linear"],
    default=["Logistic"] if summary_df is not None else [],
    disabled=summary_df is None,
)

ref_row = None
if summary_df is not None:
    matches = summary_df[
        (summary_df["Innovation Name"] == sel_innovation)
        & (summary_df["Spatial Scale"] == sel_spatial)
        & (summary_df["Indicator Number"] == sel_indicator)
        & (summary_df["Description"] == sel_description)
        & (summary_df["Metric"] == sel_metric)
    ]
    if len(matches) > 0:
        ref_row = matches.iloc[0]
    elif reference_choices:
        st.info("No matching entry found in the summary table for this exact series.")

log_y_axis = st.checkbox("Log-scale y-axis", value=False)

year_min_plot = years_all.min() - YEAR_PADDING_FOR_PLOTTING
year_max_plot = years_all.max() + YEAR_PADDING_FOR_PLOTTING
x_line = np.linspace(year_min_plot, year_max_plot, int((year_max_plot - year_min_plot) * 4) + 1)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=years_all[included_mask],
        y=values_all[included_mask],
        mode="markers",
        name="Included points",
        marker=dict(size=10, color="#1f77b4", line=dict(width=1, color="#333333")),
    )
)
if excluded_mask.any():
    fig.add_trace(
        go.Scatter(
            x=years_all[excluded_mask],
            y=values_all[excluded_mask],
            mode="markers",
            name="Excluded points",
            marker=dict(size=11, color="#bbbbbb", symbol="x"),
        )
    )

if predict_fn is not None:
    fig.add_trace(
        go.Scatter(
            x=x_line, y=predict_fn(x_line), mode="lines", name=f"Fit: {method}",
            line=dict(color="#d62728", width=2.5),
        )
    )

if ref_row is not None:
    if "Logistic" in reference_choices and pd.notna(ref_row.get("log_t0")) and pd.notna(ref_row.get("log_Dt")) and pd.notna(ref_row.get("log_K")):
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=FPLogValue_with_scaling(x_line, ref_row["log_t0"], ref_row["log_Dt"], ref_row["log_K"]),
                mode="lines", name="Stored: Logistic",
                line=dict(color="#7f7f7f", width=2, dash="dash"),
            )
        )
    if "Exponential" in reference_choices and pd.notna(ref_row.get("exp_a")) and pd.notna(ref_row.get("slope_exp")) and pd.notna(ref_row.get("exp_c")):
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=exponential_func(x_line, ref_row["exp_a"], ref_row["slope_exp"], ref_row["exp_c"]),
                mode="lines", name="Stored: Exponential",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
            )
        )
    if "Linear" in reference_choices and pd.notna(ref_row.get("slope_lin")) and pd.notna(ref_row.get("lin_intercept")):
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=ref_row["slope_lin"] * x_line + ref_row["lin_intercept"],
                mode="lines", name="Stored: Linear",
                line=dict(color="#2ca02c", width=2, dash="dash"),
            )
        )

fig.update_layout(
    title=f"{sel_innovation} — {sel_spatial}",
    xaxis_title="Year",
    yaxis_title=sel_metric,
    height=650,
)
if log_y_axis:
    fig.update_yaxes(type="log")

st.plotly_chart(fig, width="stretch")
