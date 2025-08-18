# File begun on 26 November 2024

import pandas as pd
import numpy as np

import re
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit, differential_evolution, minimize
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.backends.backend_pdf import PdfPages

from logfits_chatgpt_v1 import fit_logistic_3p

VERSION = "v27"
VERSION_FOR_FITS = "v26"
VERSION_FOR_SUMMARY_READING = "v25"
VERSION_FOR_METADATA = "v25"
VERSION_FOR_DATA = "v26"
SMALL_SUBSET = False  # Do you only want a small subset for testing?
REDO_FITS = True
RENUMBER_METADATA_CODES = False
CREATE_PDFS = True

LINE_COLOR_LOG = "blue"

PATH = "/mnt/c/Users/simon.destercke/Documents/misc/iiasa/DoSI"
fn_data = f"{PATH}/adjusted_datasets_{VERSION_FOR_DATA}.csv"

adoptions_df = pd.read_csv(fn_data, converters={"Indicator Number": str})
adoptions_df["Value"] = pd.to_numeric(adoptions_df["Value"], errors="coerce")
adoptions_df = adoptions_df.dropna(subset=["Value"])

# Correct for trailing spaces in the data
adoptions_df["Spatial Scale"] = adoptions_df["Spatial Scale"].str.rstrip()
adoptions_df["Innovation Name"] = adoptions_df["Innovation Name"].str.rstrip()

# For debugging: only subset
if SMALL_SUBSET:
    adoptions_df = adoptions_df[
        # adoptions_df["Innovation Name"].isin(["car sharing"])
        # adoptions_df["Indicator Number"].isin(["3.3", "3.5", "4.1"])
        # adoptions_df["Indicator Number"].isin(["1.1"])
        adoptions_df["Metric"].isin(["market share"])
    ]

fn_metadata = f"{PATH}/metadata_master_{VERSION_FOR_METADATA}.xlsx"


def convert_to_three_digit_notation(s):
    return re.sub(r"([a-zA-Z])(\d+)", lambda m: f"{m.group(1)}{int(m.group(2)):03}", s)


def read_metadata_table(fn, columns):
    df = pd.read_excel(fn, usecols=columns, dtype=str).dropna().reset_index(drop=True)
    df.iloc[:, 1] = df.iloc[:, 1].apply(convert_to_three_digit_notation)
    return df.set_index(df.columns[0])[df.columns[1]].to_dict()


categories_df = (
    pd.read_excel(fn_metadata, usecols="A,B", dtype=str).dropna().reset_index(drop=True)
)
categories = {
    (k.lower() if isinstance(k, str) else k): v
    for k, v in categories_df.set_index(categories_df.columns[0])[
        categories_df.columns[1]
    ].items()
}

metadata = dict()
metadata["Innovation Name"] = read_metadata_table(fn_metadata, "A,D")
metadata["Spatial Scale"] = read_metadata_table(fn_metadata, "G,I")
metadata["Indicator Number"] = read_metadata_table(
    fn_metadata, "L,O"
)  # Column M is the indicator name. Superfluous because maps 1-1 on indicator number
metadata["Description"] = read_metadata_table(fn_metadata, "R,S")
metadata["Metric"] = read_metadata_table(fn_metadata, "V,W")

# If metadata file is not in sync with the data table, remake the description dictionary
if RENUMBER_METADATA_CODES:
    metadata["Description"] = {
        val: convert_to_three_digit_notation(f"d{idx}")
        for idx, val in enumerate(sorted(adoptions_df["Description"].unique()), start=1)
    }
    # If metadata file is not in sync with the data table, remake the metric dictionary
    metadata["Metric"] = {
        val: convert_to_three_digit_notation(f"m{idx}")
        for idx, val in enumerate(sorted(adoptions_df["Metric"].unique()), start=1)
    }

_, last_value = next(reversed(metadata["Description"].items()))
description_counter = int(last_value[1:])  # Presumes a format "d023"
_, last_value = next(reversed(metadata["Metric"].items()))
metric_counter = int(last_value[1:])  # Presumes a format "m023"

for key, nested_dict in metadata.items():
    if isinstance(nested_dict, dict):  # Ensure the value is a dictionary
        metadata[key] = {
            k.lower() if isinstance(k, str) else k: v for k, v in nested_dict.items()
        }

group_vars = list(metadata.keys())

# Failed attempt below to go for alphabetic ordering of codes
grouped = adoptions_df.groupby(group_vars)

codes = []
groups = []
# Loop through each group and create the code
for group_name, group_data in grouped:
    groups.append(group_name)
    codes.append(
        "_".join(
            [
                metadata[group_vars[i]][group_name[i].lower()]
                for i in range(len(metadata))
            ]
        )
    )
sorted_indices = sorted(range(len(codes)), key=lambda i: codes[i])

group_vars.insert(3, "Indicator Name")  # Insert the indicator name after the number

# Group the data
grouped = adoptions_df.groupby(group_vars)
grouped_as_list = list(grouped)

scoring_table = pd.read_csv(
    f"""{PATH}/summary_table_{VERSION_FOR_SUMMARY_READING}.csv"""
)


def FPLogFit(x, y, threshold=0, thresholdup=0):
    # Filter data based on the threshold conditions
    mask = (y > threshold) & (y < 1 - thresholdup) & (y < 1)
    x_filtered = x[mask]
    y_filtered = y[mask]

    # Check if there are fewer than 2 valid points
    if len(x_filtered) < 2:
        return {"t0": 2300, "a": np.log(81) / 10}
    else:
        # Compute the logit transformation: log(y / (1 - y))
        logits = np.log(y_filtered / (1 - y_filtered))

        # Fit a linear model: log(y / (1 - y)) ~ x
        x_reshaped = (
            x_filtered.values.reshape(-1, 1)
            if isinstance(x_filtered, pd.Series)
            else x_filtered.reshape(-1, 1)
        )
        model = LinearRegression()
        model.fit(x_reshaped, logits)

        # Extract coefficients
        a = model.coef_[0]  # Slope
        intercept = model.intercept_  # Intercept

        # Adjust t0 based on the coefficients
        t0 = -intercept / a

        return {"t0": t0, "Dt": np.log(81) / a}


def FPLogValue(t0, Dt, Year):
    return 1 / (1 + np.exp(-np.log(81) / Dt * (Year - t0)))


def FPLogValue_with_scaling(x, t0, Dt, s):
    """
    Logistic function with vertical scaling.
    """
    return s / (1 + np.exp(-np.log(81) * (x - t0) / Dt))


def FPLogFit_with_scaling(x, y, Dt_initial_guess: float = 10):
    """
    Fit a logistic function with vertical scaling to the data.

    Parameters:
    x (array-like): Independent variable.
    y (array-like): Dependent variable.

    Returns:
    dict: Fitted parameters t0, Dt, and S.
    """

    if len(x) < 3:
        return {"t0": None, "Dt": None, "K": None}  # Default parameters
    else:
        # # Initial guesses for the parameters
        # initial_guess = [np.median(x), Dt_initial_guess, np.max(y)]

        # # Fit the logistic function
        # try:
        #     params, _ = curve_fit(
        #         FPLogValue_with_scaling,
        #         x,
        #         y,
        #         p0=initial_guess,
        #         method="trf",
        #         maxfev=5000,
        #     )
        #     t0, Dt, k = params
        # except RuntimeError:
        #     print("RuntimeError")
        #     return {"t0": None, "Dt": None, "K": None}  # Handle fitting failure
        # # if t0 + Dt < 2000 and firstrun:
        # #     return FPLogFit_with_scaling(
        # #         x, y, Dt_initial_guess=-5, firstrun=False
        # #     )  # Only one additional attempt
        # # else:
        # return {"t0": t0, "Dt": Dt, "K": k}

        popt_log, pcov_log, rmse_log = fit_logistic_3p(x, y)
        if popt_log is not None:
            A_fit, M_fit, T_fit = popt_log
            A_err, M_err, T_err = np.sqrt(np.diag(pcov_log))
        else:
            A_fit = M_fit = T_fit = A_err = M_err = T_err = np.nan

        return {"t0": T_fit, "Dt": np.log(81) / M_fit, "K": A_fit}


# Define the exponential function
def exponential_func(x, a, b, c):
    return a * np.exp(b * (x - c))


# Apply FPLogfit to each group
def apply_FPLogFit_with_scaling(group):
    slope, _, _, _, _ = linregress(group["Year"].values, group["Value"].values)
    result = FPLogFit_with_scaling(
        group["Year"],
        group["Value"],
        Dt_initial_guess=(
            np.log(81) / slope * (max(group["Value"]) if max(group["Value"]) > 0 else 1)
            if slope != 0
            else 100
        ),
    )
    return pd.Series(result)


# Curve fitting for a single group
def apply_exponential_fit(group_df):
    x = group_df["Year"].values
    y = group_df["Value"].values

    try:
        # Fit the data
        popt, _ = curve_fit(
            exponential_func,
            x,
            y,
            p0=[10, 0.001, 1950],
            maxfev=2000,
        )
        result = {"a": popt[0], "b": popt[1], "c": popt[2]}  # a, b, c
    except Exception as e:
        # Handle fitting failure
        result = {"a": None, "b": None, "c": None}
    return pd.Series(result)


# Linear
def apply_linear_fit(group_df):
    slope, intercept, r_value, p_value, std_err = linregress(
        group_df["Year"].values, group_df["Value"].values
    )
    return pd.Series(
        {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err,
        }
    )


# Calculate R^2 and adjusted R^2
def calculate_adjusted_r2(y_obs, y_pred, n_params):
    if np.any(np.isnan(y_pred)):  # If predictions are NaN, return NaN
        return np.nan, np.nan
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    n = len(y_obs)
    r2_adj = 1 - ((1 - r2) * (n - 1) / (n - n_params - 1))
    return r2, r2_adj


def exclude_rule_drop_at_end(data_series, drop_threshold_pct=0.9):
    """Identify years to be excluded because of a drop below 90% of maximum

    Parameters
    ----------
    data_series : data frame
        input data frame
    drop_threshold_pct: float
        percentage of maximum value to decide which trailing points to exclude
    """

    data_series_max = max(data_series["Value"])
    data_series.sort_values(by="Year")
    list_of_exclusions = [True] * len(data_series)


# Alternative fitting for market shares that need to be constrained
# Define an objective (cost) function
def objective(params, t, y):
    t0, Dt, s = params
    y_pred = FPLogValue_with_scaling(t, t0, Dt, s)
    return np.sum((y - y_pred) ** 2)


def alternative_log_fit(
    x,
    y,
    bounds=[
        (1000, 3000),  # t0
        (-500, 500),  # Dt
        (1e-10, 1),  # s or K (asymptote)
    ],
):
    result = differential_evolution(
        objective,
        bounds=bounds,
        args=(group_data["Year"], group_data["Value"]),
        maxiter=1000,  # You can increase from 1000 if needed
        seed=42,
    )
    t0 = result["x"][0]
    Dt = result["x"][1]
    k = result["x"][2]
    # return pd.Series({"t0": t0, "Dt": Dt, "K": k})
    return t0, Dt, k


if REDO_FITS:
    results_logistic = (
        adoptions_df.groupby(group_vars)
        .apply(apply_FPLogFit_with_scaling)
        .reset_index()
    )
    results_exponential = (
        adoptions_df.groupby(group_vars).apply(apply_exponential_fit).reset_index()
    )
    results_linear = (
        adoptions_df.groupby(group_vars).apply(apply_linear_fit).reset_index()
    )

    # Save to Pickle files
    results_logistic.to_pickle(f"results_logistic_{VERSION}.pkl")
    results_exponential.to_pickle(f"results_exponential_{VERSION}.pkl")
    results_linear.to_pickle(f"results_linear_{VERSION}.pkl")
else:
    results_logistic = pd.read_pickle(f"results_logistic_{VERSION_FOR_FITS}.pkl")
    results_exponential = pd.read_pickle(f"results_exponential_{VERSION_FOR_FITS}.pkl")
    results_linear = pd.read_pickle(f"results_linear_{VERSION_FOR_FITS}.pkl")

print(
    f"""{results_logistic["t0"].isnull().sum()} out of {len(results_logistic)} logistic fits failed"""
)
print(
    f"""{results_exponential["a"].isnull().sum()} out of {len(results_exponential)} exponential fits failed"""
)

# Scatterplots

LINE_X_BUFFER = 10

DISTANCE_TO_MEAN_THRESHOLD = 0.4
AUTOCORRELATION_THRESHOLD = 0.70
K_COVERAGE_THRESHOLD = 0.2
PERCENT_JUMP_THRESHOLD = 250
PERCENT_FALL_THRESHOLD = 80

summary_table_rows = (
    []
)  # list to which to append the summary statistics for each series

COMMON_DATABASES_INDICATOR_CODES = [
    "3.3",
    "3.5",
    "4.1",
]  # to be written to a separate pdf

if CREATE_PDFS:
    pdf_commondb = PdfPages(f"{PATH}/scatterplots_{VERSION}_COMMON.pdf")
    pdf_other = PdfPages(f"{PATH}/scatterplots_{VERSION}_OTHER.pdf")
    pdf_marketshares = PdfPages(f"{PATH}/scatterplots_{VERSION}_MARKETSHARES.pdf")
    pdf_all = PdfPages(f"{PATH}/scatterplots_{VERSION}_ALL.pdf")
    pdf_allexceptmarkedfordeletion = PdfPages(
        f"{PATH}/scatterplots_{VERSION}_ALLexceptmarkedfordeletion.pdf"
    )

adjusted_dfs = []  # For storing the adjusted data frames

# Loop through each group and create a scatterplot
for i in range(len(grouped)):

    print(f"\rProgress: {(i*100.0/len(grouped)):.1g}%", end="", flush=True)

    sorted_index = sorted_indices[i]

    code = codes[sorted_index]

    group_name, group_data = grouped_as_list[sorted_index]

    group_data.sort_values(by=["Year"], inplace=True)

    group_data.reset_index(drop=True, inplace=True)

    # Get some key values for diagnostics of the series
    n_data_points = len(group_data)
    non_zero_data_index_boolean = group_data["Value"] != 0
    non_zero_data_index_list = [
        j for j, val in enumerate(non_zero_data_index_boolean) if val
    ]
    non_zero_data_points = group_data[non_zero_data_index_boolean]
    n_non_zero_data_points = len(non_zero_data_points)
    first_year_non_zero = group_data[non_zero_data_index_boolean]["Year"].min()
    last_year_non_zero = group_data[non_zero_data_index_boolean]["Year"].max()

    if n_non_zero_data_points == 0:
        continue

    trimmed_data_points = group_data["Value"][
        min(non_zero_data_index_list) : (max(non_zero_data_index_list) + 1)
    ]
    year_range = last_year_non_zero - first_year_non_zero + 1
    min_value_non_zero = group_data[non_zero_data_index_boolean]["Value"].min()
    min_non_zero_index = group_data[non_zero_data_index_boolean]["Value"].idxmin()
    max_value = group_data["Value"].max()
    max_index = group_data["Value"].idxmax()
    n_data_points_beyond_min = max(non_zero_data_index_list) - min_non_zero_index
    n_data_points_beyond_max = max(non_zero_data_index_list) - max_index
    relative_distance_min_to_avg_year = abs(
        group_data["Year"][min_non_zero_index]
        - np.mean([first_year_non_zero, last_year_non_zero])
    ) / (last_year_non_zero - first_year_non_zero)
    relative_distance_max_to_avg_year = abs(
        group_data["Year"][max_index]
        - np.mean([first_year_non_zero, last_year_non_zero])
    ) / (last_year_non_zero - first_year_non_zero)
    relative_changes = np.diff(trimmed_data_points) / trimmed_data_points[:-1]
    # Compute autocorrelation
    autocorr = np.correlate(trimmed_data_points, trimmed_data_points, mode="full")
    # Normalize to get autocorrelation values
    autocorr = autocorr / autocorr.max()
    # Extract the autocorrelation for non-negative lags
    autocorr = autocorr[len(trimmed_data_points) - 1 :]

    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    plt.grid(True)

    x_line = np.arange(
        min(group_data["Year"]) - LINE_X_BUFFER,
        max(group_data["Year"]) + LINE_X_BUFFER,
        0.1,
    )

    # Logistic fit line
    results_filtered = results_logistic[
        results_logistic[group_vars].apply(tuple, axis=1).isin([group_name])
    ]
    t0 = results_filtered["t0"].values[0]
    Dt = results_filtered["Dt"].values[0]
    k = results_filtered["K"].values[0]

    # Nudges!

    if (group_data["Metric"].unique() == "market share") & (k > 1):
        t0, Dt, k = alternative_log_fit(group_data["Year"], group_data["Value"])

    if code == "ebi_net_1.1Ado_d333_m185":
        t0, Dt, k = alternative_log_fit(
            group_data["Year"],
            group_data["Value"],
            bounds=[(1000, 3000), (0.1, 500), (1e-10, 1)],
        )

    if code == "sus_glo_1.1Ado_d336_m185":
        t0, Dt, k = alternative_log_fit(
            group_data["Year"],
            group_data["Value"],
            bounds=[(1000, 3000), (0.1, 500), (1e-10, 1)],
        )

    if code == "crz_fra_1.1Ado_d328_m185":
        t0, Dt, k = alternative_log_fit(
            group_data["Year"],
            group_data["Value"],
            bounds=[(500, 3000), (200, 800), (0.5, 1)],
        )

    if code == "foo_usa_1.1Ado_d337_m185":
        t0, Dt, k = alternative_log_fit(
            group_data["Year"],
            group_data["Value"],
            bounds=[(2000, 2030), (0.1, 20), (0.5, 1)],
        )

    y_line_log = FPLogValue_with_scaling(x_line, t0, Dt, k)
    y_pred = FPLogValue_with_scaling(group_data["Year"], t0, Dt, k)
    r2_log, r2adj_log = calculate_adjusted_r2(group_data["Value"], y_pred, n_params=3)
    rmse_log = np.sqrt(np.mean((group_data["Value"] - y_pred) ** 2))
    mae_log = np.mean(np.abs(group_data["Value"] - y_pred))
    plt.plot(
        x_line, y_line_log, color=LINE_COLOR_LOG, label="Logistic", linewidth=1.6
    )  # , marker="+")

    # Exponential fit line
    line_color_exp = "red"
    results_filtered = results_exponential[
        results_exponential[group_vars].apply(tuple, axis=1).isin([group_name])
    ]
    a = results_filtered["a"].values[0]
    b = results_filtered["b"].values[0]
    c = results_filtered["c"].values[0]
    y_line_exp = exponential_func(x_line, a, b, c)
    y_pred = exponential_func(group_data["Year"], a, b, c)
    r2_exp, r2adj_exp = calculate_adjusted_r2(group_data["Value"], y_pred, n_params=2)
    rmse_exp = np.sqrt(np.mean((group_data["Value"] - y_pred) ** 2))
    mae_exp = np.mean(np.abs(group_data["Value"] - y_pred))
    plt.plot(x_line, y_line_exp, color=line_color_exp, label="Exponential", linewidth=1)

    # Linear regression line
    line_color_lin = "green"
    results_filtered = results_linear[
        results_linear[group_vars].apply(tuple, axis=1).isin([group_name])
    ]
    slope = results_filtered["slope"].values[0]
    intercept = results_filtered["intercept"].values[0]
    y_line_lin = slope * x_line + intercept
    y_pred = slope * group_data["Year"] + intercept
    r2_lin, r2adj_lin = calculate_adjusted_r2(group_data["Value"], y_pred, n_params=2)
    rmse_lin = np.sqrt(np.mean((group_data["Value"] - y_pred) ** 2))
    mae_lin = np.mean(np.abs(group_data["Value"] - y_pred))
    plt.plot(x_line, y_line_lin, color=line_color_lin, label="Linear", linewidth=1)

    plt.scatter(
        group_data["Year"],
        group_data["Value"],
        label="Data Points",
        color="black",
        s=40,
    )

    # Try table on top:

    column_labels = [
        "Curve type",
        "Curve parameters",
        "Slope",
        "R2",
        "R2adj",
        "RMSE",
        "MAE",
    ]

    # Add a table
    table_data = [
        column_labels,
        [
            "Logistic",
            f"""t0={t0:.0f}, Dt={Dt:.3g}, K={k:.3g}""",
            f"""{np.log(81)/Dt:.3g}""",
            f"{r2_log:.3g}",
            f"{r2adj_log:.3g}",
            f"{rmse_log:.3g}",
            f"{mae_log:.3g}",
        ],
        [
            "Exponential",
            f"""{a:.3g}*exp({b:.3g}*(x-{c:.0f}))""",
            f"""{b:.3g}""",
            f"{r2_exp:.3g}",
            f"{r2adj_exp:.3g}",
            f"{rmse_exp:.3g}",
            f"{mae_exp:.3g}",
        ],
        [
            "Linear",
            f"""intercept={intercept:.3g}, slope={slope:.3g}""",
            f"""{slope:.3g}""",
            f"{r2_lin:.3g}",
            f"{r2adj_lin:.3g}",
            f"{rmse_lin:.3g}",
            f"{mae_lin:.3g}",
        ],
    ]
    table_colors = [
        ["black"] * len(column_labels),
        [LINE_COLOR_LOG, LINE_COLOR_LOG] + ["black"] * (len(column_labels) - 2),
        [line_color_exp, line_color_exp] + ["black"] * (len(column_labels) - 2),
        [line_color_lin, line_color_lin] + ["black"] * (len(column_labels) - 2),
    ]

    # Create a table object
    table = Table(ax, bbox=[0.35, 0.99, 0.64, 0.15])  # Adjust bbox for positioning
    nrows, ncols = len(table_data), len(table_data[0])

    for row in range(nrows):
        for col in range(ncols):
            cell_text = table_data[row][col]
            cell_color = table_colors[row][col]
            cell = table.add_cell(
                row,
                col,
                width=0.1,
                height=0.1,
                text=cell_text,
                loc="left",
                facecolor="white",
                edgecolor="black",
            )
            cell.set_facecolor("#ffffff")
            cell.set_fontsize(8)
            cell.set_alpha(1.0)
            cell.set_text_props(color=cell_color)  # Set text color for the cell
    table.auto_set_column_width(col=list(range(len(column_labels))))

    # Add the table to the axes
    ax.add_table(table)
    table.set_zorder(5)

    # Find max y for plotting
    lines = ax.get_lines()

    # Find the maximum y-value
    max_y_lines = max(max(line.get_ydata()) for line in lines)

    # Plot the code/label in the graph, top left
    plt.text(
        x=0.01,  # Position on the x-axis (leftmost)
        y=0.99,  # Position on the y-axis (topmost, adjust as needed)
        s=code,  # Text to display
        transform=plt.gca().transAxes,
        fontsize=20,
        verticalalignment="top",  # Align text vertically to the top
        horizontalalignment="left",  # Align text horizontally to the left
    )

    title_list = list(group_name)
    title_list[2:4] = [title_list[2] + " " + title_list[3]]
    # title_list += ["[" + " ".join(f"{x:.3g}" for x in autocorr[:5]) + "]"]

    plt.title("\n".join(title_list), loc="left")
    plt.xlabel("Year")
    plt.ylabel(group_name[-1])
    plt.ylim(
        bottom=0,
        top=max(
            max(group_data["Value"]) * 1.1,
            min(max(group_data["Value"]) * 2, max_y_lines),
        ),
    )

    if CREATE_PDFS:
        # Save the current plot to the PDF
        pdf_all.savefig()
        if not (
            all(
                scoring_table[scoring_table["Code"] == code]["Delete from working file"]
                == "delete"
            )
        ):
            pdf_allexceptmarkedfordeletion.savefig()
        if group_name[2] in COMMON_DATABASES_INDICATOR_CODES:
            pdf_commondb.savefig()  # Save current figure into the PDF
        elif group_name[group_vars.index("Metric")] == "market share":
            pdf_marketshares.savefig()
        else:
            pdf_other.savefig()

    plt.close()  # Close the figure to free memory

    # Create the row for the summary table

    summary_table_dict = {
        "Code": code,
        **{
            f"{value}": group_name[i] for i, value in enumerate(group_vars)
        },  # Dynamically add group_vars
        "Category": categories[group_name[0].lower()],  # Lookup category dynamically
        "slope_log": np.log(81) / Dt,
        "slope_exp": b,
        "slope_lin": slope,
        "log_t0": t0,
        "log_Dt": Dt,
        "log_K": k,
        "exp_a": a,
        "exp_c": c,
        "lin_intercept": intercept,
        "log_r2": r2_log,
        "log_r2adj": r2adj_log,
        "log_rmse": rmse_log,
        "log_mae": mae_log,
        "exp_r2": r2_exp,
        "exp_r2adj": r2adj_exp,
        "exp_rmse": rmse_exp,
        "exp_mae": mae_exp,
        "lin_r2": r2_lin,
        "lin_r2adj": r2adj_lin,
        "lin_rmse": rmse_lin,
        "lin_mae": mae_lin,
        "n_data_points": len(group_data),
        "n_non_zero_data_points": (group_data["Value"] != 0).sum(),
        "max_over_K": max_value / k,
        "min_over_K": min_value_non_zero / k,
        "range_over_k": (max_value - min_value_non_zero) / k,
        "length_trimmed_series_years": year_range,
        "n_data_points_beyond_max": n_data_points_beyond_max,
        "n_data_points_beyond_min": n_data_points_beyond_min,  # CRITERIA_START
        "suspected_reversal_up2down": int(
            relative_distance_max_to_avg_year < DISTANCE_TO_MEAN_THRESHOLD
        ),
        "suspected_reversal_down2up": int(
            relative_distance_min_to_avg_year < DISTANCE_TO_MEAN_THRESHOLD
        ),
        "at_least_one_big_jump": (
            int((max(relative_changes) > 1) | (min(relative_changes) < -0.5))
            if len(relative_changes) > 0
            else None
        ),
        f"autocorr_l1": autocorr[1] if len(relative_changes) > 0 else None,
        "all_values_less_than_or_equal_to_1": int(max_value <= 1),
        "all_values_less_than_or_equal_to_100": int(max_value <= 100),
        "C1_R2": "y" if r2_log > 0.7 else ("m" if r2_log > 0.3 else "n"),
        "C2_threshold_non_zero_value_years": (
            "y" if n_non_zero_data_points > 5 else ("m" if year_range > 10 else "n")
        ),
        "C3_pct_non_zero": (
            "m" if n_non_zero_data_points / n_data_points < 0.1 else "y"
        ),
        f"C4_less_than_{PERCENT_JUMP_THRESHOLD}_jump_in_second_half": (
            "m"
            if (
                len(relative_changes) < 1
            )  # Catch series with not enough data points to calculate relative changes
            else (
                "y"
                if (
                    max((relative_changes[int(len(relative_changes) / 2) :]))
                    < PERCENT_JUMP_THRESHOLD / 100
                )
                else "m"
            )
        ),
        f"C4_less_than_{PERCENT_FALL_THRESHOLD}_fall_in_second_half": (
            "m"
            if (
                len(relative_changes) < 1
            )  # Catch series with not enough data points to calculate relative changes
            else (
                "y"
                if (
                    min((relative_changes[int(len(relative_changes) / 2) :]))
                    > -PERCENT_FALL_THRESHOLD / 100
                )
                else "m"
            )
        ),
        f"C4_combined_less_than_{PERCENT_FALL_THRESHOLD}_fall_and_{PERCENT_JUMP_THRESHOLD}_jump_in_second_half": (
            "m"
            if (
                len(relative_changes) < 1
            )  # Catch series with not enough data points to calculate relative changes
            else (
                "y"
                if (
                    max((relative_changes[int(len(relative_changes) / 2) :]))
                    < PERCENT_JUMP_THRESHOLD / 100
                )
                & (
                    min((relative_changes[int(len(relative_changes) / 2) :]))
                    > -PERCENT_FALL_THRESHOLD / 100
                )
                else "m"
            )
        ),
        f"C5_volatility_autocorrelation_lag1_threshold_{AUTOCORRELATION_THRESHOLD:.2g}": (
            "m"
            if (
                len(relative_changes) < 1
            )  # Catch series with not enough data points to calculate relative changes
            else ("m" if autocorr[1] < AUTOCORRELATION_THRESHOLD else "y")
        ),
        "C6_{K_COVERAGE_THRESHOLD}_of_k_covered_by_data": (
            "y" if (max_value - min_value_non_zero) / k > K_COVERAGE_THRESHOLD else "m"
        ),
        "C7_lin_r2": "y" if r2_lin > 0.3 else "m",  # CRITERIA_END
    }

    criteria_dictionary_short_name = {  # Criteria dictionary short names
        re.match(r"^(C\d+)", k).group(1): v
        for k, v in summary_table_dict.items()
        if re.match(r"^(C\d+)", k)
    }

    def crit(j):
        return criteria_dictionary_short_name[f"C{j}"]

    summary_table_dict["use_log"] = (
        "y"
        if all([crit(j) == "y" for j in range(1, 7)])
        else ("n" if any([crit(j) == "n" for j in range(1, 7)]) else "m")
    )
    summary_table_dict["use_lin"] = (
        "y"
        if all([crit(j) == "y" for j in [2, 3, 4, 5, 7]])
        else ("n" if any([crit(j) == "n" for j in [2, 3, 4]]) else "m")
    )

    summary_table_rows.append(summary_table_dict)

if CREATE_PDFS:
    pdf_commondb.close()
    pdf_marketshares.close()
    pdf_other.close()
    pdf_all.close()
    pdf_allexceptmarkedfordeletion.close()
    print(f"Scatterplots version {VERSION} saved to pdf.")

summary_df = pd.DataFrame(summary_table_rows)

# Add in previous scoring from Charlie and Greg
scoring_on_summary = pd.read_excel(
    f"{PATH}/summary_table_v24_21Mar.xlsx",
    sheet_name="summary_table_v24_CW",
)
scoring_on_summary["Innovation Name"] = scoring_on_summary[
    "Innovation Name"
].str.lower()

# Remove indicator name from code
summary_df["code_without_indicator_name"] = summary_df["Code"].str.replace(
    r"(_\d+\.\d+)[A-Za-z]{3}", r"\1", regex=True
)
scoring_on_summary["code_without_indicator_name"] = scoring_on_summary[
    "Code"
].str.replace(r"(_\d+\.\d+)[A-Za-z]{3}", r"\1", regex=True)


#################
# Include reallocations of indicator numbers that Charlie made
def reassign_indicator_numbers(
    df,  # column name
    innovation_name,
    descriptions,
    metrics,
    old_indicator_number,
    new_indicator_number,
    cn="code_without_indicator_name",
):
    mask = (
        (
            (descriptions == [])
            | (
                df[cn].str.contains(
                    "|".join(
                        [metadata["Description"][key.lower()] for key in descriptions]
                    ),
                    case=False,
                    na=False,
                )
            )
        )
        & (
            (metrics == [])
            | (
                df[cn].str.contains(
                    "|".join([metadata["Metric"][key.lower()] for key in metrics]),
                    case=False,
                    na=False,
                )
            )
        )
    ) & (df[cn].str.contains(innovation_name))
    df.loc[mask, cn] = df.loc[mask, cn].str.replace(
        f"_{old_indicator_number}", f"_{new_indicator_number}"
    )
    print(f"Changed {sum(mask)} codes")
    return df


scoring_on_summary = reassign_indicator_numbers(
    scoring_on_summary,
    "dri",
    [
        "% of population holding a drivers licence, by age group<=19yrs",
    ],
    [],
    "3.2",
    "1.1",
)

scoring_on_summary = reassign_indicator_numbers(
    scoring_on_summary,
    "dri",
    [
        "% of population (residents) holding a drivers licence",
    ],
    [],
    "1.1",
    "4.3",
)

scoring_on_summary = reassign_indicator_numbers(
    scoring_on_summary,
    "dri",
    [
        "% of 18-19yr age group holding a drivers licence, by gender",
        "% of 18-19yr age group in 2003 holding a drivers licence",
        "share of teenagers with drivers licenses",
        "% of population holding a drivers licence, by gender=female",
        "% of population holding a drivers licence, by gender=male",
        "% of population holding a drivers licence, by age group 20-24",
    ],
    [],
    "1.1",
    "3.2",
)

scoring_on_summary = reassign_indicator_numbers(
    scoring_on_summary,
    "sol",
    [
        "% third party owned systems (income=$100k-$150k)",
        "% third party owned systems (income=$150k-$200k)",
        "% third party owned systems (income=$200k-$250k)",
        "% third party owned systems (income>$250k)",
    ],
    [],
    "1.1",
    "3.2",
)

scoring_on_summary = reassign_indicator_numbers(
    scoring_on_summary,
    "eat",
    [
        "% poultry+pig in total meat consumption",
        "per capita beef consumption",
        "per capita other meat consumption",
        "per capita pig consumption",
        "per capita poultry consumption",
        "per capita sheep & goat consumption",
    ],
    [],
    "1.1",
    "2.5",
)

scoring_on_summary = reassign_indicator_numbers(
    scoring_on_summary,
    "eco",
    [
        "Annual Internet retail (B2C) sales value",
        "Enterprises' total turnover from e-commerce sales (all activities - B2B, B2C, B2G)",
    ],
    [],
    "1.1",
    "2.2",
)

scoring_on_summary = reassign_indicator_numbers(
    scoring_on_summary,
    "org",
    [
        "organic per capita consumption [€/person]",
    ],
    [],
    "1.1",
    "2.2",
)

scoring_on_summary = reassign_indicator_numbers(
    scoring_on_summary,
    "pas",
    [
        "new passive buildings",
    ],
    ["# of new passive buildings"],
    "1.1",
    "2.5",
)

scoring_on_summary = reassign_indicator_numbers(
    scoring_on_summary,
    "pas",
    [
        "new passive buildings",
    ],
    ["new floorspace (m2?)"],
    "1.1",
    "2.9",
)

###################

# Match on codes
summary_df_merged = summary_df.merge(
    scoring_on_summary[list(scoring_on_summary.loc[:, "use_log":].columns)],
    on="code_without_indicator_name",
    how="left",
)

print(f"There are {sum(summary_df_merged['PDF page'].isna())} unmatched rows")

summary_df_merged.drop(columns="code_without_indicator_name").to_csv(
    f"""{PATH}/summary_table_{VERSION}.csv""", float_format="%.5g", index=False
)

# Write unmatched to csv
scoring_on_summary[
    ~(
        scoring_on_summary["code_without_indicator_name"].isin(
            list(summary_df["code_without_indicator_name"].unique())
        )
    )
].to_csv(
    f"""{PATH}/summary_table_original_scoring_not_matched_{VERSION}.csv""",
    float_format="%.5g",
    index=False,
)

# # def clean_text(text):
# #     if isinstance(text, str):  # Ensure it's a string before processing
# #         return re.sub(
# #             r"[^A-Za-z0-9%#]", "", text
# #         ).lower()  # Keep only letters, numbers, % and #
# #     return text  # Return as is if not a string


# # columns_to_clean_for_matching = group_vars
# # cleaned_columns_names = [s + "_clean" for s in columns_to_clean_for_matching]
# # replacement_dict = dict(zip(columns_to_clean_for_matching, cleaned_columns_names))

# # # Replace elements in a that are found in b with corresponding values from c
# # group_vars_clean_names = [
# #     replacement_dict[item] if item in replacement_dict else item for item in group_vars
# # ]

# # summary_df[cleaned_columns_names] = summary_df[columns_to_clean_for_matching].applymap(
# #     lambda x: clean_text(str(x))
# # )
# # scoring_on_summary[cleaned_columns_names] = scoring_on_summary[
# #     columns_to_clean_for_matching
# # ].applymap(lambda x: clean_text(str(x)))

# # # DEBUG
# # summary_df_debug = pd.merge(
# #     summary_df,
# #     scoring_on_summary[
# #         group_vars_clean_names + [" GN scoring ", "GN comments", "CW scoring"]
# #     ],
# #     on=group_vars_clean_names,
# #     how="outer",
# # )
# summary_df_debug.to_csv(
#     f"""{PATH}/summary_table_debug_{VERSION}.csv""", float_format="%.5g", index=False
# )

# summary_df.drop(columns=group_vars_clean_names).to_csv(
#     f"""{PATH}/summary_table_{VERSION}.csv""", float_format="%.5g", index=False
# )

# Perform the left join using the cleaned columns, but keeping summary_df unchanged
# summary_df = pd.merge(
#     summary_df,
#     scoring_on_summary[
#         group_vars_clean_names + [" GN scoring ", "GN comments", "CW scoring"]
#     ],
#     on=group_vars_clean_names,
#     how="left",
# )


# Count the different values
summary_df_split_of_results = (
    summary_df.loc[
        :, summary_df.columns.str.match(r"^C\d+|^suspected_reversal|at_least|use_")
    ]
    .apply(lambda col: col.astype(str).value_counts())
    .fillna(0)
    .astype(int)
)
print(summary_df_split_of_results)

summary_df_split_of_results.to_csv(
    f"""{PATH}/summary_table_{VERSION}_counts.csv""", index=True
)

summary_df["Category_letters"] = summary_df["Category"].str.extract(r"^([A-Za-z]+)")

# Create summary plot
summary_plot_pdf_fn = f"{PATH}/summary_plot_{VERSION}.pdf"
with PdfPages(summary_plot_pdf_fn) as pdf:

    # Group data by letters
    summary_grouped = summary_df.groupby("Category_letters")["slope_log"]

    # Prepare data for box plot
    boxplot_data = [group.dropna().values for _, group in summary_grouped]

    # Create the box plot
    plt.figure(figsize=(8, 6))
    plt.boxplot(boxplot_data, labels=summary_grouped.groups.keys())

    # Customize the plot
    plt.title("Logistic slope parameter b by category")
    plt.xlabel("Category")
    plt.ylabel("b")
    pdf.savefig()
    plt.close()

# Write the criteria to a text file
source_file = __file__  # Replace with your .py file
output_file = f"{PATH}/criteria_as_encoded_{VERSION}.txt"

# Define the target function or code section to extract
start_marker = "CRITERIA_START"  # Adjust to the code section you want to extract
end_marker = "CRITERIA_END"  # Optional: Define an end marker if needed (e.g., next function/class)

# Initialize variables to capture the section
in_target_section = False
extracted_code = []

# Read the source file and extract the target section
with open(source_file, "r") as file:
    for line in file:
        # Detect the start of the target section
        if start_marker in line:
            in_target_section = True

        # Capture lines if within the target section
        if in_target_section:
            extracted_code.append(line)

        # Detect the end of the target section (optional)
        if end_marker and end_marker in line:
            break

# Write the extracted code to a text file
with open(output_file, "w") as file:
    file.writelines(extracted_code)

print(f"{PATH}/Extracted code written to {output_file}")
