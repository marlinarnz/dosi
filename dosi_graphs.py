# File begun on 26 November 2024

import pandas as pd
import numpy as np

import re
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.backends.backend_pdf import PdfPages

VERSION = "v20"
VERSION_FOR_FITS = "v15"
VERSION_FOR_METADATA = "v19"
SMALL_SUBSET = True  # Do you only want a small subset for testing?
REDO_FITS = False
RENUMBER_METADATA_CODES = False
APPLY_TRANSFORMATIONS_TO_DATA_FILE = True  # Should transformations such as cumulation be applied (True), or not (False)? This is important because otherwise there will be doubles

PATH = "/mnt/c/Users/simon.destercke/Documents/misc/iiasa/DoSI"
fn_data = f"{PATH}/Merged_Cleaned_Pitchbook_WebOfScience_GoogleTrends_Data_10Jan_corrected_SDS.xlsx"

adoptions_df = pd.read_excel(
    fn_data, sheet_name="Sheet1", converters={"Indicator Number": str}
)
adoptions_df["Value"] = pd.to_numeric(adoptions_df["Value"], errors="coerce")
adoptions_df = adoptions_df.dropna(subset=["Value"])

# Correct for trailing spaces in the data
adoptions_df["Spatial Scale"] = adoptions_df["Spatial Scale"].str.rstrip()
adoptions_df["Innovation Name"] = adoptions_df["Innovation Name"].str.rstrip()

# For debugging: only subset
if SMALL_SUBSET:
    adoptions_df = adoptions_df[
        adoptions_df["Innovation Name"].isin(["car sharing"])
        # adoptions_df["Indicator Number"].isin(["3.3", "3.5", "4.1"])
        # adoptions_df["Indicator Number"].isin(["1.1"])
    ]

print(adoptions_df)

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


# Read codes that need to be cumulated
with open(f"{PATH}/datasets_to_cumulate.txt", "r") as file:
    codes_to_cumulate = [line.strip() for line in file]
# Loop through codes to cumulate
cumulated_dfs = []
for code_to_cumulate in codes_to_cumulate:
    print(code_to_cumulate)
    try:
        name_data_to_cumulate, df_to_cumulate = grouped_as_list[
            sorted_indices[codes.index(code_to_cumulate)]
        ]
    except ValueError as e:
        print(f"Code {code_to_cumulate} not found!")


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
        # Initial guesses for the parameters
        initial_guess = [np.median(x), Dt_initial_guess, np.max(y)]

        # Fit the logistic function
        try:
            params, _ = curve_fit(
                FPLogValue_with_scaling,
                x,
                y,
                p0=initial_guess,
                method="trf",
                maxfev=5000,
            )
            t0, Dt, k = params
        except RuntimeError:
            print("RuntimeError")
            return {"t0": None, "Dt": None, "K": None}  # Handle fitting failure
        # if t0 + Dt < 2000 and firstrun:
        #     return FPLogFit_with_scaling(
        #         x, y, Dt_initial_guess=-5, firstrun=False
        #     )  # Only one additional attempt
        # else:
        return {"t0": t0, "Dt": Dt, "K": k}


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

pdf_commondb = PdfPages(f"{PATH}/scatterplots_{VERSION}_COMMON.pdf")
pdf_other = PdfPages(f"{PATH}/scatterplots_{VERSION}_OTHER.pdf")

adjusted_dfs = []  # For storing the adjusted data frames

# Loop through each group and create a scatterplot
for i in range(len(grouped)):

    print(i)

    sorted_index = sorted_indices[i]

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
    plt.scatter(
        group_data["Year"],
        group_data["Value"],
        label="Data Points",
        color="black",
        s=40,
    )

    x_line = np.arange(
        min(group_data["Year"]) - LINE_X_BUFFER,
        max(group_data["Year"]) + LINE_X_BUFFER,
        0.1,
    )

    # Logistic fit line
    line_color_log = "blue"
    results_filtered = results_logistic[
        results_logistic[group_vars].apply(tuple, axis=1).isin([group_name])
    ]
    t0 = results_filtered["t0"].values[0]
    Dt = results_filtered["Dt"].values[0]
    k = results_filtered["K"].values[0]
    y_line_log = FPLogValue_with_scaling(x_line, t0, Dt, k)
    y_pred = FPLogValue_with_scaling(group_data["Year"], t0, Dt, k)
    r2_log, r2adj_log = calculate_adjusted_r2(group_data["Value"], y_pred, n_params=3)
    rmse_log = np.sqrt(np.mean((group_data["Value"] - y_pred) ** 2))
    mae_log = np.mean(np.abs(group_data["Value"] - y_pred))
    plt.plot(
        x_line, y_line_log, color=line_color_log, label="Logistic"
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
    plt.plot(x_line, y_line_exp, color=line_color_exp, label="Exponential")

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
    plt.plot(x_line, y_line_lin, color=line_color_lin, label="Linear")

    code = codes[sorted_index]

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
        [line_color_log, line_color_log] + ["black"] * (len(column_labels) - 2),
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
    plt.grid(True)

    # Save the current plot to the PDF
    if group_name[2] in COMMON_DATABASES_INDICATOR_CODES:
        pdf_commondb.savefig()  # Save current figure into the PDF
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

    # Now write to the new data file

    ALSO_ADD_ORIGINAL = True
    OVERLAP = False

    if APPLY_TRANSFORMATIONS_TO_DATA_FILE:
        if (
            (summary_table_dict["all_values_less_than_or_equal_to_1"] == 0)
            & (summary_table_dict["all_values_less_than_or_equal_to_100"] == 1)
            & (
                bool(
                    re.search(
                        pattern=r"(?i)percent|(?<!\d)%",
                        string=group_name[group_vars.index("Metric")],
                    )
                )
            )
        ):  # Condition for 'standardizing'
            adjusted_df = group_data
            adjusted_df["Value"] = adjusted_df["Value"] / 100
            adjusted_dfs.append(adjusted_df)
            ALSO_ADD_ORIGINAL = False

        if (group_name[group_vars.index("Indicator Number")] == "3.5") | (
            (group_name[group_vars.index("Indicator Number")] == "1.1")
            & (
                group_name[group_vars.index("Innovation Name")]
                in ["climate protest", "passive building retrofits"]
            )
        ):  # condition for cumulating
            adjusted_df = group_data
            adjusted_df["Value"] = adjusted_df["Value"].cumsum()
            description_new = "cumulative " + str(
                adjusted_df["Description"].unique()[0]
            )
            adjusted_df["Description"] = description_new
            metric_new = "cumulative " + str(adjusted_df["Metric"].unique()[0])
            adjusted_df["Metric"] = metric_new
            # Now also update dictionaries
            description_counter += 1
            metadata["Description"][description_new] = convert_to_three_digit_notation(
                f"d{description_counter}"
            )
            metric_counter += 1
            metadata["Metric"][metric_new] = convert_to_three_digit_notation(
                f"m{metric_counter}"
            )
            adjusted_dfs.append(adjusted_df)

        if (
            group_name[group_vars.index("Indicator Number")] in ["4.1", "3.5", "4.2"]
        ) | (
            (group_name[group_vars.index("Indicator Number")] == "1.1")
            & (
                group_name[group_vars.index("Innovation Name")]
                in ["eating less meat", "microfinance", "solar leasing"]
            )
        ):  # Time series to be partialized up to maximum
            adjusted_df = group_data
            max_index = adjusted_df["Value"].idxmax()  # index of maximum
            adjusted_df = adjusted_df.loc[:max_index]  # truncate
            description_new = "Partial up to max " + str(
                adjusted_df["Description"].unique()[0]
            )
            adjusted_df["Description"] = description_new
            metric_new = "Partial up to max " + str(adjusted_df["Metric"].unique()[0])
            adjusted_df["Metric"] = metric_new
            # Now also update dictionaries
            description_counter += 1
            metadata["Description"][description_new] = convert_to_three_digit_notation(
                f"d{description_counter}"
            )
            metric_counter += 1
            metadata["Metric"][metric_new] = convert_to_three_digit_notation(
                f"m{metric_counter}"
            )
            adjusted_dfs.append(adjusted_df)

        if (group_name[group_vars.index("Indicator Number")] == "1.1") & (
            group_name[group_vars.index("Innovation Name")] in ["teleworking"]
        ):  # Time series to be partialized up to 2019
            adjusted_df = group_data
            index_2019 = adjusted_df.index[
                adjusted_df["Year"] == 2019
            ].item()  # index of maximum
            adjusted_df = adjusted_df.loc[:index_2019]  # truncate
            description_new = "Partial up to 2019 " + str(
                adjusted_df["Description"].unique()[0]
            )
            adjusted_df["Description"] = description_new
            metric_new = "Partial up to 2019 " + str(adjusted_df["Metric"].unique()[0])
            adjusted_df["Metric"] = metric_new
            # Now also update dictionaries
            description_counter += 1
            metadata["Description"][description_new] = convert_to_three_digit_notation(
                f"d{description_counter}"
            )
            metric_counter += 1
            metadata["Metric"][metric_new] = convert_to_three_digit_notation(
                f"m{metric_counter}"
            )
            adjusted_dfs.append(adjusted_df)

        if (group_name[group_vars.index("Indicator Number")] == "1.1") & (
            group_name[group_vars.index("Innovation Name")]
            in ["microfinance", "eating less meat"]
        ):  # Time series to be partialized up to minimum
            adjusted_df = group_data
            min_index = adjusted_df["Value"].idxmax()  # index of maximum
            adjusted_df = adjusted_df.loc[:min_index]  # truncate
            description_new = "Partial up to min " + str(
                adjusted_df["Description"].unique()[0]
            )
            adjusted_df["Description"] = description_new
            metric_new = "Partial up to min " + str(adjusted_df["Metric"].unique()[0])
            adjusted_df["Metric"] = metric_new
            # Now also update dictionaries
            description_counter += 1
            metadata["Description"][description_new] = convert_to_three_digit_notation(
                f"d{description_counter}"
            )
            metric_counter += 1
            metadata["Metric"][metric_new] = convert_to_three_digit_notation(
                f"m{metric_counter}"
            )
            adjusted_dfs.append(adjusted_df)

    if ALSO_ADD_ORIGINAL:
        adjusted_dfs.append(group_data)


pdf_commondb.close()
pdf_other.close()
print(f"Scatterplots version {VERSION} saved to pdf.")

dfs = []
for key in ["Description", "Metric"]:
    # Create a DataFrame for the current dictionary
    df = pd.DataFrame(list(metadata[key].items()), columns=[key, "code"])
    # Add an empty column
    df[""] = ""
    dfs.append(df)
# Concatenate all DataFrames side-by-side
final_df = pd.concat(dfs, axis=1)
# Store the dictionaries with the number codes
metadata_new_fn = f"{PATH}/metadata_numbercodes_{VERSION}.xlsx"
# Save the result to an Excel file
final_df.to_excel(metadata_new_fn, index=False)
print(f"Updated metadata number codes successfully written to {metadata_new_fn}")

pd.concat(adjusted_dfs).to_csv(
    f"""{PATH}/adjusted_datasets_{VERSION}.csv""", index=False
)

summary_df = pd.DataFrame(summary_table_rows)

summary_df.to_csv(
    f"""{PATH}/summary_table_{VERSION}.csv""", float_format="%.5g", index=False
)

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
