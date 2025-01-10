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

VERSION = "v15"
SMALL_SUBSET = True  # Do you only want a small subset for testing?

path = "/mnt/c/Users/simon.destercke/Documents/misc/iiasa/DoSI"
fn_data = f"{path}/Merged_Cleaned_Pitchbook_WebOfScience_GoogleTrends_Data_10Jan_corrected_SDS.xlsx"

adoptions_df = pd.read_excel(
    fn_data, sheet_name="Sheet1", converters={"Indicator Number": str}
)
adoptions_df["Value"] = pd.to_numeric(adoptions_df["Value"], errors="coerce")
adoptions_df = adoptions_df.dropna(subset=["Value"])

# Correct for trailing spaces in the data
adoptions_df["Spatial Scale"] = adoptions_df["Spatial Scale"].str.rstrip()
adoptions_df["Innovation Name"] = adoptions_df["Innovation Name"].str.rstrip()

# adoptions_df.loc[
#     adoptions_df["Innovation Name"] == "drivers licence", "Innovation Name"
# ] = "drivers license"

# For debugging: only 2 innovation names
if SMALL_SUBSET:
    adoptions_df = adoptions_df[
        adoptions_df["Innovation Name"].isin(["car sharing"])
        # adoptions_df["Indicator Number"].isin(["3.3", "3.5", "4.1"])
        # adoptions_df["Indicator Number"].isin(["1.1"])
    ]

print(adoptions_df)

fn_metadata = f"{path}/metadata labels 5Dec_SDS_update20250110.xlsx"


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
metadata["Description"] = {
    val: f"d{idx}"
    for idx, val in enumerate(sorted(adoptions_df["Description"].unique()), start=1)
}
# If metadata file is not in sync with the data table, remake the description dictionary
metadata["Metric"] = {
    val: f"m{idx}"
    for idx, val in enumerate(sorted(adoptions_df["Metric"].unique()), start=1)
}


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
        Dt_initial_guess=np.log(81) / slope * max(group["Value"]),
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


results_logistic = (
    adoptions_df.groupby(group_vars).apply(apply_FPLogFit_with_scaling).reset_index()
)
results_exponential = (
    adoptions_df.groupby(group_vars).apply(apply_exponential_fit).reset_index()
)
results_linear = adoptions_df.groupby(group_vars).apply(apply_linear_fit).reset_index()

print(results_logistic)
print(
    f"""{results_logistic["t0"].isnull().sum()} out of {len(results_logistic)} logistic fits failed"""
)
print(results_exponential)
print(
    f"""{results_exponential["a"].isnull().sum()} out of {len(results_exponential)} exponential fits failed"""
)

## Scatterplots

LINE_X_BUFFER = 10

DISTANCE_TO_MEAN_THRESHOLD = 0.48
AUTOCORRELATION_THRESHOLD = 0.75

summary_table_rows = (
    []
)  # list to which to append the summary statistics for each series

# Initialize PDF
plots_pdf_fn = f"{path}/scatterplots_{VERSION}.pdf"
with PdfPages(plots_pdf_fn) as pdf:
    # Group the data
    grouped = adoptions_df.groupby(group_vars)

    # Loop through each group and create a scatterplot
    for i in range(len(grouped)):

        print(i)

        sorted_index = sorted_indices[i]

        group_name, group_data = list(grouped)[sorted_index]

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
        r2_log, r2adj_log = calculate_adjusted_r2(
            group_data["Value"], y_pred, n_params=3
        )
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
        r2_exp, r2adj_exp = calculate_adjusted_r2(
            group_data["Value"], y_pred, n_params=2
        )
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
        r2_lin, r2adj_lin = calculate_adjusted_r2(
            group_data["Value"], y_pred, n_params=2
        )
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
        pdf.savefig()  # Save current figure into the PDF
        plt.close()  # Close the figure to free memory

        # Create the row for the summary table

        summary_table_rows.append(
            {
                "Code": code,
                **{
                    f"{value}": group_name[i] for i, value in enumerate(group_vars)
                },  # Dynamically add group_vars
                "Category": categories[
                    group_name[0].lower()
                ],  # Lookup category dynamically
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
                "n_data_points_beyond_min": n_data_points_beyond_min,
                "suspected_reversal_up2down": int(
                    relative_distance_max_to_avg_year < DISTANCE_TO_MEAN_THRESHOLD
                ),
                "suspected_reversal_down2up": int(
                    relative_distance_min_to_avg_year < DISTANCE_TO_MEAN_THRESHOLD
                ),
                "at_least_one_big_jump": (
                    int((max(relative_changes) > 2) | (min(relative_changes) < 0.5))
                    if len(relative_changes) > 0
                    else None
                ),
                f"volatility_autocorrelation_lag1_threshold_{AUTOCORRELATION_THRESHOLD:.2g}": (
                    int(autocorr[1] < AUTOCORRELATION_THRESHOLD)
                    if len(relative_changes) > 0
                    else None
                ),
                "all_values_less_than_or_equal_to_1": int(max_value <= 1),
                "all_values_less_than_or_equal_to_100": int(max_value <= 100),
            }
        )

print(f"Scatterplots saved to {plots_pdf_fn}")

summary_df = pd.DataFrame(summary_table_rows)
index_of_first_numeric_column = (
    len(group_vars) + 2
)  # DEPENDS ON THE DICTIONARY! IF e.g. ORDER CHANGES, THEN CHANGE THIS!
summary_df.iloc[:, index_of_first_numeric_column:] = summary_df.iloc[
    :, index_of_first_numeric_column:
].astype(float)

summary_df.to_csv(
    f"""{path}/summary_table_{VERSION}.csv""", float_format="%.5g", index=False
)

summary_df["Category_letters"] = summary_df["Category"].str.extract(r"^([A-Za-z]+)")

# Create summary plot
summary_plot_pdf_fn = f"{path}/summary_plot_{VERSION}.pdf"
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
