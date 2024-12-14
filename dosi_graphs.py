# File begun on 26 November 2024

# Task: Could you script the first data analysis task? This could be done already on the attached.
# - plot each data series as a standard line graph (metric on y-axis, time on x-axis; graph title with innovation name + indicator number) [*1]
# - fit logistic, exponential, linear models to data series (and superimpose plots on data series graph)
# - report model parameters & fit statistics in inset box on graph [**2]

# [*1] we’ll come up with a short label as a unique identifier of each time series using meta data: innovation name, indicator number + name + description, metric,
# [**2] some of these fits will be junk; combination of visual check + fit statistics (e.g. R2) will help us select best function

import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.backends.backend_pdf import PdfPages

VERSION = "v10"
SMALL_SUBSET = True  # Do you only want a small subset for testing?

path = "/mnt/c/Users/simon.destercke/Documents/misc/iiasa/DoSI"
fn_data = f"{path}/Merged_Cleaned_Pitchbook_WebOfScience_Data.xlsx"

adoptions_df = pd.read_excel(
    fn_data, sheet_name="Sheet1", converters={"Indicator Number": str}
)
adoptions_df["Value"] = pd.to_numeric(adoptions_df["Value"], errors="coerce")
adoptions_df = adoptions_df.dropna(subset=["Value"])

adoptions_df.loc[
    adoptions_df["Innovation Name"] == "drivers licence", "Innovation Name"
] = "drivers license"

# For debugging: only 2 innovation names
if SMALL_SUBSET:
    adoptions_df = adoptions_df[
        adoptions_df["Innovation Name"].isin(["Quitting smoking", "car sharing"])
    ]

print(adoptions_df)

fn_metadata = f"{path}/metadata labels 5Dec_SDS.xlsx"


def convert_to_three_digit_notation(s):
    return re.sub(r"([a-zA-Z])(\d+)", lambda m: f"{m.group(1)}{int(m.group(2)):03}", s)


def read_metadata_table(fn, columns):
    df = pd.read_excel(fn, usecols=columns, dtype=str).dropna().reset_index(drop=True)
    df.iloc[:, 1] = df.iloc[:, 1].apply(convert_to_three_digit_notation)
    return df.set_index(df.columns[0])[df.columns[1]].to_dict()


categories = read_metadata_table(fn_metadata, "A,B")

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


def FPLogFit_with_scaling(x, y, Dt_initial_guess: float = 10, firstrun: bool = True):
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
                maxfev=2000,
            )
            t0, Dt, k = params
        except RuntimeError:
            print("RuntimeError")
            return {"t0": None, "Dt": None, "K": None}  # Handle fitting failure
        if t0 + Dt < 2000 and firstrun:
            return FPLogFit_with_scaling(
                x, y, Dt_initial_guess=-5, firstrun=False
            )  # Only one additional attempt
        else:
            return {"t0": t0, "Dt": Dt, "K": k}


# Define the exponential function
def exponential_func(x, a, b, c):
    return a * np.exp(b * (x - c))


# Apply FPLogfit to each group
def apply_FPLogFit_with_scaling(group):
    result = FPLogFit_with_scaling(group["Year"], group["Value"])
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
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    n = len(y_obs)
    r2_adj = 1 - ((1 - r2) * (n - 1) / (n - n_params - 1))
    return r2, r2_adj


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

line_x_buffer = 10

# Initialize PDF
pdf_file = f"{path}/scatterplots_{VERSION}.pdf"
with PdfPages(pdf_file) as pdf:
    # # Group the data
    grouped = adoptions_df.groupby(group_vars)

    # Loop through each group and create a scatterplot
    for i in range(len(grouped)):

        sorted_index = sorted_indices[i]

        group_name, group_data = list(grouped)[sorted_index]

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
            min(group_data["Year"]) - line_x_buffer,
            max(group_data["Year"]) + line_x_buffer,
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
        # table_colors = [
        #     ["black"] * len(column_labels),
        #     [line_color_log] * len(column_labels),
        #     [line_color_exp] * len(column_labels),
        #     [line_color_lin] * len(column_labels),
        # ]

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

        title_list = list(group_name)
        title_list[2:4] = [title_list[2] + " " + title_list[3]]

        plt.title("\n".join(title_list + [code]), loc="left")
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.ylim(bottom=0, top=min(max(group_data["Value"]) * 2, max_y_lines))
        plt.grid(True)

        # Save the current plot to the PDF
        pdf.savefig()  # Save current figure into the PDF
        plt.close()  # Close the figure to free memory

print(f"Scatterplots saved to {pdf_file}")
