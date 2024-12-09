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

path = "/mnt/c/Users/simon.destercke/Documents/misc/iiasa/DoSI"
fn_data = f"{path}/Cleaned_File_Data_Results 5Dec.xlsx"

adoptions_df = pd.read_excel(
    fn_data, sheet_name="Sheet1", converters={"Indicator Number": str}
)
adoptions_df["Value"] = pd.to_numeric(adoptions_df["Value"], errors="coerce")
adoptions_df = adoptions_df.dropna(subset=["Value"])

adoptions_df.loc[
    adoptions_df["Innovation Name"] == "drivers licence", "Innovation Name"
] = "drivers license"

# For debugging: only 2 innovation names
if True:
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

for key, nested_dict in metadata.items():
    if isinstance(nested_dict, dict):  # Ensure the value is a dictionary
        metadata[key] = {
            k.lower() if isinstance(k, str) else k: v for k, v in nested_dict.items()
        }

group_vars = list(metadata.keys()) + ["Indicator Name"]

# Failed attempt below to go for alphabetic ordering of codes
grouped = adoptions_df.groupby(group_vars)

code = []
groups = []

# Loop through each group and create the code
for group_name, group_data in grouped:
    groups.append(group_name)
    code.append(
        "_".join(
            [
                metadata[group_vars[i]][group_name[i].lower()]
                for i in range(len(metadata))
            ]
        )
    )

# sorted_groups = [t for _, t in sorted(zip(code, groups), key=lambda x: x[0])]
# grouped = pd.concat([grouped.get_group(key) for key in sorted_groups])


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
        return {"t0": 2300, "Dt": 2000, "S": 1.0}  # Default parameters
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
            t0, Dt, s = params
        except RuntimeError:
            print("RuntimeError")
            return {"t0": None, "Dt": None, "S": None}  # Handle fitting failure
        if t0 + Dt < 2000 and firstrun:
            return FPLogFit_with_scaling(
                x, y, Dt_initial_guess=-5, firstrun=False
            )  # Only one additional attempt
        else:
            return {"t0": t0, "Dt": Dt, "S": s}


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
pdf_file = f"{path}/scatterplots_v4.pdf"
with PdfPages(pdf_file) as pdf:
    # # Group the data
    grouped = adoptions_df.groupby(group_vars)

    # Loop through each group and create a scatterplot
    for group_name, group_data in grouped:
        fig = plt.figure(figsize=(12, 9), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(
            group_data["Year"],
            group_data["Value"],
            label="Data Points",
            color="black",
            s=20,
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
        s = results_filtered["S"].values[0]
        y_line_log = FPLogValue_with_scaling(x_line, t0, Dt, s)
        y_pred = FPLogValue_with_scaling(group_data["Year"], t0, Dt, s)
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
        rmse_lin = np.sqrt(np.mean((group_data["Value"] - y_pred) ** 2))
        mae_lin = np.mean(np.abs(group_data["Value"] - y_pred))
        plt.plot(x_line, y_line_lin, color=line_color_lin, label="Linear")

        # Box with parameters
        props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray")

        # Logistic
        plt.text(
            0.95,
            0.95,
            f"""Logistic t0={t0:.0f}, Dt={Dt:.3g}, S={s:.3g} - RMSE = {rmse_log:.3g} - MAE = {mae_log:.3g}""",
            transform=plt.gca().transAxes,
            fontsize=10,
            color=line_color_log,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )

        plt.text(
            0.95,
            0.90,
            f"""Exponential {a:.3g}*exp({b:.3g}*(x-{c:.0f}) - RMSE = {rmse_exp:.3g} - MAE = {mae_exp:.3g}""",
            transform=plt.gca().transAxes,
            fontsize=10,
            color=line_color_exp,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )
        plt.text(
            0.95,
            0.85,
            f"""Linear slope={slope:.3g}, intercept={intercept:.3g} - RMSE = {rmse_lin:.3g} - MAE = {mae_lin:.3g}""",
            transform=plt.gca().transAxes,
            fontsize=10,
            color=line_color_lin,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )

        code = "_".join(
            [
                metadata[group_vars[i]][group_name[i].lower()]
                for i in range(len(metadata))
            ]
        )

        # Try table on top:

        # Add a table
        table_data = [["A", "B", "C"], ["1", "2", "3"], ["4", "5", "6"]]
        table_colors = [
            ["red", "green", "blue"],
            ["blue", "red", "green"],
            ["green", "blue", "red"],
        ]

        # Create a table object
        table = Table(ax, bbox=[0.5, 0.5, 0.4, 0.3])  # Adjust bbox for positioning
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
                    loc="center",
                    facecolor="white",
                    edgecolor="black",
                )
                cell.set_text_props(color=cell_color)  # Set text color for the cell

        # Add the table to the axes
        ax.add_table(table)

        # Find max y for plotting
        lines = ax.get_lines()

        # Find the maximum y-value
        max_y_lines = max(max(line.get_ydata()) for line in lines)

        plt.title("\n".join(list(group_name) + [code]))
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.ylim(bottom=0, top=min(max(group_data["Value"]) * 2, max_y_lines))
        plt.grid(True)

        # Save the current plot to the PDF
        pdf.savefig()  # Save current figure into the PDF
        plt.close()  # Close the figure to free memory

print(f"Scatterplots saved to {pdf_file}")
