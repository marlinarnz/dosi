# File begun on 26 November 2024

# Task: Could you script the first data analysis task? This could be done already on the attached.
# - plot each data series as a standard line graph (metric on y-axis, time on x-axis; graph title with innovation name + indicator number) [*1]
# - fit logistic, exponential, linear models to data series (and superimpose plots on data series graph)
# - report model parameters & fit statistics in inset box on graph [**2]

# [*1] we’ll come up with a short label as a unique identifier of each time series using meta data: innovation name, indicator number + name + description, metric,
# [**2] some of these fits will be junk; combination of visual check + fit statistics (e.g. R2) will help us select best function

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

path = "/mnt/c/Users/simon.destercke/Documents/misc/iiasa/DoSI"
fn_data = f"{path}/Trial_Data_Results.xlsx"

adoptions_df = pd.read_excel(
    fn_data, sheet_name="Sheet1", converters={"Indicator Number": str}
)
adoptions_df["Value"] = pd.to_numeric(adoptions_df["Value"], errors="coerce")
adoptions_df = adoptions_df.dropna(subset=["Value"])

print(adoptions_df)

## Formula for logistic fit
# 1/(1+exp(-log(81,base=exp(1))/deltaT*(Year-t0))


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


def FPLogFit_with_scaling(x, y):
    """
    Fit a logistic function with vertical scaling to the data.

    Parameters:
    x (array-like): Independent variable.
    y (array-like): Dependent variable.

    Returns:
    dict: Fitted parameters t0, Dt, and S.
    """

    if len(x) < 3:
        return {"t0": 2300, "Dt": np.log(81) / 10, "S": 1.0}  # Default parameters
    else:
        # Initial guesses for the parameters
        initial_guess = [np.median(x), 10, np.max(y)]

        # Fit the logistic function
        try:
            params, _ = curve_fit(
                FPLogValue_with_scaling,
                x,
                y,
                p0=initial_guess,
                method="trf",
                maxfev=1000,
            )
            t0, Dt, s = params
        except RuntimeError:
            print("RuntimeError")
            return {"t0": None, "Dt": None, "S": None}  # Handle fitting failure

        return {"t0": t0, "Dt": Dt, "S": s}


# Group by variables
group_vars = [
    "Innovation Name",
    "Indicator Number",
    "Indicator Name",
    "Description",
    "Metric",
    "Sheet",
]


# Apply FPLogfit to each group
def apply_FPLogFit_with_scaling(group):
    print(group)
    result = FPLogFit_with_scaling(group["Year"], group["Value"])
    return pd.Series(result)


results = (
    adoptions_df.groupby(group_vars).apply(apply_FPLogFit_with_scaling).reset_index()
)

print(results)

## Scatterplots

line_x_buffer = 20

# Initialize PDF
pdf_file = f"{path}/scatterplots.pdf"
with PdfPages(pdf_file) as pdf:
    # Group the data
    grouped = adoptions_df.groupby(group_vars)

    # Loop through each group and create a scatterplot
    for group_name, group_data in grouped:
        plt.figure(figsize=(12, 8))
        plt.scatter(
            group_data["Year"], group_data["Value"], label="Data Points", color="black"
        )

        x_line = np.arange(
            min(group_data["Year"]) - line_x_buffer,
            max(group_data["Year"]) + line_x_buffer,
            1,
        )

        # Logarithmic Substitution line
        line_color_log = "blue"
        results_filtered = results[
            results[group_vars].apply(tuple, axis=1).isin([group_name])
        ]
        t0 = results_filtered["t0"].values[0]
        Dt = results_filtered["Dt"].values[0]
        s = results_filtered["S"].values[0]
        y_line_log = FPLogValue_with_scaling(x_line, t0, Dt, s)
        plt.plot(x_line, y_line_log, color=line_color_log, label="Log Substitution")

        # Box with parameters
        props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray")
        plt.text(
            0.95,
            0.95,
            f"""Log Substitutions t0={t0:.3g}, Dt={Dt:.3g}, S={s:.3g}""",
            transform=plt.gca().transAxes,
            fontsize=10,
            color=line_color_log,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )

        plt.title(f"Scatterplot for Group: {group_name}")
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.ylim(bottom=0)
        plt.grid(True)

        # Save the current plot to the PDF
        pdf.savefig()  # Save current figure into the PDF
        plt.close()  # Close the figure to free memory

print(f"Scatterplots saved to {pdf_file}")
