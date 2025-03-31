import pandas as pd
import numpy as np

import re
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit, differential_evolution, minimize
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.backends.backend_pdf import PdfPages
import random

VERSION = "v-1"
VERSION_FOR_FITS = "v24"
VERSION_FOR_METADATA = "v23"
VERSION_FOR_DATA = "v23"
SMALL_SUBSET = False  # Do you only want a small subset for testing?
REDO_FITS = False
RENUMBER_METADATA_CODES = False

LINE_X_BUFFER = 10
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
        adoptions_df["Innovation Name"].isin(["car sharing"])
        # adoptions_df["Indicator Number"].isin(["3.3", "3.5", "4.1"])
        # adoptions_df["Indicator Number"].isin(["1.1"])
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


# Better logistic fits for problem data
problem_fits_codes = [
    "low_ger_1.1Ado_d329_m185",
    "org_uki_1.1Ado_d163_m140",
    "mic_ban_1.1Ado_d285_m173",
    "dig_den_1.1Ado_d150_m037",
]

problem_fits_data = list(
    grouped_as_list[j]
    for j in range(len(codes))
    if codes[j] in problem_fits_codes
    # [
    #     sorted_indices[i] for i in range(len(codes)) if codes[i] in problem_fits_codes
    # ]
)


def FPLogValue_with_scaling(x, t0, Dt, s):
    """
    Logistic function with vertical scaling.
    """
    return s / (1 + np.exp(-np.log(81) * (x - t0) / Dt))


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


sorted_index = 0

group_name, group_data = problem_fits_data[sorted_index]

group_data.sort_values(by=["Year"], inplace=True)

group_data.reset_index(drop=True, inplace=True)


# Global Optimization (differential evolution) - from ChatGPT


# Define an objective (cost) function
def objective(params, t, y):
    t0, Dt, s = params
    y_pred = FPLogValue_with_scaling(t, t0, Dt, s)
    return np.sum((y - y_pred) ** 2)


# Example usage
# Suppose t_data and y_data are your time points and observations.

# Bounds for A, B, r (adjust as appropriate)
bounds = [
    (1900, 2100),  # t0
    (0.1, 500),  # Dt
    (1e-2, 1000),  # s or K (asymptote)
]

# result = differential_evolution(
#     objective,
#     bounds=bounds,
#     args=(group_data["Year"], group_data["Value"]),
#     maxiter=1000,  # You can increase from 1000 if needed
#     seed=42,
# )

# best_params = result.x
# print("Best params (t0, Dt, K):", best_params)
# print("Objective function value:", result.fun)


def multi_start_fit(n_starts=10):
    best_val = None
    best_params = None

    for _ in range(n_starts):
        # Generate random initial guesses in a sensible range
        init_A = random.uniform(0, 3000)
        init_B = random.uniform(1e-6, 1000)
        init_r = random.uniform(1e-6, 10**20)
        init_guess = [init_A, init_B, init_r]

        res = minimize(
            objective,
            init_guess,
            args=(group_data["Year"], group_data["Value"]),
            method="BFGS",
        )  # or "Nelder-Mead", etc.

        if best_val is None or res.fun < best_val:
            best_val = res.fun
            best_params = res.x

    return best_params, best_val


params, val = multi_start_fit(n_starts=100)
print("Best (A, B, r):", params)
print("Best objective:", val)

# Plot
x_line = np.arange(
    min(group_data["Year"]) - LINE_X_BUFFER,
    max(group_data["Year"]) + LINE_X_BUFFER,
    0.1,
)

fig = plt.figure(figsize=(12, 9), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1)
plt.scatter(
    group_data["Year"],
    group_data["Value"],
    label="Data Points",
    color="black",
    s=40,
)

# t0 = result["x"][0]
# Dt = result["x"][1]
# k = result["x"][2]
t0 = params[0]
Dt = params[1]
k = params[2]
y_line_log = FPLogValue_with_scaling(x_line, t0, Dt, k)
y_pred = FPLogValue_with_scaling(group_data["Year"], t0, Dt, k)
r2_log, r2adj_log = calculate_adjusted_r2(group_data["Value"], y_pred, n_params=3)
rmse_log = np.sqrt(np.mean((group_data["Value"] - y_pred) ** 2))
mae_log = np.mean(np.abs(group_data["Value"] - y_pred))
plt.plot(x_line, y_line_log, color=LINE_COLOR_LOG, label="Logistic")  # , marker="+")

# Find max y for plotting
lines = ax.get_lines()

# Find the maximum y-value
max_y_lines = max(max(line.get_ydata()) for line in lines)

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

plt.show()

input("...")

plt.close()
