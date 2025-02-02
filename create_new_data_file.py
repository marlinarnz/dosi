import pandas as pd
import numpy as np

import re
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.backends.backend_pdf import PdfPages

VERSION = "v21"
VERSION_FOR_FITS = "v15"
VERSION_FOR_METADATA = "v19"
SMALL_SUBSET = False  # Do you only want a small subset for testing?
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

adjusted_dfs = []  # For storing the adjusted data frames

counter_scaling = 0
counter_added_metric = 0
counter_added_description = 0

# Loop through each group and create new datasets if needed
for i in range(len(grouped)):

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

    all_values_less_than_or_equal_to_1 = int(max_value <= 1)
    all_values_less_than_or_equal_to_100 = int(max_value <= 100)

    # Apply transformations
    if (
        (all_values_less_than_or_equal_to_1 == 0)
        & (all_values_less_than_or_equal_to_100 == 1)
        & (
            bool(
                re.search(
                    pattern=r"(?i)percent|(?<!\d)%",
                    string=group_name[group_vars.index("Metric")],
                )
            )
        )
    ):
        adjusted_df = group_data.copy()
        adjusted_df["Value"] = adjusted_df["Value"] / 100
        adjusted_dfs.append(adjusted_df)
        print(f"Rescaled {group_name}")
        counter_scaling += 1
    else:
        adjusted_dfs.append(group_data)

    if (group_name[group_vars.index("Indicator Number")] == "3.5") | (
        (group_name[group_vars.index("Indicator Number")] == "1.1")
        & (
            group_name[group_vars.index("Innovation Name")]
            in ["climate protest", "passive building retrofits"]
        )
    ):  # condition for cumulating
        adjusted_df = group_data.copy()
        adjusted_df["Value"] = adjusted_df["Value"].cumsum()
        description_new = "cumulative " + str(adjusted_df["Description"].unique()[0])
        adjusted_df["Description"] = description_new
        metric_new = "cumulative " + str(adjusted_df["Metric"].unique()[0])
        adjusted_df["Metric"] = metric_new
        # Now also update dictionaries
        if description_new not in metadata["Description"]:
            description_counter += 1
            metadata["Description"][description_new] = convert_to_three_digit_notation(
                f"d{description_counter}"
            )
            counter_added_description += 1
        if metric_new not in metadata["Metric"]:
            metric_counter += 1
            metadata["Metric"][metric_new] = convert_to_three_digit_notation(
                f"m{metric_counter}"
            )
            counter_added_metric += 1
        adjusted_dfs.append(adjusted_df)

    if (group_name[group_vars.index("Indicator Number")] in ["4.1", "3.5", "4.2"]) | (
        (group_name[group_vars.index("Indicator Number")] == "1.1")
        & (
            group_name[group_vars.index("Innovation Name")]
            in ["eating less meat", "microfinance", "solar leasing"]
        )
    ):  # Time series to be partialized up to maximum
        adjusted_df = group_data.copy()
        max_index = adjusted_df["Value"].idxmax()  # index of maximum
        adjusted_df = adjusted_df.loc[:max_index]  # truncate
        description_new = "Partial up to max " + str(
            adjusted_df["Description"].unique()[0]
        )
        adjusted_df["Description"] = description_new
        metric_new = "Partial up to max " + str(adjusted_df["Metric"].unique()[0])
        adjusted_df["Metric"] = metric_new
        # Now also update dictionaries
        if description_new not in metadata["Description"]:
            description_counter += 1
            metadata["Description"][description_new] = convert_to_three_digit_notation(
                f"d{description_counter}"
            )
            counter_added_description += 1
        if metric_new not in metadata["Metric"]:
            metric_counter += 1
            metadata["Metric"][metric_new] = convert_to_three_digit_notation(
                f"m{metric_counter}"
            )
            counter_added_metric += 1
        adjusted_dfs.append(adjusted_df)

    if (group_name[group_vars.index("Indicator Number")] == "1.1") & (
        group_name[group_vars.index("Innovation Name")] in ["teleworking"]
    ):  # Time series to be partialized up to 2019
        adjusted_df = group_data.copy()
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
        if description_new not in metadata["Description"]:
            description_counter += 1
            metadata["Description"][description_new] = convert_to_three_digit_notation(
                f"d{description_counter}"
            )
            counter_added_description += 1
        if metric_new not in metadata["Metric"]:
            metric_counter += 1
            metadata["Metric"][metric_new] = convert_to_three_digit_notation(
                f"m{metric_counter}"
            )
            counter_added_metric += 1
        adjusted_dfs.append(adjusted_df)

    # Commenting out block below as I'm not sure what to do with zero values
    # if (group_name[group_vars.index("Indicator Number")] == "1.1") & (
    #     group_name[group_vars.index("Innovation Name")]
    #     in ["microfinance", "eating less meat"]
    # ):  # Time series to be partialized up to minimum
    #     adjusted_df = group_data.copy()
    #     min_index = adjusted_df["Value"].idxmax()  # index of minimum
    #     adjusted_df = adjusted_df.loc[:min_index]  # truncate
    #     description_new = "Partial up to min " + str(
    #         adjusted_df["Description"].unique()[0]
    #     )
    #     adjusted_df["Description"] = description_new
    #     metric_new = "Partial up to min " + str(adjusted_df["Metric"].unique()[0])
    #     adjusted_df["Metric"] = metric_new
    #     # Now also update dictionaries
    #     if description_new not in metadata["Description"]:
    #         description_counter += 1
    #         metadata["Description"][description_new] = convert_to_three_digit_notation(
    #             f"d{description_counter}"
    #         )
    #         counter_added_description += 1
    #     if metric_new not in metadata["Metric"]:
    #         metric_counter += 1
    #         metadata["Metric"][metric_new] = convert_to_three_digit_notation(
    #             f"m{metric_counter}"
    #         )
    #         counter_added_metric += 1
    #     adjusted_dfs.append(adjusted_df)


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

print(f"{counter_scaling} datasets rescaled")

print(f"{counter_added_description} new Descriptions added")
print(f"{counter_added_metric} new Metrics added")

adoptions_df_new = pd.concat(adjusted_dfs)
new_data_fn = f"""{PATH}/adjusted_datasets_{VERSION}.csv"""
adoptions_df_new.to_csv(new_data_fn, index=False)
print(f"New data file successfully written to {new_data_fn}")
