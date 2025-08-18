import pandas as pd
import numpy as np

import re
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.backends.backend_pdf import PdfPages

VERSION = "v26"
# VERSION_FOR_METADATA = "v23"
SMALL_SUBSET = False  # Do you only want a small subset for testing?
RENUMBER_METADATA_CODES = False
APPLY_TRANSFORMATIONS_TO_DATA_FILE = True  # Should transformations such as cumulation be applied (True), or not (False)? This is important because otherwise there will be doubles

PATH = "/mnt/c/Users/simon.destercke/Documents/misc/iiasa/DoSI"
fn_data = f"{PATH}/Merged_Cleaned_Pitchbook_WebOfScience_GoogleTrends_Data_12Mar25.xlsx"

fn_market_share_indicators = (
    f"{PATH}/Supplemental MS denominator data_20250208_SDScorrection20250215.xlsx"
)
sheetname_market_share_indicators = "Data"

adoptions_df = pd.read_excel(
    fn_data, sheet_name="Sheet1", converters={"Indicator Number": str}
)
adoptions_df["Value"] = pd.to_numeric(adoptions_df["Value"], errors="coerce")
adoptions_df = adoptions_df.dropna(subset=["Value"])

# Correct for trailing spaces in the data
adoptions_df["Spatial Scale"] = adoptions_df["Spatial Scale"].str.rstrip()
adoptions_df["Innovation Name"] = adoptions_df["Innovation Name"].str.rstrip()

# Correct for 'passive building retrofits missnaming'
adoptions_df["Innovation Name"] = adoptions_df["Innovation Name"].replace(
    "passive building retrofits", "passive buildings"
)

# For debugging: only subset
if SMALL_SUBSET:
    adoptions_df = adoptions_df[
        adoptions_df["Innovation Name"].isin(["car sharing"])
        # adoptions_df["Indicator Number"].isin(["3.3", "3.5", "4.1"])
        # adoptions_df["Indicator Number"].isin(["1.1"])
    ]

fn_metadata = f"{PATH}/metadata_master_v23_CWedit_SDS20250404.xlsx"


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


def get_value_case_insensitive(d, search_key):
    for key in d:
        if key.lower() == search_key.lower():
            return d[key]
    raise KeyError(f"Key '{search_key}' not found (case-insensitive match)")


if False:  # Make everything lowercase?
    for key, nested_dict in metadata.items():
        if isinstance(nested_dict, dict):  # Ensure the value is a dictionary
            metadata[key] = {
                k.lower() if isinstance(k, str) else k: v
                for k, v in nested_dict.items()
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
                get_value_case_insensitive(metadata[group_vars[i]], group_name[i])
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


def update_dictionaries(description_new, metric_new, df):

    global metadata, description_counter, metric_counter, counter_added_description, counter_added_metric

    if description_new.lower() not in [x.lower() for x in metadata["Description"]]:
        description_counter += 1
        metadata["Description"][description_new] = convert_to_three_digit_notation(
            f"d{description_counter}"
        )
        counter_added_description += 1
    if metric_new.lower() not in [x.lower() for x in metadata["Metric"]]:
        metric_counter += 1
        metadata["Metric"][metric_new] = convert_to_three_digit_notation(
            f"m{metric_counter}"
        )
        counter_added_metric += 1

    df["Description"] = description_new
    df["Metric"] = metric_new

    return df


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
            in ["climate protest", "passive building retrofits", "passive buildings"]
        )
    ):  # condition for cumulating
        adjusted_df = group_data.copy()
        adjusted_df["Value"] = adjusted_df["Value"].cumsum()
        description_new = "cumulative " + str(adjusted_df["Description"].unique()[0])
        metric_new = "cumulative " + str(adjusted_df["Metric"].unique()[0])
        adjusted_df = update_dictionaries(description_new, metric_new, adjusted_df)
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
        metric_new = "Partial up to max " + str(adjusted_df["Metric"].unique()[0])
        adjusted_df = update_dictionaries(description_new, metric_new, adjusted_df)
        adjusted_dfs.append(adjusted_df)

    if (group_name[group_vars.index("Indicator Number")] == "1.1") & (
        group_name[group_vars.index("Innovation Name")] in ["eating less meat"]
    ):  # Time series to be partialized after the maximum
        adjusted_df = group_data.copy()
        max_index = adjusted_df["Value"].idxmax()  # index of maximum
        adjusted_df = adjusted_df.loc[max_index:]  # truncate
        description_new = "Partial max and after " + str(
            adjusted_df["Description"].unique()[0]
        )
        metric_new = "Partial max and after " + str(adjusted_df["Metric"].unique()[0])
        adjusted_df = update_dictionaries(description_new, metric_new, adjusted_df)
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
        metric_new = "Partial up to 2019 " + str(adjusted_df["Metric"].unique()[0])
        adjusted_df = update_dictionaries(description_new, metric_new, adjusted_df)
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

# Now include market shares
market_share_indicators = pd.read_excel(
    fn_market_share_indicators,
    sheet_name=sheetname_market_share_indicators,
    header=None,
).T
# Set the first row as column headers
market_share_indicators.columns = market_share_indicators.iloc[
    0
]  # Assign the first row as header
market_share_indicators = market_share_indicators[1:].reset_index(
    drop=True
)  # Drop the first row from the data
# Drop unneeded columns
market_share_indicators.drop(columns=["Source", "URL"], inplace=True)
# Melt
market_share_indicators = market_share_indicators.melt(
    id_vars=["SI", "Indicator", "Country", "units"], var_name="Year", value_name="Value"
)


def ms_check_empty(df, inn_name):
    if len(df) == 0:
        print(f"PROBLEM IN MARKET SHARE CALCULATION FOR {inn_name}")


# Build new market share data frame

WEIGHT_HEAVIEST_CAR_KG = 3129  # heaviest vehicle available in 2025 (kg) = 3129 (Tesla Cybertruck Beast).  Use that value for all years.
columns_to_keep = list(
    adoptions_df.columns
)  # To keep only the core after merging the indicators dataframe for auxiliary calculations

innovation_name = "eating less meat"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (adoptions_df["Description"] == "% red in total meat consumption")
].copy()
ms_check_empty(market_share_df, innovation_name)
description_new = "red meat as a share of meat consumption"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
adjusted_dfs.append(market_share_df)

innovation_name = "organic food consumption"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (adoptions_df["Description"] == "Organic retail sales share [%]")
].copy()
ms_check_empty(market_share_df, innovation_name)
market_share_df["Value"] = market_share_df["Value"] / 100
description_new = "organic as a share of retail sales"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
adjusted_dfs.append(market_share_df)

innovation_name = "mobesity"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (
        adoptions_df["Description"].isin(
            [
                "Average weight of all new car sales / registrations (kg)",
                "Average weight of all new sales / registrations (kg)",
            ]
        )
    )  # Alternative descriptions most likely meaning the same
].copy()
ms_check_empty(market_share_df, innovation_name)
market_share_df["Value"] = market_share_df["Value"] / WEIGHT_HEAVIEST_CAR_KG
description_new = (
    "Weight of all new car sales as a share of heaviest vehicle available in 2025."
)
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
adjusted_dfs.append(market_share_df)

innovation_name = "low-carbon long distance travel"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (adoptions_df["Description"] == "Passengers carried in railways")
].copy()
ms_check_empty(market_share_df, innovation_name)
market_share_df = market_share_df.merge(
    market_share_indicators[market_share_indicators["Indicator"] == "road+rail pkm"],
    left_on=["Spatial Scale", "Year"],
    right_on=["Country", "Year"],
    how="left",
)
market_share_df["Value"] = market_share_df["Value_x"] * 1e6 / market_share_df["Value_y"]
description_new = "share of pkm by rail"
metric_new = "market share"
market_share_df = market_share_df[columns_to_keep]
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
adjusted_dfs.append(market_share_df)


innovation_name = "car ownership"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (
        adoptions_df["Metric"].isin(
            ["cars per 1,000 inhabitants", "cars per 1000 inhabitants"]
        )
    )
].copy()
ms_check_empty(market_share_df, innovation_name)
market_share_df["Value"] = market_share_df["Value"] / 1e3
description_new = "cars per person"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
adjusted_dfs.append(market_share_df)

# Greg's request from 2025-05-10
innovation_name = "car ownership"
market_share_df = adoptions_df[
    (adoptions_df["Spatial Scale"] == "Berlin")
    & (adoptions_df["Innovation Name"] == innovation_name)
    & (
        adoptions_df["Metric"].isin(
            ["cars per 1,000 inhabitants", "cars per 1000 inhabitants"]
        )
    )
].copy()
ms_check_empty(market_share_df, innovation_name)
market_share_df["Value"] = market_share_df["Value"] / 1e3
# Find the Year corresponding to the maximum Value
max_value_year = market_share_df.loc[market_share_df["Value"].idxmax(), "Year"]
# Remove rows where Year is greater than or equal to that Year
market_share_df = market_share_df[market_share_df["Year"] >= max_value_year]
description_new = "cars per person PARTIAL FROM MAX ONWARDS"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
adjusted_dfs.append(market_share_df)

innovation_name = "teleworking"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (
        adoptions_df["Description"].isin(
            [
                "Employed persons teleworking as a % of total employment",
                "Employed persons teleworking as a percentage of the total employment (%)",
            ]
        )
    )  # Alternative descriptions most likely meaning the same
].copy()
ms_check_empty(market_share_df, innovation_name)
description_new = "teleworkers as a share of all employed persons"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
adjusted_dfs.append(market_share_df)

innovation_name = "active mobility"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (
        adoptions_df["Description"].isin(
            [
                "Passenger kilometres travelled by bike",  # Weird symbol needed, is in Excel file
                "Passenger kilometres travelled by foot",  # Weird symbol needed, is in Excel file
                "Modal share of all trips by residents (bike)",
                "Modal share of all trips by residents (walk)",
                "Bicycle modal share",
            ]
        )
    )  # Alternative descriptions most likely meaning the same
].copy()
market_share_df = (
    market_share_df.groupby(
        [
            "Year",
            "Spatial Scale",
            "Innovation Name",
            "Indicator Number",
            "Indicator Name",
        ]
    )
    .agg(
        {
            "Value": "sum",
            "Data Source": lambda x: ", ".join(map(str, set(x.dropna()))),
            "Comments": lambda x: ", ".join(map(str, set(x.dropna()))),
            "File": lambda x: ", ".join(map(str, set(x.dropna()))),
            "Sheet": lambda x: ", ".join(
                map(str, set(x.dropna()))
            ),  # Join unique values as a string
        }
    )
    .reset_index()
)
ms_check_empty(market_share_df, innovation_name)
market_share_df = market_share_df.merge(
    market_share_indicators[
        market_share_indicators["Indicator"] == "pkm- road & rail (millions)"
    ],
    left_on=[
        "Sheet",
        "Year",
    ],  # Because the spatial scale is "The Netherlands" and we need "Netherlands" for matching
    right_on=["Country", "Year"],
    how="left",
)
market_share_df["Value"] = market_share_df["Value_x"] / 100
market_share_df.loc[market_share_df["Spatial Scale"] == "The Netherlands", "Value"] = (
    market_share_df.loc[
        market_share_df["Spatial Scale"] == "The Netherlands", "Value_x"
    ]
    * 1000
    / market_share_df.loc[
        market_share_df["Spatial Scale"] == "The Netherlands", "Value_y"
    ]
).astype(float)
description_new = "% trips by walking and biking"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
market_share_df = market_share_df[columns_to_keep]
adjusted_dfs.append(market_share_df)


# Request from Greg 2025-05-10
updated_df = pd.concat(adjusted_dfs)
adjusted_df = pd.concat(adjusted_dfs)[
    (updated_df["Innovation Name"] == "active mobility")
    & (updated_df["Spatial Scale"] == "Amsterdam")
    & (updated_df["Indicator Number"] == "1.1")
    & (
        (
            (
                updated_df["Description"]
                == "Modal share of all trips by residents (walk)"
            )
            & (updated_df["Metric"] == "% trips by walking")
        )
    )
    & (updated_df["Year"] != 1930)
].copy()
description_new = "Modal share of all trips by residents (walk) EXCLUDING 1930"
metric_new = "% trips by walking"
adjusted_df = update_dictionaries(description_new, metric_new, adjusted_df)
adjusted_dfs.append(adjusted_df)

# Request from Greg 2025-05-10
updated_df = pd.concat(adjusted_dfs)
adjusted_df = pd.concat(adjusted_dfs)[
    (updated_df["Innovation Name"] == "active mobility")
    & (updated_df["Spatial Scale"] == "Amsterdam")
    & (updated_df["Indicator Number"] == "1.1")
    & (
        (
            (updated_df["Description"] == "% trips by walking and biking")
            & (updated_df["Metric"] == "market share")
        )
    )
    & (updated_df["Year"] != 1930)
].copy()
description_new = "% trips by walking and biking EXCLUDING 1930"
metric_new = "market share"
adjusted_df = update_dictionaries(description_new, metric_new, adjusted_df)
adjusted_dfs.append(adjusted_df)


innovation_name = "e-bikes"  # Part one, no adjustment needed
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (adoptions_df["Description"] == "Market share")
].copy()
ms_check_empty(market_share_df, innovation_name)
market_share_df["Value"] = market_share_df["Value"] / 100
description_new = "e-bikes as a share of bikes sold"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
adjusted_dfs.append(market_share_df)

innovation_name = "e-bikes"  # Part two, adjustment needed
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (adoptions_df["Description"] == "E-bike sales volumes")
].copy()
ms_check_empty(market_share_df, innovation_name)
market_share_df = market_share_df.merge(
    market_share_indicators[market_share_indicators["Indicator"] == "total bike sales"],
    left_on=["Spatial Scale", "Year"],
    right_on=["Country", "Year"],
    how="left",
)
market_share_df["Value"] = (
    market_share_df["Value_x"]
    / market_share_df["Value_y"]
    / np.where(market_share_df["Spatial Scale"] == "China", 1, 1000)
)
description_new = "e-bikes as a share of bikes sold"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
market_share_df = market_share_df[columns_to_keep]
adjusted_dfs.append(market_share_df)


innovation_name = "drivers licence"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (
        (
            (adoptions_df["Metric"].isin(["% of <=19 yr olds"]))
            & (adoptions_df["Spatial Scale"].isin(["US", "Washington DC"]))
            & (adoptions_df["Indicator Number"] == "1.1")
        )
        | (
            (
                adoptions_df["Description"]
                == "% of 18-19yr age group holding a drivers licence"
            )
            & (adoptions_df["Spatial Scale"].isin(["Sweden", "Stockholm"]))
        )
    )
].copy()
ms_check_empty(market_share_df, innovation_name)
description_new = "share of teenagers with drivers licenses"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
adjusted_dfs.append(market_share_df)


innovation_name = "downsizing"  # Part one, no adjustment needed
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (
        adoptions_df["Description"]
        == "Share of people living in a small dwelling with high wellbeing"
    )
].copy()
ms_check_empty(market_share_df, innovation_name)
market_share_df["Value"] = market_share_df["Value"] / 100
description_new = "share of people living in a small dwelling with high wellbeing"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
adjusted_dfs.append(market_share_df)


innovation_name = "co-housing"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (adoptions_df["Metric"].isin(["# residents", "# projects", "# cooperatives"]))
].copy()
ms_check_empty(market_share_df, innovation_name)
market_share_indicators["Country"] = market_share_indicators["Country"].replace(
    {"Vaud": "Canton de Vaud (Switzerland)"}
)
market_share_df = market_share_df.merge(
    market_share_indicators[
        (market_share_indicators["Indicator"] == "Population")
        & (market_share_indicators["SI"] == "co-housing")
    ],
    left_on=["Spatial Scale", "Year"],
    right_on=["Country", "Year"],
    how="left",
)
market_share_df["Value"] = (
    market_share_df["Value_x"]
    / market_share_df["Value_y"]
    * np.where(
        market_share_df["Spatial Scale"].isin(
            ["Canton de Vaud (Switzerland)", "Germany"]
        ),
        22,
        1,
    )
)
description_new = "share of population living in co-housing projects"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
market_share_df = market_share_df[columns_to_keep]
adjusted_dfs.append(market_share_df)


innovation_name = "sustainable fashion"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (adoptions_df["Metric"] == "% market share (sustainable apparel)")
].copy()
ms_check_empty(market_share_df, innovation_name)
description_new = "sustainable apparel as a share of apparel"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
adjusted_dfs.append(market_share_df)

innovation_name = "food waste reduction"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (
        adoptions_df["Description"].isin(
            [
                "Food waste generated in the US",
                "Global edible food waste per capita, total",
            ]
        )
    )
].copy()
market_share_df = market_share_df.merge(
    market_share_indicators[
        (market_share_indicators["Indicator"] == "total kg food")
        & (market_share_indicators["SI"] == "eating less meat")
    ],
    left_on=["Spatial Scale", "Year"],
    right_on=["Country", "Year"],
    how="left",
)
ms_check_empty(market_share_df, innovation_name)
market_share_df["Value"] = market_share_df["Value_x"] / market_share_df["Value_y"]
description_new = "share of food that is wasted"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
market_share_df = market_share_df[columns_to_keep]
adjusted_dfs.append(market_share_df)


innovation_name = "e-commerce"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (
        adoptions_df["Description"].isin(
            [
                "Internet sales as a percentage of total retail sales (ratio) (%)",
                "Internet sales as a percentage of total retail (B2C) sales (ratio) (%)",
            ]
        )
    )
].copy()
ms_check_empty(market_share_df, innovation_name)
market_share_df["Value"] = market_share_df["Value"] / 100
description_new = "Internet sales as a share of total retail sales"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
adjusted_dfs.append(market_share_df)


innovation_name = "passive buildings"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (adoptions_df["Description"].isin(["passive retrofits"]))
    & (adoptions_df["Metric"].isin(["# of retrofitted units"]))
].copy()
market_share_df = market_share_df.merge(
    market_share_indicators[
        (market_share_indicators["Indicator"] == "# new renovated houses")
        & (market_share_indicators["SI"] == "passive building retrofits")
    ],
    left_on=["Spatial Scale", "Year"],
    right_on=["Country", "Year"],
    how="left",
)
ms_check_empty(market_share_df, innovation_name)
market_share_df["Value"] = market_share_df["Value_x"] / market_share_df["Value_y"]
description_new = "share of building stock getting passive-bldg retrofits"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
market_share_df = market_share_df[columns_to_keep]
adjusted_dfs.append(market_share_df)


innovation_name = "car sharing"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (adoptions_df["Description"].isin(["registered drivers"]))
    & (adoptions_df["Metric"].isin(["# drivers"]))
].copy()
market_share_df = market_share_df.merge(
    market_share_indicators[
        (market_share_indicators["Indicator"] == "population")
        & (market_share_indicators["SI"] == "car sharing")
    ],
    left_on=["Spatial Scale", "Year"],
    right_on=["Country", "Year"],
    how="left",
)
ms_check_empty(market_share_df, innovation_name)
market_share_df["Value"] = market_share_df["Value_x"] / market_share_df["Value_y"]
description_new = "share of drivers who car share"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
market_share_df = market_share_df[columns_to_keep]
adjusted_dfs.append(market_share_df)


innovation_name = "microfinance"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (adoptions_df["Description"].isin(["Number of active borrowers"]))
    & (adoptions_df["Metric"].isin(["No."]))
].copy()
market_share_df = market_share_df.merge(
    market_share_indicators[
        (market_share_indicators["Indicator"] == "Population")
        & (market_share_indicators["SI"] == "microfinance")
    ],
    left_on=["Spatial Scale", "Year"],
    right_on=["Country", "Year"],
    how="left",
)
ms_check_empty(market_share_df, innovation_name)
market_share_df["Value"] = market_share_df["Value_x"] / market_share_df["Value_y"]
description_new = "active borrowers as a share of population"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
market_share_df = market_share_df[columns_to_keep]
adjusted_dfs.append(market_share_df)


innovation_name = "solar leasing"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (
        adoptions_df["Description"].isin(
            [
                "% third party owned systems (income=$50k-$100k)",
                "% third party owned systems (income<$50k)",
            ]
        )
    )
    & (adoptions_df["Metric"].isin(["%"]))
].copy()
market_share_df = (
    market_share_df.groupby(
        [
            "Year",
            "Spatial Scale",
            "Innovation Name",
            "Indicator Number",
            "Indicator Name",
        ]
    )
    .agg(
        {
            "Value": "mean",
            "Data Source": lambda x: ", ".join(map(str, set(x.dropna()))),
            "Comments": lambda x: ", ".join(map(str, set(x.dropna()))),
            "File": lambda x: ", ".join(map(str, set(x.dropna()))),
            "Sheet": lambda x: ", ".join(
                map(str, set(x.dropna()))
            ),  # Join unique values as a string
        }
    )
    .reset_index()
)
ms_check_empty(market_share_df, innovation_name)
description_new = "share of new solar owned by 3rd parties (HH<$100k)"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
market_share_df = market_share_df[columns_to_keep]
adjusted_dfs.append(market_share_df)


# Request from Greg 2025-05-10 to add in new series based on aggregated ragions
# function from DeepSeek
def add_aggregated_rows(
    df,
    aggregation_mapping,
    group_cols=["Year"],
    agg_col="Value",
    spatial_scale_col="Spatial Scale",
):
    """
    Adds aggregated rows to a DataFrame based on specified groupings of spatial scales

    Parameters:
    df (pd.DataFrame): Input DataFrame
    aggregation_mapping (dict): Dictionary mapping {new_category: [component_categories]}
    group_cols (list): Columns to group by when aggregating (default: ['Year'])
    agg_col (str): Column name to aggregate (default: 'Value')
    spatial_scale_col (str): Column name containing spatial categories (default: 'Spatial Scale')

    Returns:
    pd.DataFrame: DataFrame with original rows plus aggregated rows
    """

    # Validate inputs
    if not isinstance(aggregation_mapping, dict):
        raise ValueError("aggregation_mapping must be a dictionary")

    # Create list to hold aggregated DataFrames
    aggregated_dfs = []

    # Get all columns except group columns, agg column, and spatial scale column
    other_cols = [
        col
        for col in df.columns
        if col not in group_cols + [agg_col, spatial_scale_col]
    ]

    for agg_name, components in aggregation_mapping.items():
        # Filter rows for current components
        mask = df[spatial_scale_col].isin(components)
        filtered_df = df[mask]

        if filtered_df.empty:
            continue

        # Create aggregation dictionary
        agg_dict = {agg_col: "sum", **{col: "first" for col in other_cols}}

        # Group and aggregate
        grouped = filtered_df.groupby(group_cols, as_index=False).agg(agg_dict)

        # Add aggregated spatial scale name
        grouped[spatial_scale_col] = agg_name

        # Maintain original column order
        grouped = grouped[df.columns.tolist()]

        aggregated_dfs.append(grouped)

    # Combine original DF with all aggregated rows
    return pd.concat([df] + aggregated_dfs, ignore_index=True)


innovation_name = "firm ESG reporting"
market_share_df = add_aggregated_rows(
    df=adoptions_df[
        (adoptions_df["Innovation Name"] == innovation_name)
        & (adoptions_df["Description"].isin(["Voluntary adoption of GRI reporting"]))
        & (adoptions_df["Metric"].isin(["# of companies"]))
    ].copy(),
    aggregation_mapping={
        "Asia-Pacific": ["Asia", "Oceania"],
        "Americas": ["North America", "LatinAmericaCarib"],
        "Middle East & Africa": ["Africa"],
    },
)
market_share_df = market_share_df.merge(
    market_share_indicators[
        (market_share_indicators["Indicator"] == "# of firms")
        & (market_share_indicators["SI"] == innovation_name)
    ],
    left_on=["Spatial Scale", "Year"],
    right_on=["Country", "Year"],
    how="left",
)
ms_check_empty(market_share_df, innovation_name)
market_share_df["Value"] = market_share_df["Value_x"] / market_share_df["Value_y"]
description_new = "share of firms voluntarily adopting gri reporting"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
market_share_df = market_share_df[columns_to_keep]
adjusted_dfs.append(market_share_df)


# Textile recycling
innovation_name = "textile recycling"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (
        adoptions_df["Description"].isin(
            ["Recycled textiles as a share of textiles generation."]
        )
    )
    & (adoptions_df["Metric"] == "%")
].copy()
ms_check_empty(market_share_df, innovation_name)
description_new = "recycled textiles as a share of textiles generation."
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
adjusted_dfs.append(market_share_df)


# Digital skills
innovation_name = "digital skills"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (
        adoptions_df["Description"].isin(
            [
                "Online activity: doing online course",
                "Online activity: emailing",
                "Online activity: social networks",
                "Online activity: finding info",
                "Online activity: banking",
                "Online activity: selling",
            ]
        )
    )
    & (adoptions_df["Metric"].isin(["% individuals"]))
].copy()
market_share_df = (
    market_share_df.groupby(
        [
            "Year",
            "Spatial Scale",
            "Innovation Name",
            "Indicator Number",
            "Indicator Name",
        ]
    )
    .agg(
        {
            "Value": "mean",
            "Data Source": lambda x: ", ".join(map(str, set(x.dropna()))),
            "Comments": lambda x: ", ".join(map(str, set(x.dropna()))),
            "File": lambda x: ", ".join(map(str, set(x.dropna()))),
            "Sheet": lambda x: ", ".join(
                map(str, set(x.dropna()))
            ),  # Join unique values as a string
        }
    )
    .reset_index()
)
ms_check_empty(market_share_df, innovation_name)
market_share_df["Value"] = (
    market_share_df["Value"] / 100
)  # Because values are >1 and we want shares 0-1
description_new = "share of people engaged in 6 online activities"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
market_share_df = market_share_df[columns_to_keep]
adjusted_dfs.append(market_share_df)


# e-government
innovation_name = "e-government"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (
        adoptions_df["Description"]
        == "% people who interacted online with public authorities (in the past year)"
    )
    & (adoptions_df["Metric"] == "%")
].copy()
ms_check_empty(market_share_df, innovation_name)
market_share_df["Value"] = (
    market_share_df["Value"] / 100
)  # Because values are >1 and we want shares 0-1
description_new = (
    "share of people who interacted with public authorities (in the past year)"
)
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
adjusted_dfs.append(market_share_df)


# climate protest
innovation_name = "climate protest"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (
        adoptions_df["Description"]
        == "Count of participants at protest events related to climate"
    )
    & (adoptions_df["Metric"] == "# people (estimated)")
].copy()
market_share_df["Value"] = market_share_df.groupby(
    [
        "Spatial Scale",
        "Innovation Name",
        "Indicator Number",
        "Indicator Name",
    ]
)[
    "Value"
].cumsum()  # Get cumulative sum
market_share_df = market_share_df.merge(
    market_share_indicators[
        (market_share_indicators["Indicator"] == "Population")
        & (market_share_indicators["SI"] == innovation_name)
    ],
    left_on=["Spatial Scale", "Year"],
    right_on=["Country", "Year"],
    how="left",
)
ms_check_empty(market_share_df, innovation_name)
market_share_df["Value"] = market_share_df["Value_x"] / market_share_df["Value_y"]
description_new = (
    "cumulative share of population participating in protest events related to climate"
)
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
market_share_df = market_share_df[columns_to_keep]
adjusted_dfs.append(market_share_df)


# energy community
innovation_name = "energy community"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (adoptions_df["Description"] == "Total energy communities")
    & (adoptions_df["Metric"] == "# communities")
].copy()
market_share_df = market_share_df.merge(
    market_share_indicators[
        (market_share_indicators["Indicator"] == "Population")
        & (market_share_indicators["SI"] == innovation_name)
    ],
    left_on=["Spatial Scale", "Year"],
    right_on=["Country", "Year"],
    how="left",
)
ms_check_empty(market_share_df, innovation_name)
market_share_df["Value"] = 200 * market_share_df["Value_x"] / market_share_df["Value_y"]
description_new = "share of population in energy communities"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
market_share_df = market_share_df[columns_to_keep]
adjusted_dfs.append(market_share_df)


# quitting smoking
innovation_name = "quitting smoking"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (adoptions_df["Description"] == "Share of adults who smoke")
    & (adoptions_df["Metric"] == "% of adults")
].copy()
ms_check_empty(market_share_df, innovation_name)
market_share_df["Value"] = (
    market_share_df["Value"] / 100
)  # Because values are >1 and we want shares 0-1
description_new = "share of payments that are non-cash"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
adjusted_dfs.append(market_share_df)


# postage stamps - no market share data needed


# non-cash transactions
innovation_name = "non-cash transactions"
market_share_df = adoptions_df[
    (adoptions_df["Innovation Name"] == innovation_name)
    & (
        (
            (
                adoptions_df["Description"]
                == "Share of payment instrument use for all payments"
            )
            & (adoptions_df["Metric"] == "% cash payments as % of all payments")
            & (adoptions_df["Spatial Scale"] == "US")
        )
        | (
            (
                adoptions_df["Description"]
                == "proportion of cash payments to all payment types (total numbers)"
            )
            & (adoptions_df["Metric"] == "% cash payments of total number of payments")
            & (adoptions_df["Spatial Scale"] == "UK")
        )
        | (
            (
                adoptions_df["Description"]
                == "Percentage of people who paid cash for their last in-store purchase"
            )
            & (adoptions_df["Metric"] == "% most recent in-store purchase in cash")
            & (adoptions_df["Spatial Scale"] == "Sweden")
        )
    )
].copy()
ms_check_empty(market_share_df, innovation_name)
market_share_df["Value"] = 1 - (
    market_share_df["Value"] / 100
)  # Because values are >1 and we want shares 0-1
description_new = "share of payments that are non-cash"
metric_new = "market share"
market_share_df = update_dictionaries(description_new, metric_new, market_share_df)
adjusted_dfs.append(market_share_df)


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
