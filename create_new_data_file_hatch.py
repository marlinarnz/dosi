import pandas as pd
import numpy as np

import re
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.backends.backend_pdf import PdfPages

VERSION = "HACTH_v26"
# VERSION_FOR_METADATA = "v23"
SMALL_SUBSET = False  # Do you only want a small subset for testing?
RENUMBER_METADATA_CODES = False
APPLY_TRANSFORMATIONS_TO_DATA_FILE = True  # Should transformations such as cumulation be applied (True), or not (False)? This is important because otherwise there will be doubles

ONEDRIVE_PATH = (
    "/mnt/c/Users/simon.destercke/IIASA/EDITS - FT25-1_PosTip/Data/HATCH files/"
)

PATH = "/mnt/c/Users/simon.destercke/Documents/misc/iiasa/DoSI"
fn_data = f"{ONEDRIVE_PATH}/hatch_data_dosi_format.csv"

fn_market_share_indicators = (
    f"{PATH}/Supplemental MS denominator data_20250208_SDScorrection20250215.xlsx"
)
sheetname_market_share_indicators = "Data"

adoptions_df = pd.read_csv(fn_data, converters={"Indicator Number": str})
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

fn_metadata = f"{PATH}/../ClusterFunc/metadata_master_v25_withhatch.xlsx"


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

# Get all the unique values in adoptions_df["Description"] that are not in metadata["Description"].values()
for col in ["Description", "Metric"]:
    unique_values = adoptions_df[col].unique()
    known_values = set(metadata[col].keys())
    unknown_values = [v for v in unique_values if v not in known_values]
    if len(unknown_values) > 0:
        # print(f"Unknown values in column {col}: {unknown_values}")
        # Print a value per row
        for v in unknown_values:
            print(f"{v}")
