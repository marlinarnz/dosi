import streamlit as st


import pandas as pd
import numpy as np

import re
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import nbformat

import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

# Get the path of the current script (inside streamlit/)
CURRENT_DIR = Path(__file__).parent.parent

VERSION_FOR_DATA = "v27"
VERSION_FOR_FITPARAMETERS = "v28"
VERSION_FOR_METADATA = "v25"
YEAR_PADDING_FOR_PLOTTING = 10
PATH = "../data"


st.set_page_config(layout="wide")  # add at the very top, before st.title

st.title("Within innovations")

version_data = st.text_input("Enter DoSI data file version to be used (must be > v26)", value=VERSION_FOR_DATA)
version_summary = st.text_input("Enter summary data file version to be used (must be > v26)", value=VERSION_FOR_FITPARAMETERS)

# Load data and logfit estimation summary
@st.cache_data
def load_data(version_data):
    return pd.read_csv(CURRENT_DIR / PATH / f"adjusted_datasets_{version_data}.csv", converters={"Indicator Number": str})
try:
    dosi_df = load_data(version_data)
except FileNotFoundError:
    st.error(f"⚠️ Data version '{version_data}' not found. Please enter a valid version. Otherwise, the default {VERSION_FOR_DATA} is used.")
    dosi_df = load_data(VERSION_FOR_DATA)

@st.cache_data
def load_summary(version_summary):
    return pd.read_csv(CURRENT_DIR / PATH / f"summary_table_{version_summary}.csv", converters={"Indicator Number": str})
try:
    summary_df = load_summary(version_summary)
except FileNotFoundError:
    st.error(f"⚠️ Data version '{version_summary}' not found. Please enter a valid version. Otherwise, the default {VERSION_FOR_FITPARAMETERS} is used.")
    summary_df = load_summary(VERSION_FOR_FITPARAMETERS)

# Load supplementary data
fn_early = CURRENT_DIR / PATH / "EarlyAdopterRegions_perInnovation_21March.csv"  # Early Adopting regions
fn_metadata = CURRENT_DIR / PATH / f"metadata_master_{VERSION_FOR_METADATA}.xlsx"
fn_clusters = CURRENT_DIR / PATH / "innovation_list_HWLclusters_v3.0.xlsx"

dosi_df["Value"] = pd.to_numeric(dosi_df["Value"], errors="coerce")
dosi_df = dosi_df.dropna(subset=["Value"])
# Correct for trailing spaces in the data
dosi_df["Spatial Scale"] = dosi_df["Spatial Scale"].str.rstrip()
dosi_df["Innovation Name"] = dosi_df["Innovation Name"].str.rstrip()

early_df = pd.read_csv(fn_early, usecols=[0, 1])
early_dict = dict(zip(early_df.iloc[:, 0], early_df.iloc[:, 1]))


# Metadata / codes

def convert_to_three_digit_notation(s):
    return re.sub(r"([a-zA-Z])(\d+)", lambda m: f"{m.group(1)}{int(m.group(2)):03}", s)


def read_metadata_table(fn, columns):
    df = pd.read_excel(fn, usecols=columns, dtype=str).dropna().reset_index(drop=True)
    df.iloc[:, 1] = df.iloc[:, 1].apply(convert_to_three_digit_notation)
    return df.set_index(df.columns[0])[df.columns[1]].to_dict()


metadata = dict()
metadata["Innovation Name"] = read_metadata_table(fn_metadata, "A,D")
metadata["Spatial Scale"] = read_metadata_table(fn_metadata, "G,I")
metadata["Indicator Name"] = read_metadata_table(fn_metadata, "L,M")
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

# Update metadata and early adopt innovation name label with cluster assignment file
clusters = pd.read_excel(fn_clusters)
clusters_names = clusters.set_index("innovation_name")["innovation_label"].to_dict()
metadata["Innovation Name"].update(clusters_names)
early_dict.update({code: 0 for code in clusters_names.values() if not code in early_dict.keys()})

# Attach codes to data file

dosi_df["Innovation Name"] = dosi_df["Innovation Name"].str.lower()
dosi_df["Innovation Code"] = dosi_df["Innovation Name"].map(metadata["Innovation Name"])
dosi_df["Region Code"] = (
    dosi_df["Spatial Scale"].str.lower().map(metadata["Spatial Scale"])
)
dosi_df["Early Adopter Code"] = dosi_df["Innovation Code"].map(early_dict)
dosi_df["Indicator Code"] = (
    dosi_df["Indicator Number"].str.lower().map(metadata["Indicator Number"])
)
dosi_df["Description Code"] = (
    dosi_df["Description"].str.lower().map(metadata["Description"])
)
dosi_df["Metric Code"] = dosi_df["Metric"].str.lower().map(metadata["Metric"])
#dosi_df["Code"] = dosi_df[
#    [
#        "Innovation Code",
#        "Region Code",
#        "Indicator Code",
#        "Description Code",
#        "Metric Code",
#    ]
#].agg("_".join, axis=1)
dosi_df["descriptionmetric"] = dosi_df[["Description", "Metric"]].agg(
    " / ".join, axis=1
)

innovation_names = dosi_df["Innovation Name"].unique().tolist()
innovation_names.sort()
indicator_codes = dosi_df["Indicator Number"].unique().tolist()
indicator_codes.sort()

summary_df["Region Code"] = (
    summary_df["Spatial Scale"].str.lower().map(metadata["Spatial Scale"])
)


def FPLogValue_with_scaling(x, t0, Dt, s):
    """
    Logistic function with vertical scaling.|
    """
    return s / (1 + np.exp(-np.log(81) * (x - t0) / Dt))

# Create menu

col1, col2 = st.columns(2)

with col1:
    selected_innovation = st.selectbox("Select innovation", innovation_names, index=0)

early_adopting_region_code = early_dict[
    metadata["Innovation Name"][selected_innovation]
]

other_region_codes = list(
    map(metadata["Spatial Scale"].get,
        [s.lower()
         for s in dosi_df[dosi_df["Innovation Name"] == selected_innovation]["Spatial Scale"
            ].unique().tolist()
        ]))


# function for reverse dictionary, taking the first element
def rev_dict(dictionary: dict, label):
    try:
        return next(key for key, value in dictionary.items() if value == label)
    except StopIteration:
        return list(dictionary.keys())[0]


with col2:

    selected_spatial_scale_code = st.selectbox(
        "Select spatial scale (first option is the early adopter region)",
        [early_adopting_region_code] + other_region_codes,
        # labels different from values
        format_func=lambda x: rev_dict(metadata["Spatial Scale"], x),
        # [rev_dict(metadata["Spatial Scale"], early_adopting_region_code)]
        # + [rev_dict(metadata["Spatial Scale"], x) for x in other_region_codes],
        index=0,
    )

innovation_df = dosi_df[
    (
        (dosi_df["Innovation Name"] == selected_innovation)
        & (dosi_df["Region Code"] == selected_spatial_scale_code)
    )
].copy()


NUMBER_OF_COLUMNS = 10  # Number of columns in the grid

st.subheader("Indicators included:")
cols = st.columns(NUMBER_OF_COLUMNS)
feature_states = {}
descriptionmetric_states = {}

for idx, label in enumerate(indicator_codes):
    with cols[idx % NUMBER_OF_COLUMNS]:
        descriptionmetric_states[label] = []
        feature_states[label] = st.checkbox(
            label + " " + metadata["Indicator Name"][label],
            value=label in innovation_df["Indicator Number"].unique().tolist(),
        )


innovation_df_for_plotting = innovation_df[
    innovation_df["Indicator Number"].isin([k for k, v in feature_states.items() if v])
].copy()

# Filter summary dataframe for selected innovation and spatial scale
innovation_summary_df = summary_df[
    (
        (summary_df["Innovation Name"] == selected_innovation)
        & (summary_df["Region Code"] == selected_spatial_scale_code)
    )
]


# ──────────────────────────────────────────────────────────────
# 4.  PLOTLY FIGURE  ───────────────────────────────────────────
# ----------------------------------------------------------------
def build_plot(innovation_df, innovation_summary_df) -> go.Figure:
    """Dummy plot builder – drop in your real logic here."""

    year_min = innovation_df["Year"].min() - YEAR_PADDING_FOR_PLOTTING
    year_max = innovation_df["Year"].max() + YEAR_PADDING_FOR_PLOTTING
    if not isinstance(year_min, int): year_min = 2000 - YEAR_PADDING_FOR_PLOTTING
    if not isinstance(year_max, int): year_max = 2000 + YEAR_PADDING_FOR_PLOTTING

    years_for_plotting = np.linspace(
        year_min, year_max, (year_max - year_min) + 1
    )  # 10 + 1)

    # Generate a color palette using Plotly (or you can use matplotlib or another method)
    colors = px.colors.qualitative.Set1  # Set1 is a predefined color palette

    fig = go.Figure()

    for i, code in enumerate(innovation_df["Indicator Number"].unique()):
        t0 = innovation_summary_df[innovation_summary_df["Indicator Number"] == code][
            "log_t0"
        ].iloc[0]
        Dt = innovation_summary_df[innovation_summary_df["Indicator Number"] == code][
            "log_Dt"
        ].iloc[0]
        K = innovation_summary_df[innovation_summary_df["Indicator Number"] == code][
            "log_K"
        ].iloc[0]

        # Assign color from the color cycle
        color = colors[
            i % len(colors)
        ]  # Cycle through the colors if more codes than colors

        # Add the points trace (same color as line)
        fig.add_trace(
            go.Scatter(
                x=innovation_df[innovation_df["Indicator Number"] == code]["Year"],
                y=innovation_df[innovation_df["Indicator Number"] == code]["Value"] / K,
                mode="markers",
                name=f"""{code} {metadata['Indicator Name'][code]} (Metric {innovation_df[innovation_df["Indicator Number"] == code]["Metric"].unique().tolist()}) K-normalized data """,  # This can be the same name to link with the line in the legend
                hovertemplate=f"""{code} {metadata['Indicator Name'][code]} (Metric {innovation_df[innovation_df["Indicator Number"] == code]["Metric"].unique().tolist()}) <br>{code} Point<br>Year=%{{x:.0f}}<br>value=%{{y:.2f}}<extra></extra>""",  # Custom tooltip
                marker=dict(size=8, color=color),  # Same color for points as the line
            )
        )

        fig.add_trace(
            go.Scatter(
                x=years_for_plotting,
                y=FPLogValue_with_scaling(years_for_plotting, t0, Dt, K) / K,
                mode="lines",
                name=code,  # Legend label
                showlegend=False,
                line=dict(color=color, width=2),
                hovertemplate=f"""{code} {metadata['Indicator Name'][code]} <br>{code} (Metric {innovation_df[innovation_df["Indicator Number"] == code]["Metric"].unique().tolist()}) <br>Year=%{{x:.0f}}<br>Value=%{{y:.2f}}<br>Dt={Dt:.0f} t0={t0:.0f} K={K:.2f}<extra></extra>""",  # Custom tooltip
            )
        )

        fig.update_layout(
            title="Innovation "
            + selected_innovation
            + " in "
            + rev_dict(metadata["Spatial Scale"], selected_spatial_scale_code),
            xaxis_title="Year",
            yaxis_title="K-normalized value",
            # hovermode='x unified'
            yaxis=dict(range=[0, 1.2]),  # Set the y-axis limits to [0, 5]
        )

        # centroid of the scatter points
        x_centroid = innovation_df.loc[
            innovation_df["Indicator Number"] == code, "Year"
        ].mean()
        y_centroid = (
            innovation_df.loc[innovation_df["Indicator Number"] == code, "Value"] / K
        ).mean()

        fig.add_annotation(
            x=x_centroid,
            y=y_centroid,
            text=f"""{code} {metadata['Indicator Name'][code]}""",
            showarrow=False,
            xanchor="center",
            yanchor="middle",
            font=dict(color=color),  # label colour = line colour
        )

        fig.update_layout(showlegend=False)  # put this once, just before `return fig`

        # ⬆️ add this *once* after all traces and just before returning the figure
        fig.update_layout(height=900)  # make the plot taller

    return fig


fig = build_plot(
    innovation_df=innovation_df_for_plotting,
    innovation_summary_df=innovation_summary_df,
)
st.plotly_chart(fig, use_container_width=True)
