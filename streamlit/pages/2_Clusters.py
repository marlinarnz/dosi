import streamlit as st

import pandas as pd
import numpy as np

import re
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import nbformat

import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

# Get the path of the current script (inside streamlit/)
CURRENT_DIR = Path(__file__).parent

VERSION_FOR_DATA = "v25"
VERSION_FOR_FITPARAMETERS = "v26"
VERSION_FOR_METADATA = "v25"
YEAR_PADDING_FOR_PLOTTING = 10

PATH = "../data"
fn_data = CURRENT_DIR / PATH / f"adjusted_datasets_{VERSION_FOR_DATA}.csv"
fn_summary = CURRENT_DIR / PATH / f"""summary_table_{VERSION_FOR_FITPARAMETERS}.csv"""
fn_clusters = CURRENT_DIR / PATH / "PosTip_Clusters.csv"  # Summary file by Charlie
fn_early = (
    CURRENT_DIR / PATH / "EarlyAdopterRegions_perInnovation_21March.csv"
)  # Early Adopting regions
fn_metadata = CURRENT_DIR / PATH / f"metadata_master_{VERSION_FOR_METADATA}.xlsx"

dosi_df = pd.read_csv(fn_data, converters={"Indicator Number": str})
dosi_df["Value"] = pd.to_numeric(dosi_df["Value"], errors="coerce")
dosi_df = dosi_df.dropna(subset=["Value"])

# Correct for trailing spaces in the data
dosi_df["Spatial Scale"] = dosi_df["Spatial Scale"].str.rstrip()
dosi_df["Innovation Name"] = dosi_df["Innovation Name"].str.rstrip()

summary_df = pd.read_csv(fn_summary, converters={"Indicator Number": str})

clusters_df = pd.read_csv(
    fn_clusters,
    skiprows=15,
    nrows=28,
    usecols=[8, 35, 36, 37, 38, 39],
    encoding="ISO-8859-1",
    header=0,
)
clusters_df.rename(
    columns={clusters_df.columns[0]: "innovation code"}, inplace=True
)  # If there is an error here, then there may be a column reference error, e.g. the first column of the csv file is empty and pd.red_csv skips it
clusters_dict = {
    col: clusters_df.loc[~clusters_df[col].isna(), "innovation code"].tolist()
    for col in clusters_df.columns[1:]
}

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

# Attach codes to data file

dosi_df["Innovation Code"] = (
    dosi_df["Innovation Name"].str.lower().map(metadata["Innovation Name"])
)
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
dosi_df["Code"] = dosi_df[
    [
        "Innovation Code",
        "Region Code",
        "Indicator Code",
        "Description Code",
        "Metric Code",
    ]
].agg("_".join, axis=1)


def FPLogValue_with_scaling(x, t0, Dt, s):
    """
    Logistic function with vertical scaling.|
    """
    return s / (1 + np.exp(-np.log(81) * (x - t0) / Dt))


st.set_page_config(layout="wide")  # add at the very top, before st.title

st.title("Clusters")

# ──────────────────────────────────────────────────────────────
# 1.  RADIO-BUTTON ROW  ────────────────────────────────────────
# ----------------------------------------------------------------
PRESET_CHOICES = [key for key in clusters_dict.keys()]  # <-- your fixed options
CUSTOM_TAG = "Custom"  # always the last radio entry
radio_labels = PRESET_CHOICES + [CUSTOM_TAG]

chosen_radio = st.radio(
    "Choose a preset or create your own:",
    radio_labels,
    horizontal=True,
)

if chosen_radio == CUSTOM_TAG:
    custom_value = st.text_input("Enter a custom name:", placeholder="e.g. My Preset")
    active_choice = custom_value.strip() or CUSTOM_TAG
else:
    active_choice = chosen_radio

cluster = active_choice

st.markdown(f"**Cluster:** `{cluster}`")

# ──────────────────────────────────────────────────────────────
# 2.  CHECKBOX GRID  (responsive 5-column layout)  ─────────────
# ----------------------------------------------------------------

ALL_INNOVATION_CODES = dosi_df[
    "Innovation Code"
].unique()  # example feature list (replace!)
prechecked = set(clusters_dict.get(active_choice, []))

NUMBER_OF_COLUMNS = 10  # Number of columns in the grid

st.subheader("Innovations included:")
cols = st.columns(NUMBER_OF_COLUMNS)
feature_states = {}

for idx, label in enumerate(ALL_INNOVATION_CODES):
    with cols[idx % NUMBER_OF_COLUMNS]:
        feature_states[label] = st.checkbox(
            next(
                key
                for key, value in metadata["Innovation Name"].items()
                if value == label
            ),
            value=label in prechecked,
        )

# ──────────────────────────────────────────────────────────────
# 3.  TOGGLE  ──────────────────────────────────────────────────
# ----------------------------------------------------------------
EARLY_ADOPTING_REGIONS_ONLY = st.toggle("Show only early-adopting regions", value=True)

# Now within clusters, only adoption


# ──────────────────────────────────────────────────────────────
# 4.  PLOTLY FIGURE  ───────────────────────────────────────────
# ----------------------------------------------------------------
def build_plot(selected_innovations, early_only: bool) -> go.Figure:
    """Dummy plot builder – drop in your real logic here."""

    cluster_innovations_df = dosi_df[
        (dosi_df["Innovation Code"].isin(selected_innovations))
        & (dosi_df["Indicator Number"] == "1.1")
    ].copy()
    cluster_innovations_summary_df = summary_df[
        (summary_df["Code"].str.split("_").str[0].isin(selected_innovations))
        & (summary_df["Indicator Number"] == "1.1")
    ]

    if early_only:
        # Only the early adopting regions?
        cluster_innovations_df = dosi_df[
            (dosi_df["Innovation Code"].isin(selected_innovations))
            & (dosi_df["Indicator Number"] == "1.1")
            & (dosi_df["Region Code"] == dosi_df["Early Adopter Code"])
        ].copy()
        cluster_innovations_summary_df = summary_df[
            (summary_df["Code"].str.split("_").str[0].isin(selected_innovations))
            & (
                summary_df["Code"].str.split("_").str[1]
                == summary_df["Code"].str.split("_").str[0].map(early_dict)
            )
            & (summary_df["Indicator Number"] == "1.1")
        ]

        # Only the early adopting regions and market shares?
        cluster_innovations_df = dosi_df[
            (dosi_df["Innovation Code"].isin(selected_innovations))
            & (dosi_df["Indicator Number"] == "1.1")
            & (dosi_df["Region Code"] == dosi_df["Early Adopter Code"])
            & (dosi_df["Metric"] == "market share")
        ].copy()
        cluster_innovations_summary_df = summary_df[
            (summary_df["Code"].str.split("_").str[0].isin(selected_innovations))
            & (
                summary_df["Code"].str.split("_").str[1]
                == summary_df["Code"].str.split("_").str[0].map(early_dict)
            )
            & (summary_df["Indicator Number"] == "1.1")
            & (summary_df["Metric"] == "market share")
        ]

    year_min = cluster_innovations_df["Year"].min() - YEAR_PADDING_FOR_PLOTTING
    year_max = cluster_innovations_df["Year"].max() + YEAR_PADDING_FOR_PLOTTING

    years_for_plotting = np.linspace(
        year_min, year_max, (year_max - year_min) + 1
    )  # 10 + 1)

    # Generate a color palette using Plotly (or you can use matplotlib or another method)
    colors = px.colors.qualitative.Set1  # Set1 is a predefined color palette

    fig = go.Figure()

    for i, code in enumerate(cluster_innovations_summary_df["Code"]):
        t0 = cluster_innovations_summary_df[
            cluster_innovations_summary_df["Code"] == code
        ]["log_t0"].iloc[0]
        Dt = cluster_innovations_summary_df[
            cluster_innovations_summary_df["Code"] == code
        ]["log_Dt"].iloc[0]
        K = cluster_innovations_summary_df[
            cluster_innovations_summary_df["Code"] == code
        ]["log_K"].iloc[0]

        innovation_name = cluster_innovations_summary_df[
            cluster_innovations_summary_df["Code"] == code
        ]["Innovation Name"].iloc[0]
        region_name = cluster_innovations_summary_df[
            cluster_innovations_summary_df["Code"] == code
        ]["Spatial Scale"].iloc[0]

        # Assign color from the color cycle
        color = colors[
            i % len(colors)
        ]  # Cycle through the colors if more codes than colors

        # Add the points trace (same color as line)
        fig.add_trace(
            go.Scatter(
                x=dosi_df[dosi_df["Code"] == code]["Year"],
                y=dosi_df[dosi_df["Code"] == code]["Value"] / K,
                mode="markers",
                name=f"{innovation_name} K-normalized data ({region_name})",  # This can be the same name to link with the line in the legend
                hovertemplate=f"{innovation_name} ({region_name})<br>{code} Point<br>Year=%{{x:.0f}}<br>value=%{{y:.2f}}<extra></extra>",  # Custom tooltip
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
                hovertemplate=f"{innovation_name} ({region_name})<br>{code}<br>Year=%{{x:.0f}}<br>Value=%{{y:.2f}}<br>Dt={Dt:.0f} t0={t0:.0f} K={K:.2f}<extra></extra>",  # Custom tooltip
            )
        )

        fig.update_layout(
            title="Cluster "
            + cluster
            + (" (Early Adopting Regions Only)" if EARLY_ADOPTING_REGIONS_ONLY else ""),
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            # hovermode='x unified'
            yaxis=dict(range=[0, 1.2]),  # Set the y-axis limits to [0, 5]
        )

        # centroid of the scatter points
        x_centroid = dosi_df.loc[dosi_df["Code"] == code, "Year"].mean()
        y_centroid = (dosi_df.loc[dosi_df["Code"] == code, "Value"] / K).mean()

        fig.add_annotation(
            x=x_centroid,
            y=y_centroid,
            text=f"{innovation_name} ({region_name})",
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
    selected_innovations=[
        label for label, checked in feature_states.items() if checked
    ],
    early_only=EARLY_ADOPTING_REGIONS_ONLY,
)
st.plotly_chart(fig, use_container_width=True)
