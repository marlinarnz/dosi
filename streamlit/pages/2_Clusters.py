import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import nbformat
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


st.set_page_config(layout="wide")  # add at the very top, before st.title

st.title("Clusters")


# Get the path of the current script (inside streamlit/)
CURRENT_DIR = Path(__file__).parent.parent
PATH = "../data"
VERSION_FOR_METADATA = "v26"
YEAR_PADDING_FOR_PLOTTING = 10


files = [entry for entry in os.listdir(CURRENT_DIR / PATH)
         if entry.startswith('summary_table_v')
         and entry.endswith('.csv')
         and os.path.isfile(CURRENT_DIR / PATH / entry)]
version_summary = max([int(file[-6:-4]) for file in files if len(file.split('_')[-1])==7])

files = [entry for entry in os.listdir(CURRENT_DIR / PATH)
         if entry.startswith('adjusted_datasets_v')
         and entry.endswith('.csv')
         and os.path.isfile(CURRENT_DIR / PATH / entry)]
version_data = max([int(file[-6:-4]) for file in files if len(file.split('_')[-1])==7])

fn_data = CURRENT_DIR / PATH / f"adjusted_datasets_v{version_data}.csv"
fn_data_hatch = CURRENT_DIR / PATH / f"hatch/hatch_data_dosi_format.csv"
fn_summary = CURRENT_DIR / PATH / f"summary_table_v{version_summary}.csv"
fn_summary_hatch = CURRENT_DIR / PATH / f"summary_table_HATCH_v27.csv"
fn_clusters = CURRENT_DIR / PATH / "innovation_list_HWLclusters_v3.0.xlsx"
fn_early = CURRENT_DIR / PATH / "EarlyAdopterRegions_perInnovation_21March.csv"  # Early Adopting regions
fn_early_hatch = CURRENT_DIR / PATH / "hatch/hatch_early_dict.csv"  # Early Adopting regions
fn_metadata = CURRENT_DIR / PATH / f"metadata_master_{VERSION_FOR_METADATA}.xlsx"


# time series data
dosi_df = pd.concat([
        pd.read_csv(fn_data, converters={"Indicator Number": str}),
        pd.read_csv(fn_data_hatch, converters={"Indicator Number": str}),
])
dosi_df["Value"] = pd.to_numeric(dosi_df["Value"], errors="coerce")
dosi_df = dosi_df.dropna(subset=["Value"])
# Correct for trailing spaces in the data
dosi_df["Spatial Scale"] = dosi_df["Spatial Scale"].str.rstrip()
dosi_df["Innovation Name"] = dosi_df["Innovation Name"].str.rstrip()

# Homologate to lowercase: source spreadsheets used inconsistent casing (e.g. "E-commerce"
# vs "e-commerce") for the same innovation, which would otherwise split it into two names
# when matching dosi_df rows to summary_df rows via the 'name' identifier built below.
dosi_df["Innovation Name"] = dosi_df["Innovation Name"].str.lower()


# Logfit estimates
summary_df = pd.concat([
        pd.read_csv(fn_summary, converters={"Indicator Number": str}),
        pd.read_csv(fn_summary_hatch, converters={"Indicator Number": str}),
])
summary_df["Innovation Name"] = summary_df["Innovation Name"].str.rstrip().str.lower()


# early-adopting regions
early_df = pd.concat([
    pd.read_csv(fn_early, usecols=[0, 1]),
    pd.read_csv(fn_early_hatch, usecols=[0, 1])
])
early_dict = dict(zip(early_df.iloc[:, 0], early_df.iloc[:, 1]))


# Metadata / codes (OLD VERSION)
def convert_to_three_digit_notation(s):
    return re.sub(r"([a-zA-Z])(\d+)", lambda m: f"{m.group(1)}{int(m.group(2)):03}", s)
def read_metadata_table(fn, columns):
    df = pd.read_excel(fn, usecols=columns, dtype=str).dropna().reset_index(drop=True)
    df.iloc[:, 1] = df.iloc[:, 1].apply(convert_to_three_digit_notation)
    return df.set_index(df.columns[0])[df.columns[1]].to_dict()
metadata = dict()
metadata["Innovation Name"] = read_metadata_table(fn_metadata, "A,D")
metadata["Spatial Scale"] = read_metadata_table(fn_metadata, "G,I")
metadata["Indicator Number"] = read_metadata_table(fn_metadata, "L,O")  # Column M is the indicator name. Superfluous because maps 1-1 on indicator number
metadata["Description"] = read_metadata_table(fn_metadata, "R,S")
metadata["Metric"] = read_metadata_table(fn_metadata, "V,W")
for key, nested_dict in metadata.items():
    if isinstance(nested_dict, dict):  # Ensure the value is a dictionary
        metadata[key] = {
            k.lower() if isinstance(k, str) else k: v for k, v in nested_dict.items()
        }


# Clusters
clusters = ['digital', 'prosumer', 'health', 'sufficiency']
clusters_df = pd.read_excel(fn_clusters)
# Update metadata innovation name label with cluster assignment file
clusters_names = clusters_df.set_index("innovation_name")["innovation_label"].to_dict()
metadata["Innovation Name"].update(clusters_names)
# Generate cluster-innovation mapping
clusters_dict = {}
for c in clusters:
    clusters_dict[c] = list(clusters_df.loc[
        (clusters_df[c]==1) & (clusters_df['timeseries']==1), 'innovation_label'
    ])


# Attach codes to data file (OLD VERSION)
dosi_df["Innovation Code"] = dosi_df["Innovation Name"].str.lower().map(metadata["Innovation Name"])
dosi_df["Region Code"] = dosi_df["Spatial Scale"].str.lower().map(metadata["Spatial Scale"])
dosi_df["Early Adopter Code"] = dosi_df["Innovation Code"].map(early_dict)
dosi_df["Indicator Code"] = dosi_df["Indicator Number"].str.lower().map(metadata["Indicator Number"])
dosi_df["Description Code"] = dosi_df["Description"].str.lower().map(metadata["Description"]).fillna("")
dosi_df["Metric Code"] = dosi_df["Metric"].str.lower().map(metadata["Metric"]).fillna("")
code_cols = [
        "Innovation Code",
        "Region Code",
        "Indicator Code",
        "Description Code",
        "Metric Code",
]
dosi_df[code_cols] = dosi_df[code_cols].astype(str)
dosi_df["Code"] = dosi_df[code_cols].agg("_".join, axis=1)

# create a unique identifyier per time series
group_vars = ['Innovation Name', 'Spatial Scale', 'Description', 'Metric'] # defines one time series
sep = ' - '
dosi_df['name'] = dosi_df[group_vars[0]]
summary_df['name'] = summary_df[group_vars[0]]
for i in range(1, len(group_vars)):
    dosi_df['name'] += sep + dosi_df[group_vars[i]]
    summary_df['name'] += sep + summary_df[group_vars[i]]


def FPLogValue_with_scaling(x, t0, Dt, s):
    """
    Logistic function with vertical scaling.|
    """
    return s / (1 + np.exp(-np.log(81) * (x - t0) / Dt))


# ──────────────────────────────────────────────────────────────
# 1. Clusters: RADIO-BUTTON ROW
# ----------------------------------------------------------------

active_choice = st.radio(
    "Choose a cluster:",
    clusters,
    horizontal=True,
)
cluster = active_choice

# ──────────────────────────────────────────────────────────────
# 2. Innovations: CHECKBOX GRID  (responsive column layout)
# ----------------------------------------------------------------

#ALL_INNOVATION_CODES = dosi_df["Innovation Code"].unique()
ALL_INNOVATION_CODES = set(clusters_dict.get(active_choice, []))
prechecked = set(clusters_dict.get(active_choice, []))

NUMBER_OF_COLUMNS = 5  # Number of columns in the grid

st.subheader("Innovations included:")
cols = st.columns(NUMBER_OF_COLUMNS)
feature_states = {}

for idx, label in enumerate(ALL_INNOVATION_CODES):
    display_name = next(
        (key for key, value in clusters_names.items() if value == label),
        None,  # default if not found
    )
    if display_name is None:
        # Skip labels that don't exist in metadata
        continue
    with cols[idx % NUMBER_OF_COLUMNS]:
        feature_states[label] = st.checkbox(
            display_name,
            value=label in prechecked,
        )

indicator_radio = st.radio(
    "Choose an innovation indicator number:",
    list(dosi_df['Indicator Number'].unique()),
    horizontal=True,
)

# ──────────────────────────────────────────────────────────────
# 3. Countries: CHECKBOX GRID  (responsive column layout)
# ----------------------------------------------------------------

spatials = sorted(list(dosi_df.loc[
    (dosi_df['Innovation Code'].isin(prechecked)) & (dosi_df['Indicator Number']==indicator_radio),
    'Spatial Scale'].unique()))

N_COLS_SPATIAL = 8
st.subheader("Spatial scales to display:")
cols = st.columns(N_COLS_SPATIAL)

spatial_states = {}
for idx, label in enumerate(spatials):
    with cols[idx % N_COLS_SPATIAL]:
        spatial_states[label] = st.checkbox(label, value=False if idx>1 else True)

align_t0 = st.toggle("Align t0?", value=False)


# ──────────────────────────────────────────────────────────────
# 4.  PLOTLY FIGURE  ───────────────────────────────────────────
# ----------------------------------------------------------------

def build_plot(
    selected_innovations, countries_selected: list, indicator: str
):

    cluster_innovations_df = dosi_df[
        (dosi_df["Innovation Code"].isin(selected_innovations))
        & (dosi_df["Indicator Number"] == indicator)
        & (dosi_df["Spatial Scale"].isin(countries_selected))
    ].copy()
    cluster_innovations_summary_df = summary_df[
        (summary_df["Code"].str.split("_").str[0].isin(selected_innovations))
        & (summary_df["Indicator Number"] == indicator)
        & (summary_df["Spatial Scale"].isin(countries_selected))
    ].copy()
    
    if len(cluster_innovations_df) > 0:

        year_min = cluster_innovations_df["Year"].min() - YEAR_PADDING_FOR_PLOTTING
        year_max = cluster_innovations_df["Year"].max() + YEAR_PADDING_FOR_PLOTTING

        years_for_plotting = np.linspace(
            year_min, year_max, (year_max - year_min) + 1
        )  # 10 + 1)

        # Generate a color palette using Plotly (or you can use matplotlib or another method)
        colors = px.colors.qualitative.Set1  # Set1 is a predefined color palette

        fig = go.Figure()

        for i, code in enumerate(cluster_innovations_summary_df["name"]):
            t0 = cluster_innovations_summary_df[
                cluster_innovations_summary_df["name"] == code
            ]["log_t0"].iloc[0]
            Dt = cluster_innovations_summary_df[
                cluster_innovations_summary_df["name"] == code
            ]["log_Dt"].iloc[0]
            K = cluster_innovations_summary_df[
                cluster_innovations_summary_df["name"] == code
            ]["log_K"].iloc[0]

            innovation_name = cluster_innovations_summary_df[
                cluster_innovations_summary_df["name"] == code
            ]["Innovation Name"].iloc[0]
            region_name = cluster_innovations_summary_df[
                cluster_innovations_summary_df["name"] == code
            ]["Spatial Scale"].iloc[0]
            metric_name = cluster_innovations_summary_df[
                cluster_innovations_summary_df["name"] == code
            ]["Metric"].iloc[0]
            description_name = cluster_innovations_summary_df[
                cluster_innovations_summary_df["name"] == code
            ]["Description"].iloc[0]

            if Dt < 0:
                print(f"Dt < 0 for {code}. Plotting mirrored curve.")

            # Assign color from the color cycle
            color = colors[
                i % len(colors)
            ]  # Cycle through the colors if more codes than colors

            # Add the points trace (same color as line)
            fig.add_trace(
                go.Scatter(
                    x=dosi_df[dosi_df["name"] == code]["Year"] - (t0 if align_t0 else 0),
                    y=(1 / K if Dt < 0 else 0)
                    + (-1 if Dt < 0 else 1) * dosi_df[dosi_df["name"] == code]["Value"] / K,
                    mode="markers",
                    name=f"{innovation_name} K-normalized data ({region_name})",  # This can be the same name to link with the line in the legend
                    hovertemplate=f"{innovation_name} ({region_name})<br>Description: {description_name}<br>Metric: {metric_name}<br>{code} Point<br>Year=%{{x:.0f}}<br>value=%{{y:.2f}}<extra></extra>",  # Custom tooltip
                    marker=dict(size=8, color=color),  # Same color for points as the line
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=years_for_plotting - (t0 if align_t0 else 0),
                    y=(1 / K if Dt < 0 else 0)
                    + (-1 if Dt < 0 else 1)
                    * FPLogValue_with_scaling(years_for_plotting, t0, Dt, K)
                    / K,
                    mode="lines",
                    name=code,  # Legend label
                    showlegend=False,
                    line=dict(color=color, width=2),
                    hovertemplate=f"{innovation_name} ({region_name})<br>{code}<br>Year=%{{x:.0f}}<br>Value=%{{y:.2f}}<br>Dt={Dt:.0f} t0={t0:.0f} K={K:.2f}<extra></extra>",  # Custom tooltip
                )
            )

            fig.update_layout(
                title="Cluster " + cluster,
                xaxis_title="X Axis",
                yaxis_title="Y Axis",
                # hovermode='x unified'
                yaxis=dict(range=[0, 1.2]),  # Set the y-axis limits to [0, 5]
            )

            # centroid of the scatter points
            x_centroid = dosi_df.loc[dosi_df["name"] == code, "Year"].mean()
            y_centroid = (1 / K if Dt < 0 else 0) + (-1 if Dt < 0 else 1) * (
                dosi_df.loc[dosi_df["name"] == code, "Value"] / K
            ).mean()

            fig.add_annotation(
                x=x_centroid - (t0 if align_t0 else 0),
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
    selected_innovations=[label for label, checked in feature_states.items() if checked],
    countries_selected=[label for label, checked in spatial_states.items() if checked],
    indicator=indicator_radio
)
try:
    st.plotly_chart(fig, width='stretch')
except:
    st.error('Choose a country with data available')
