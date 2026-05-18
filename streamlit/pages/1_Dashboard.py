import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Get the path of the current script (inside streamlit/)
CURRENT_DIR = Path(__file__).parent.parent

VERSION_FOR_DATA = "v28"
VERSION_FOR_FITPARAMETERS = "v28"
YEAR_PADDING_FOR_PLOTTING = 5
PATH = "../data"


st.set_page_config(layout="wide")  # add at the very top, before st.title

st.title("Within innovations")

# Choose DoSI source
source = st.radio(
    "Choose data source",
    ["Repo file", "Upload local file"]
)

# Load data
dosi_df = None
@st.cache_data
def load_data(version_data):
    return pd.read_csv(CURRENT_DIR / PATH / f"adjusted_datasets_{version_data}.csv", converters={"Indicator Number": str})
if source == "Upload local file":
    uploaded_file = st.file_uploader(
        "Upload local DoSI data file",
        type=["csv"]
    )
    if uploaded_file is not None:
        dosi_df = pd.read_csv(uploaded_file, converters={"Indicator Number": str})
else:
    version_data = st.text_input(
        "Enter DoSI data file version to be used (must be > v26)",
        value=VERSION_FOR_DATA
    )
    try:
        dosi_df = load_data(version_data)
    except FileNotFoundError:
        st.error(f"⚠️ Data version '{version_data}' not found. Please enter a valid version. Otherwise, the default {VERSION_FOR_DATA} is used.")
        dosi_df = load_data(VERSION_FOR_DATA)
if dosi_df is None:
    st.stop()

# Load logfit estimation summary
version_summary = st.text_input("Enter summary data file version to be used (must be > v26)", value=VERSION_FOR_FITPARAMETERS)
@st.cache_data
def load_summary(version_summary):
    return pd.read_csv(CURRENT_DIR / PATH / f"summary_table_{version_summary}.csv", converters={"Indicator Number": str})
try:
    summary_df = load_summary(version_summary)
except FileNotFoundError:
    st.error(f"⚠️ Data version '{version_summary}' not found. Please enter a valid version. Otherwise, the default {VERSION_FOR_FITPARAMETERS} is used.")
    summary_df = load_summary(VERSION_FOR_FITPARAMETERS)


dosi_df["Value"] = pd.to_numeric(dosi_df["Value"], errors="coerce")
dosi_df = dosi_df.dropna(subset=["Value"])
# Correct for trailing spaces in the data
dosi_df["Spatial Scale"] = dosi_df["Spatial Scale"].str.rstrip()
dosi_df["Innovation Name"] = dosi_df["Innovation Name"].str.rstrip()

innovation_names = sorted(dosi_df["Innovation Name"].unique().tolist())
indicator_codes = pd.concat([summary_df["Indicator Number"], dosi_df["Indicator Number"]]).unique().tolist()
indicator_names = pd.concat([summary_df["Indicator Name"], dosi_df["Indicator Name"]]).unique().tolist()

# create a unique identifyier per time series
group_vars = ['Description', 'Metric'] # defines one time series with 'Innovation Name', 'Spatial Scale'
dosi_df['name'] = dosi_df[group_vars[0]]
summary_df['name'] = summary_df[group_vars[0]]
for i in range(1, len(group_vars)):
    dosi_df['name'] += ' - ' + dosi_df[group_vars[i]]
    summary_df['name'] += ' - ' + summary_df[group_vars[i]]


def FPLogValue_with_scaling(x, t0, Dt, s):
    """
    Logistic function with vertical scaling.|
    """
    return s / (1 + np.exp(-np.log(81) * (x - t0) / Dt))

# Create menu

col1, col2 = st.columns(2)

with col1:
    selected_innovation = st.selectbox("Select innovation", innovation_names, index=0)

with col2:
    selected_spatial_scale = st.selectbox(
        "Select spatial scale",
        list(dosi_df.loc[dosi_df["Innovation Name"] == selected_innovation, "Spatial Scale"].unique()),
        index=0,
    )


NUMBER_OF_COLUMNS = 8  # Number of columns in the grid
st.subheader("Indicators included:")
cols = st.columns(NUMBER_OF_COLUMNS)
feature_states = {}
for idx, label in enumerate(indicator_codes):
    with cols[idx % NUMBER_OF_COLUMNS]:
        feature_states[label] = st.checkbox(
            label + " " + indicator_names[idx],
            value=label in list(dosi_df.loc[
                (dosi_df["Innovation Name"]==selected_innovation)
                & (dosi_df["Spatial Scale"]==selected_spatial_scale)
                , "Indicator Number"
            ].unique()),
        )

selected_only = st.toggle("Show only logfits of selected or new timeseries", value=True)


# ──────────────────────────────────────────────────────────────
# 4.  PLOTLY FIGURE  ───────────────────────────────────────────
# ----------------------------------------------------------------
def build_plot(inno, summary, inno_name, indicator_selection, spatial_selection) -> go.Figure:
    
    # Filter data
    innovation_df = inno.loc[
        (inno["Indicator Number"].isin([k for k, v in indicator_selection.items() if v]))
        & (inno["Spatial Scale"]==spatial_selection)
        & (inno["Innovation Name"]==inno_name)
    ].copy()
    innovation_summary_df = summary.loc[
        (summary["Indicator Number"].isin([k for k, v in indicator_selection.items() if v]))
        & (summary["Spatial Scale"]==spatial_selection)
        & (summary["Innovation Name"]==inno_name)
    ].copy()
    if selected_only:
        innovation_summary_df = innovation_summary_df.loc[
            (innovation_summary_df["select_1.1_allregions_FIN"]==1)
            | (innovation_summary_df["select_1.1_allregions_FIN"].isna())]

    year_min = innovation_df["Year"].min() - YEAR_PADDING_FOR_PLOTTING
    year_max = innovation_df["Year"].max() + YEAR_PADDING_FOR_PLOTTING
    #if not isinstance(year_min, int): year_min = 2000 - YEAR_PADDING_FOR_PLOTTING
    #if not isinstance(year_max, int): year_max = 2000 + YEAR_PADDING_FOR_PLOTTING

    years_for_plotting = np.linspace(
        year_min, year_max, (year_max - year_min) + 1
    )  # 10 + 1)

    # Generate a color palette using Plotly (or you can use matplotlib or another method)
    colors = px.colors.qualitative.Set1  # Set1 is a predefined color palette

    fig = go.Figure()

    for i, code in enumerate(innovation_df["Indicator Number"].unique()):

        # Assign color from the color cycle
        color = colors[
            i % len(colors)
        ]  # Cycle through the colors if more codes than colors
        
        K_dict = {name: 1 for name in innovation_df["name"].unique()}
            
        # Search for logfit parameters and add curve to plot, if available
        if code in list(innovation_summary_df["Indicator Number"].unique()):
            groups = innovation_summary_df[innovation_summary_df["Indicator Number"] == code].groupby(group_vars)
            for j, (metric, timeseries) in enumerate(groups):
                t0 = timeseries["log_t0"].iloc[0]
                Dt = timeseries["log_Dt"].iloc[0]
                K = timeseries["log_K"].iloc[0]

                fig.add_trace(
                    go.Scatter(
                        x=years_for_plotting,
                        y=FPLogValue_with_scaling(years_for_plotting, t0, Dt, K) / K,
                        mode="lines",
                        name=f"{code} - {metric}",  # Legend label
                        showlegend=False,
                        line=dict(color=color, width=2),
                        opacity=1-j*0.8/len(groups),
                        hovertemplate=f"""{code} <br>{code} ({metric[0]}: {metric[1]}) <br>Year=%{{x:.0f}}<br>Value=%{{y:.2f}}<br>Dt={Dt:.0f} t0={t0:.0f} K={K:.2f}<extra></extra>""",  # Custom tooltip
                    )
                )
                
                fig.update_layout(
                    yaxis_title="K-normalized value",
                    yaxis=dict(range=[0, 1.2])
                )
                K_dict[timeseries["name"].iloc[0]] = K
        
        groups = innovation_df[innovation_df["Indicator Number"] == code].groupby(group_vars)
        for j, (metric, timeseries) in enumerate(groups):

            # Add the points trace (same color as line)
            fig.add_trace(
                go.Scatter(
                    x=timeseries["Year"],
                    y=timeseries["Value"] / K_dict[timeseries["name"].iloc[0]],
                    mode="markers",
                    name=f"{code} ({metric[0]}: {metric[1]})",  # This can be the same name to link with the line in the legend
                    hovertemplate=f"""{code} ({metric[0]}: {metric[1]}) <br>{code} Point<br>Year=%{{x:.0f}}<br>value=%{{y:.2f}}<extra></extra>""",  # Custom tooltip
                    marker=dict(size=8, color=color, line=dict(width=1, color="#777777")),
                    opacity=1-j*0.8/len(groups),
                )
            )

            # centroid of the scatter points
            x_centroid = timeseries["Year"].mean()
            y_centroid = (timeseries["Value"] / K_dict[timeseries["name"].iloc[0]]).mean()
            fig.add_annotation(
                x=x_centroid,
                y=y_centroid,
                text=str(code),
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(color=color),  # label colour = line colour
            )

    fig.update_layout(
        title="Innovation " + inno_name + " in " + spatial_selection,
        xaxis_title="Year"
    )
    fig.update_layout(showlegend=False)
    fig.update_layout(height=900)  # make the plot taller
    return fig


fig = build_plot(dosi_df, summary_df, selected_innovation, feature_states, selected_spatial_scale)
st.plotly_chart(fig, width='stretch')
