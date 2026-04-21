import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


st.set_page_config(layout="wide")  # add at the very top, before st.title

st.title("Hubs and bridges")

# load data
# Get the path of the current script (inside streamlit/)
ST_DIR = Path(__file__).parent.parent
VERSION_FOR_SUMMARY_READING = "v28"
VERSION_CLUSTER_DF = "innovation_list_HWLclusters_v3.0.xlsx"

version = st.text_input("Enter summary data file version to be used (must be > v26)", value=VERSION_FOR_SUMMARY_READING)

# Load logfit estimation summary
@st.cache_data
def load_data(version):
    print(ST_DIR)
    return pd.read_csv(ST_DIR / f"../data/summary_table_{version}.csv", converters={"Indicator Number": str})
try:
    data_df = load_data(version)
except FileNotFoundError:
    st.error(f"⚠️ Data version '{version}' not found. Please enter a valid version. Otherwise, the default {VERSION_FOR_SUMMARY_READING} is used.")
    data_df = load_data(VERSION_FOR_SUMMARY_READING)
# filter for chosen time series (and nan)
data_df = data_df.loc[(data_df['select_1.1_allregions_FIN']!=0) | (data_df['select_1.1_allregions_FIN'].isna())]

# Load cluster assignment
clusters_df = pd.read_excel(ST_DIR / f'../data/{VERSION_CLUSTER_DF}', sheet_name=0)
# Take only innovations with time series
clusters_df = clusters_df.loc[clusters_df['timeseries']==1]

analysis = [x for x in ['hub', 'bridge'] if x in clusters_df.columns]
clusters = [c for c in ['digital', 'health', 'prosumer', 'sufficiency'] if c in clusters_df.columns]
indicators = list(data_df['Indicator Number'].unique())
metrics = [x for x in ['slope_log', 'log_t0', 'log_Dt', 'log_K', 'log_r2'] if x in data_df.columns]

# Map clusters
clusters_dict = {
    c: clusters_df.loc[~clusters_df[c].isna(), "innovation_name"].str.lower().tolist()
    for c in clusters}
for c, innos in clusters_dict.items():
    assert len(innos) > 0, c+' cluster has no innovations assigned'
    data_df[c] = 0
    data_df.loc[data_df['Innovation Name'].isin(innos), c] = 1

# ──────────────────────────────────────────────────────────────
# 1.  RADIO-BUTTON ROW  ────────────────────────────────────────
# --------------------------------------------------------------

analysis_radio = st.radio(
    "Choose hubs or bridges analysis:",
    analysis,
    horizontal=True,
)

cluster_radio = st.radio(
    "Choose a cluster:",
    ['All'] + clusters,
    horizontal=True,
)

indicator_radio = st.radio(
    "Choose an innovation indicator number:",
    indicators,
    horizontal=True,
)

metric_radio = st.radio(
    "Choose a metric to display in the plots:",
    metrics,
    horizontal=True,
)

# ──────────────────────────────────────────────────────────────
# 2.  CHECKBOX GRID  (responsive X-column layout)  ─────────────
# --------------------------------------------------------------

# filter data for cluster and innovation number
# also choose only spatial scales where at least one hub/bridge exists
if cluster_radio in clusters:
    mask = (data_df[cluster_radio]==1) \
        & (data_df['Indicator Number']==indicator_radio) \
        & ((data_df['Innovation Name'].str.lower()).isin(
            clusters_df.loc[clusters_df[analysis_radio]==1, 'innovation_name'].str.lower()))
else:
    mask = (data_df['Indicator Number']==indicator_radio) \
        & ((data_df['Innovation Name'].str.lower()).isin(
            clusters_df.loc[clusters_df[analysis_radio]==1, 'innovation_name'].str.lower()))

available_spatial = sorted(list(data_df.loc[mask, 'Spatial Scale'].unique()))

N_COLS_SPATIAL = 8
st.subheader("Spatial scales to display:")
cols_spatial = st.columns(N_COLS_SPATIAL)
spatial_states = {}
for idx, label in enumerate(available_spatial):
    with cols_spatial[idx % N_COLS_SPATIAL]:
        spatial_states[label] = st.checkbox(label, value=True, key='s_'+label)

#N_COLS_INNOS = 6
#st.subheader(analysis_radio+" innovations to illustrate:")
#cols_innos = st.columns(N_COLS_INNOS)
#inno_states = {}
#for idx, label in enumerate(
#        sorted(list(data_df.loc[mask & data_df['Spatial Scale'].isin(
#            [s for s,state in spatial_states.items() if state==True]
#            ), 'Innovation Name'].unique()))
#        ):
#    with cols_innos[idx % N_COLS_INNOS]:
#        inno_states[label] = st.checkbox(label, value=True, key='i_'+label)

# ──────────────────────────────────────────────────────────────
# 3.  PLOTLY FIGURE  ───────────────────────────────────────────
# --------------------------------------------------------------

def build_plot(df, analysis, cluster, indicator, metric, x_min, x_max, y_min, y_max) -> go.Figure:
    
    # Filter for cluster and innovation indicator
    if cluster in clusters:
        mask = (df[cluster]==1) & (df['Indicator Number']==indicator)
    else:
        mask = (df['Indicator Number']==indicator)
    data = df.loc[mask].copy()
    # Filter for spatial scales ticked (only those with hub/bridge innos are available)
    data = data.loc[data['Spatial Scale'].isin([s for s,state in spatial_states.items() if state==True])]
    
    # Find hubs or bridges and save as marker shape
    analysis_innos = list(clusters_df.loc[clusters_df[analysis]==1, 'innovation_name'].str.lower())
    #analysis_innos = [i for i in analysis_innos if i in inno_states.keys() and inno_states[i]==True]
    data['marker'] = 'circle'
    data.loc[(data['Innovation Name'].str.lower()).isin(analysis_innos), 'marker'] = 'cross'
    
    # calculate x-axis reference point and time lag per spatial scale
    x_metric = "log_t0"
    xref_dict = data.loc[(data['Innovation Name'].str.lower()).isin(analysis_innos)
                        ].groupby('Spatial Scale')[x_metric].mean()
    data['xref'] = data['Spatial Scale'].map(xref_dict)
    data["diff"] = data[x_metric] - data['xref']

    fig = px.scatter(
        data,
        x="diff",
        y=metric,
        color="Spatial Scale",
        symbol="marker",
        symbol_map={m: m for m in data["marker"].dropna().unique()},
        hover_data={
            "Innovation Name": True,
            "Spatial Scale": True,
            "marker": True
        },
        labels={"diff": f"time lag to {analysis}s", metric: metric}
    )
    
    # Apply axis limits if provided
    if x_min is not None or x_max is not None:
        fig.update_xaxes(range=[x_min, x_max])
    if y_min is not None or y_max is not None:
        fig.update_yaxes(range=[y_min, y_max])

    #fig.update_layout(legend_title="Spatial Scale", hovermode="closest")
    fig.update_layout(
        legend=dict(itemclick="toggleothers", itemdoubleclick="toggle"),
        legend_title=f"Markers: cross={analysis}, circle=others")
    
    return fig

x_min, x_max = st.slider("X range", -50, 200, (-10, 50))
y_min, y_max = st.slider("Y range", -1, 50, (-1, 5))

fig = build_plot(data_df, analysis_radio, cluster_radio, indicator_radio, metric_radio, x_min, x_max, y_min, y_max)
st.plotly_chart(fig, width='stretch')