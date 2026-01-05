import pandas as pd
import numpy as np
import streamlit as st
#import matplotlib.pyplot as plt
#import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


st.set_page_config(layout="wide")  # add at the very top, before st.title

st.title("Coevolutions")

# load data
# Get the path of the current script (inside streamlit/)
ST_DIR = Path(__file__).parent.parent
data_df = pd.read_csv(ST_DIR / '../data/results_coevolution_logistic.csv')
adjustments = list(data_df['adjustment'].unique())
clusters = list(data_df['cluster'].unique())
indicators = list(data_df['indicator'].unique())
coev_metrics = [i for i in ['R_square', 'R_square_adj'] if i in data_df.columns]
grouper_options = [i for i in ['ID', 'innovation', 'metric', 'description'] if 'i_'+i in data_df.columns]

# ──────────────────────────────────────────────────────────────
# 1.  RADIO-BUTTON ROW  ────────────────────────────────────────
# --------------------------------------------------------------

cluster_radio = st.radio(
    "Choose a cluster:",
    clusters,
    horizontal=True,
)

indicator_radio = st.radio(
    "Choose an innovation indicator number:",
    indicators,
    horizontal=True,
)

metric_radio = st.radio(
    "Choose a metric to display in the plots:",
    coev_metrics,
    horizontal=True,
)

grouper_radio = st.radio(
    "Choose the level of analysis for pairwise comparison:",
    grouper_options,
    horizontal=True,
)

# ──────────────────────────────────────────────────────────────
# 2.  CHECKBOX GRID  (responsive 5-column layout)  ─────────────
# --------------------------------------------------------------

# filter data
if cluster_radio in clusters:
    mask = (data_df['cluster']==cluster_radio) & (data_df['indicator']==indicator_radio)
else:
    mask = data_df['indicator']==indicator_radio
spatials = sorted(list(data_df.loc[mask, 'spatial'].unique()))

N_COLS_SPATIAL = 8
st.subheader("Spatial scales to display:")
cols = st.columns(N_COLS_SPATIAL)
spatial_states = {}
for idx, label in enumerate(spatials):
    with cols[idx % N_COLS_SPATIAL]:
        spatial_states[label] = st.checkbox(label, value=False)

# ──────────────────────────────────────────────────────────────
# 3.  PLOTLY FIGURE  ───────────────────────────────────────────
# --------------------------------------------------------------

def build_plot(df, spatial, cluster, indicator, metric, grouper, best_fits, n_best_fits) -> go.Figure:
    
    if cluster in clusters:
        mask = (df['spatial']==spatial) & (df['cluster']==cluster) & (df['indicator']==indicator)
    else:
        mask = (df['spatial']==spatial) & (df['indicator']==indicator)
    data = df.loc[mask].copy()
    if best_fits:
        data = data.sort_values(metric, ascending=False)\
                   .groupby(['cluster', 'indicator', 'spatial', 'i_innovation'])\
                   .apply(lambda g: g.loc[g['i_innovation']!=g['j_innovation']].head(n_best_fits))\
                   .reset_index(drop=True)
    #data[metric] = data[metric].fillna(0).clip(lower=0)
    data = data.groupby(['i_'+grouper, 'j_'+grouper])[metric].mean().unstack()
    
    y = data.index if grouper in ['innovation', 'metric'] else list(range(len(data.index)))
    x = data.columns if grouper in ['innovation', 'metric'] else list(range(len(data.columns)))
    fig = px.imshow(data.values, x=x, y=y,
                    text_auto='.2f', aspect='auto', title=spatial,
                    color_continuous_scale='reds')
    
    if not best_fits:
        hoverdata = [[[data.index[j], data.columns[i]]
                      for j in range(len(x))]
                     for i in range(len(y))]
        hovertemplate = (
                'X: %{customdata[0]}<br>'
                'Y: %{customdata[1]}<br>'
                'Value: %{z}<extra></extra>')
        fig.update_traces(customdata=hoverdata, hovertemplate=hovertemplate)
    
    if data.mean().mean() < 1:
        fig.update_coloraxes(cmin=0, cmax=1, showscale=False)
    #fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig

st.subheader("Pairwise timeseries fitting results for selected region(s):")
n_fig_cols_radio = st.radio(
    "Number of figures next to each other:",
    [1, 2, 3],
    horizontal=True,
)
adjust_radio = st.toggle("View only best fits", value=False)
if adjust_radio:
    n_best_fits = st.number_input("Number of best fits", min_value=1, max_value=3)
else:
    n_best_fits = 0

cols = st.columns(n_fig_cols_radio)
spatials_ticked = [k for k,v in spatial_states.items() if v]
for idx, label in enumerate(spatials_ticked):
    with cols[idx % n_fig_cols_radio]:
        fig = build_plot(data_df, label, cluster_radio, indicator_radio, metric_radio, grouper_radio, adjust_radio, n_best_fits)
        st.plotly_chart(fig, use_container_width=True)