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


def _persist(widget_fn, *args, key, **kwargs):
    """Makes a keyed widget's value survive navigating to another page and back.

    Streamlit deletes a widget's own session_state[key] entry whenever that widget isn't
    rendered during a run (e.g. while a different page is showing) -
    https://docs.streamlit.io/develop/concepts/multipage-apps/widgets. We mirror the value
    into a second, plain (non-widget) session_state entry that this cleanup never touches,
    and re-seed the widget's key from that mirror whenever it comes back.
    """
    shadow_key = f"_persist_{key}"
    if shadow_key in st.session_state and key not in st.session_state:
        st.session_state[key] = st.session_state[shadow_key]
    value = widget_fn(*args, key=key, **kwargs)
    st.session_state[shadow_key] = value
    return value

# load data
# Get the path of the current script (inside streamlit/)
ST_DIR = Path(__file__).parent.parent
data_df = pd.read_csv(ST_DIR / '../data/results_coevolution_logistic_selected.csv')
adjustments = list(data_df['adjustment'].unique())
clusters = list(data_df['cluster'].unique())
indicators = list(data_df['indicator'].unique())
coev_metrics = [i for i in ['R_square_weighted', 'R_square', 'R_square_adj', 'time_lag'] if i in data_df.columns]
grouper_options = [i for i in ['innovation', 'ID', 'metric', 'description'] if 'i_'+i in data_df.columns]

# ──────────────────────────────────────────────────────────────
# 1.  RADIO-BUTTON ROW  ────────────────────────────────────────
# --------------------------------------------------------------

cluster_radio = _persist(
    st.radio,
    "Choose a cluster:",
    clusters,
    horizontal=True,
    key="coev_cluster_radio",
)

indicator_radio = _persist(
    st.radio,
    "Choose an innovation indicator number:",
    indicators,
    horizontal=True,
    key="coev_indicator_radio",
)

metric_radio = _persist(
    st.radio,
    "Choose a metric to display in the plots:",
    coev_metrics,
    horizontal=True,
    key="coev_metric_radio",
)

grouper_radio = _persist(
    st.radio,
    "Choose the level of analysis for pairwise comparison:",
    grouper_options,
    horizontal=True,
    key="coev_grouper_radio",
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
        # Keyed per cluster/indicator, matching the analogous pattern in 2_Clusters.py.
        spatial_states[label] = _persist(
            st.checkbox,
            label, value=False, key=f"coev_spatial_{cluster_radio}_{indicator_radio}_{label}",
        )

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
    data = data.loc[data[metric].notna()]
    
    #print('Number of innovations x axis: {}; y axis {}'.format(len(data['i_ID'].unique()), len(data['j_ID'].unique())))
    grouped = data.groupby(['i_'+grouper, 'j_'+grouper])
    data = grouped[metric].mean().unstack()
    id_i = list(grouped['i_ID'].apply(lambda x: '; '.join(list(x))))
    id_j = list(grouped['j_ID'].apply(lambda x: '; '.join(list(x))))
    
    if metric == 'time_lag':
        data[data > 50] = np.nan
        data[data < -50] = np.nan
    else:
        data[data < 0.001] = np.nan
    
    y = data.index if grouper in ['innovation', 'metric'] else list(range(len(data.index)))
    x = data.columns if grouper in ['innovation', 'metric'] else list(range(len(data.columns)))
    #assert len(id_i)==len(x) and len(id_j)==len(y), 'Aggregation failed, index lengths '+str(len(x))+' vs. '+str(len(id_i))+str(id_i)
    fig = px.imshow(data.values, x=x, y=y,
                    text_auto='.2f', aspect='auto', title=spatial,
                    color_continuous_scale='reds')
    
    if not best_fits:
        try:
            id_i = grouped['i_ID'].apply(lambda x: '; '.join(list(x))).unstack()
            id_j = grouped['j_ID'].apply(lambda x: '; '.join(list(x))).unstack()
            hoverdata = np.stack((id_j.values, id_i.values), axis=-1)
            #hoverdata = [[[data.index[j], data.columns[i]]
            #              for j in range(len(x))]
            #             for i in range(len(y))]
            hovertemplate = (
                    'X: %{customdata[0]}<br>'
                    'Y: %{customdata[1]}<br>'
                    +metric+': %{z}<extra></extra>')
            fig.update_traces(customdata=hoverdata, hovertemplate=hovertemplate)
        except IndexError:
            pass
    
    if data.mean().mean() < 1 and metric != 'time_lag':
        fig.update_coloraxes(cmin=0, cmax=1, showscale=False)
    elif metric == 'time_lag':
        fig.update_coloraxes(cmid=0, colorscale=[[0.0, "#b30000"], [0.5, "#ffffff"], [1.0, "#b30000"]])

    return fig

st.subheader("Pairwise timeseries fitting results for selected region(s):")
n_fig_cols_radio = _persist(
    st.radio,
    "Number of figures next to each other:",
    [1, 2, 3],
    horizontal=True,
    key="coev_n_fig_cols_radio",
)
adjust_radio = _persist(st.toggle, "View only best coevolutions", value=False, key="coev_adjust_radio")
if adjust_radio:
    n_best_fits = _persist(
        st.number_input,
        "Number of best fits", min_value=1, max_value=4, key="coev_n_best_fits",
    )
else:
    n_best_fits = 0

cols = st.columns(n_fig_cols_radio)
spatials_ticked = [k for k,v in spatial_states.items() if v]
for idx, label in enumerate(spatials_ticked):
    with cols[idx % n_fig_cols_radio]:
        fig = build_plot(data_df, label, cluster_radio, indicator_radio, metric_radio, grouper_radio, adjust_radio, n_best_fits)
        st.plotly_chart(fig, use_container_width=True)