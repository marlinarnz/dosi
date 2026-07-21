import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Get the path of the current script (inside streamlit/)
CURRENT_DIR = Path(__file__).parent.parent
YEAR_PADDING_FOR_PLOTTING = 5
PATH = "../data"


st.set_page_config(layout="wide")  # add at the very top, before st.title

st.title("Within innovations")


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


# Choose DoSI source
source = _persist(
    st.radio,
    "Choose data source",
    ["Repo file", "Upload local file"],
    key="dashboard_source",
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
    files = [entry for entry in os.listdir(CURRENT_DIR / PATH)
             if entry.startswith('adjusted_datasets_v')
             and entry.endswith('.csv')
             and os.path.isfile(CURRENT_DIR / PATH / entry)]
    version_data = max([int(file[-6:-4]) for file in files if len(file.split('_')[-1])==7])
    try:
        dosi_df = load_data('v'+str(version_data))
    except FileNotFoundError:
        st.error(f"⚠️ Data version '{version_data}' not found. Please make sure the most recent data file ends with a 'v' followed by a two-digit number. Otherwise, the default 'v30' is used.")
        dosi_df = load_data('v30')
if dosi_df is None:
    st.stop()

# Load logfit estimation summary
files = [entry for entry in os.listdir(CURRENT_DIR / PATH)
         if entry.startswith('summary_table_v')
         and entry.endswith('.csv')
         and os.path.isfile(CURRENT_DIR / PATH / entry)]
version_summary = max([int(file[-6:-4]) for file in files if len(file.split('_')[-1])==7])
try:
    summary_df = pd.read_csv(CURRENT_DIR / PATH / f"summary_table_v{version_summary}.csv", converters={"Indicator Number": str})
except FileNotFoundError:
    st.error(f"⚠️ Summary version '{version_summary}' not found. Please make sure the most recent logfit estimation file ends with a 'v' followed by a two-digit number.")
    st.stop()


dosi_df["Value"] = pd.to_numeric(dosi_df["Value"], errors="coerce")
dosi_df = dosi_df.dropna(subset=["Value"])
# Correct for trailing spaces in the data
dosi_df["Spatial Scale"] = dosi_df["Spatial Scale"].str.rstrip()
dosi_df["Innovation Name"] = dosi_df["Innovation Name"].str.rstrip()
summary_df["Innovation Name"] = summary_df["Innovation Name"].str.rstrip()

# Homologate innovation names to lowercase: the source spreadsheets used inconsistent
# casing (e.g. "E-commerce" vs "e-commerce") for the same innovation, which otherwise
# shows up as separate, incomplete entries in the selector below.
raw_dosi_innovation_names = dosi_df["Innovation Name"]
raw_summary_innovation_names = summary_df["Innovation Name"]
dosi_df["Innovation Name"] = raw_dosi_innovation_names.str.lower()
summary_df["Innovation Name"] = raw_summary_innovation_names.str.lower()

sorted_inno_index = dosi_df.sort_values("Innovation Name", key=lambda col: col.str.lower()).index
innovation_names = dosi_df.loc[sorted_inno_index, "Innovation Name"].drop_duplicates().tolist()
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


# Points sharing the same year (data is annual, so no window is needed there) count as
# "overlapping" if their y-values are within this fraction of a series' own magnitude -
# a per-series tolerance, so a sparse/tiny series isn't swamped by a tolerance sized for
# a much larger series sharing the same plot. Overlapping points get a combined tooltip
# with the hovered point's own detail first (bolded) and the others condensed to one line.
OVERLAP_LOOSE_Y_TOL_FRAC = 0.02

# Within an already-overlapping pair, if the two y-values are ALSO within this fraction
# of the smaller of the two - i.e. they're effectively duplicates, not just "close on the
# chart" - show that point in full instead of condensing it to one line.
OVERLAP_TIGHT_Y_TOL_FRAC = 0.005

# Marker fill alpha for scatter points, capped here (and used as the flat value in
# 2_Clusters.py, which has no per-group opacity tiers) so overlapping points blend
# into a visibly different color rather than one fully occluding another.
MARKER_MAX_OPACITY = 0.6


def _series_epsilon(values, frac=OVERLAP_LOOSE_Y_TOL_FRAC, floor=1e-9):
    """Tolerance derived from a series' own magnitude, so sparse/tiny series aren't
    swamped by a tolerance sized for a much larger series sharing the same plot."""
    values = np.abs(np.asarray(values, dtype=float))
    return max(frac * values.max(), floor) if values.size else floor


def _build_overlap_aware_hovertext(x, y, own_html, other_html, eps_y):
    """Combines tooltips for points that share the same year and fall within a
    scale-dependent tolerance window (eps_y, per series) of each other, so near-identical
    points don't silently hide one another. Never moves a point (no jitter) - this only
    changes what the tooltip shows. The hovered point's own detail is always listed first
    and bolded. Among the rest, pairs that are within OVERLAP_TIGHT_Y_TOL_FRAC of each
    other (true near-duplicates, not just "close on this chart's scale") are shown in
    full; anything looser is condensed into a one-line summary."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    hovertext = []
    for i in range(n):
        tight, loose = [], []
        for j in range(n):
            if j == i or x[j] != x[i]:
                continue
            if abs(y[j] - y[i]) > max(eps_y[i], eps_y[j]):
                continue  # not overlapping at all
            tight_tol = OVERLAP_TIGHT_Y_TOL_FRAC * min(abs(y[i]), abs(y[j]))
            if abs(y[j] - y[i]) <= tight_tol:
                tight.append(own_html[j])
            else:
                loose.append(other_html[j])

        if not tight and not loose:
            hovertext.append(own_html[i])
            continue
        blocks = [f"<b>{own_html[i]}</b>"] + tight
        if loose:
            blocks.append(f"+{len(loose)} overlapping point(s):<br>" + "<br>".join(loose))
        hovertext.append("<br><br>".join(blocks))
    return hovertext


# ──────────────────────────────────────────────────────────────
# Casing homologation report: which merged innovations have data
# split across their old casings and would benefit from a unified refit
# ──────────────────────────────────────────────────────────────

casing_map = {}
for raw in pd.concat([raw_dosi_innovation_names, raw_summary_innovation_names]).dropna().unique():
    casing_map.setdefault(raw.lower(), set()).add(raw)
merged_innovations = {k: sorted(v) for k, v in casing_map.items() if len(v) > 1}

with st.expander(
    f"ℹ️ Innovation names homologated to lowercase ({len(merged_innovations)} innovations affected)",
    expanded=False,
):
    if not merged_innovations:
        st.success("No casing duplicates found in the currently loaded data.")
    else:
        st.write(
            "These innovations appeared under multiple casings in the source data "
            "(e.g. from different source spreadsheets) and are merged below under a single lowercase name:"
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"Innovation (homologated)": k, "Original casings found": ", ".join(v)}
                    for k, v in sorted(merged_innovations.items())
                ]
            ),
            hide_index=True,
        )

        summary_df["_raw_innovation_name"] = raw_summary_innovation_names
        group_cols = ["Innovation Name", "Spatial Scale", "Indicator Number", "Description", "Metric"]
        flag_rows = []
        for group_key, group in summary_df[
            summary_df["Innovation Name"].isin(merged_innovations.keys())
        ].groupby(group_cols):
            variants = group["_raw_innovation_name"].dropna().unique()
            if len(variants) > 1:
                for _, row in group.iterrows():
                    flag_rows.append(
                        {
                            "Innovation (homologated)": group_key[0],
                            "Spatial Scale": group_key[1],
                            "Indicator Number": group_key[2],
                            "Description": group_key[3],
                            "Metric": group_key[4],
                            "Original casing": row["_raw_innovation_name"],
                            "log_t0": row.get("log_t0"),
                            "log_Dt": row.get("log_Dt"),
                            "log_K": row.get("log_K"),
                            "log_r2": row.get("log_r2"),
                            "n_data_points": row.get("n_data_points"),
                        }
                    )
        summary_df.drop(columns=["_raw_innovation_name"], inplace=True)

        if not flag_rows:
            st.success(
                "No overlapping (spatial scale, indicator, description, metric) combos found "
                "across the merged casings — homologation is a pure rename here, no re-fit needed."
            )
        else:
            flag_df = pd.DataFrame(flag_rows)
            n_flagged = flag_df[group_cols[:1] + group_cols[1:]].drop_duplicates().shape[0]
            st.warning(
                f"{n_flagged} series now share the same (innovation, spatial scale, indicator, "
                "description, metric) key across different original casings, each with its own "
                "separately-fitted curve stored in the summary table. These are candidates for "
                "being refit as a single, unified series — flagged here for manual review, "
                "**not** refit automatically."
            )
            st.dataframe(flag_df, hide_index=True)
            st.download_button(
                "Download re-fit flag report (CSV)",
                flag_df.to_csv(index=False),
                file_name="innovation_casing_refit_flags.csv",
                mime="text/csv",
            )

# Create menu

col1, col2 = st.columns(2)

with col1:
    selected_innovation = _persist(
        st.selectbox,
        "Select innovation", innovation_names, index=0, key="dashboard_selected_innovation",
    )

with col2:
    # Keyed per-innovation, so switching innovation still resets this to its first option
    # instead of erroring/sticking on a spatial scale that doesn't exist for the new one.
    selected_spatial_scale = _persist(
        st.selectbox,
        "Select spatial scale",
        sorted(list(dosi_df.loc[dosi_df["Innovation Name"] == selected_innovation, "Spatial Scale"].unique())),
        index=0,
        key=f"dashboard_selected_spatial_scale_{selected_innovation}",
    )

NUMBER_OF_COLUMNS = 8  # Number of columns in the grid
st.subheader("Indicators included:")

# Keys are scoped to the current innovation/spatial scale selection, so switching either
# one still resets the checkboxes to their computed defaults instead of reusing stale state.
def _feature_key(label):
    return f"feature_{selected_innovation}_{selected_spatial_scale}_{label}"

col_all, col_none = st.columns(2)
with col_all:
    if st.button("All", key="features_select_all"):
        for label in indicator_codes:
            st.session_state[_feature_key(label)] = True
with col_none:
    if st.button("None", key="features_select_none"):
        for label in indicator_codes:
            st.session_state[_feature_key(label)] = False

cols = st.columns(NUMBER_OF_COLUMNS)
feature_states = {}
for idx, label in enumerate(indicator_codes):
    with cols[idx % NUMBER_OF_COLUMNS]:
        feature_states[label] = _persist(
            st.checkbox,
            label + " " + indicator_names[idx],
            value=label in list(dosi_df.loc[
                (dosi_df["Innovation Name"]==selected_innovation)
                & (dosi_df["Spatial Scale"]==selected_spatial_scale)
                , "Indicator Number"
            ].unique()),
            key=_feature_key(label),
        )

selected_only = _persist(
    st.toggle,
    "Show only logfits of selected or new timeseries", value=True, key="dashboard_selected_only",
)


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
    
    if len(innovation_df) > 0:

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
        marker_traces_meta = []

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
                            y=FPLogValue_with_scaling(years_for_plotting, t0, Dt, K),# / K,
                            mode="lines",
                            name=f"{code} - {metric}",  # Legend label
                            showlegend=False,
                            line=dict(color=color, width=2),
                            opacity=1-j*0.8/len(groups),
                            hovertemplate=f"""{code} <br>{code} ({metric[0]}: {metric[1]}) <br>Year=%{{x:.0f}}<br>Value=%{{y:.3g}}<br>Dt={Dt:.0f} t0={t0:.0f} K={K:.3g}<extra></extra>""",  # Custom tooltip
                        )
                    )
                    
                    #fig.update_layout(
                    #    yaxis_title="K-normalised value",
                    #    yaxis=dict(range=[0, 1.2])
                    #)
                    #K_dict[timeseries["name"].iloc[0]] = K
            
            groups = innovation_df[innovation_df["Indicator Number"] == code].groupby(group_vars)
            for j, (metric, timeseries) in enumerate(groups):

                # Defer the points trace: overlap-aware hover text needs to see every
                # code/metric's points together, so we collect them and add the traces
                # once the whole loop (across all codes) has finished, below.
                values = timeseries["Value"]# / K_dict[timeseries["name"].iloc[0]]
                eps_y = _series_epsilon(values)
                own_html = [
                    f"{code} ({metric[0]}: {metric[1]})<br>{code} Point<br>Year={year:.0f}<br>value={value:.3g}"
                    for year, value in zip(timeseries["Year"], values)
                ]
                other_html = [
                    f"{code} ({metric[0]}: {metric[1]}): Year={year:.0f}, value={value:.3g}"
                    for year, value in zip(timeseries["Year"], values)
                ]
                marker_traces_meta.append(dict(
                    x=timeseries["Year"].tolist(),
                    y=values.tolist(),
                    eps_y=[eps_y] * len(values),
                    own_html=own_html,
                    other_html=other_html,
                    name=f"{code} ({metric[0]}: {metric[1]})",  # This can be the same name to link with the line in the legend
                    color=color,
                    opacity=min(MARKER_MAX_OPACITY, 1-j*0.8/len(groups)),
                ))

                # centroid of the scatter points
                x_centroid = timeseries["Year"].mean()
                y_centroid = (timeseries["Value"]).mean() # / K_dict[timeseries["name"].iloc[0]]).mean()
                fig.add_annotation(
                    x=x_centroid,
                    y=y_centroid,
                    text=str(code),
                    showarrow=False,
                    xanchor="center",
                    yanchor="middle",
                    font=dict(color=color),  # label colour = line colour
                )

        if marker_traces_meta:
            all_x = [x for m in marker_traces_meta for x in m["x"]]
            all_y = [y for m in marker_traces_meta for y in m["y"]]
            all_eps_y = [e for m in marker_traces_meta for e in m["eps_y"]]
            all_own = [h for m in marker_traces_meta for h in m["own_html"]]
            all_other = [h for m in marker_traces_meta for h in m["other_html"]]
            all_hovertext = _build_overlap_aware_hovertext(all_x, all_y, all_own, all_other, all_eps_y)

            offset = 0
            for m in marker_traces_meta:
                n_points = len(m["x"])
                fig.add_trace(
                    go.Scatter(
                        x=m["x"],
                        y=m["y"],
                        mode="markers",
                        name=m["name"],
                        text=all_hovertext[offset:offset + n_points],
                        hovertemplate="%{text}<extra></extra>",
                        marker=dict(size=8, color=m["color"], line=dict(width=0.5, color="rgba(255,255,255,0.5)")),
                        opacity=m["opacity"],
                    )
                )
                offset += n_points

        fig.update_layout(
            title="Innovation " + inno_name + " in " + spatial_selection,
            xaxis_title="Year"
        )
        fig.update_layout(showlegend=False)
        fig.update_layout(height=900)  # make the plot taller
        return fig


fig = build_plot(dosi_df, summary_df, selected_innovation, feature_states, selected_spatial_scale)
if fig is None:
    st.info("No indicators selected - check at least one above to show a plot.")
else:
    st.plotly_chart(fig, width='stretch')
