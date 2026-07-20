import streamlit as st

st.set_page_config(page_title="PosTip Data Diagnotistics", layout="wide")
st.write("# 👋 Welcome")

st.markdown(
    """
    Use the **page selector** in the sidebar to switch between pages:
    
    * Dashboard: Visualise time series data points by selecting a specific innovation and spatial scale
    * Single Series Fit: Deep-dive into one time series, exclude points/ranges, and compare logistic/exponential/linear curve fits
    * Clusters: Visualise time series data and their logistic fits by cluster (optionally filter for early-adopting regions)
    * Coevolution: Analyse the similarity of each two innovations within the selected cluster, spatial scale, innovation indicator
    * Hubs and bridges: Show the relation of temporal delay and logfit slope (or other metrics) for hub or bridge innovations of a specific spatial scale, cluster, and/or innovation indicator
    
    The pages let you select the data and logfit version to display. They display the latest as default.
    """
)
