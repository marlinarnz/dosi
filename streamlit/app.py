import streamlit as st

st.set_page_config(page_title="Multi-Page Demo", layout="wide")
st.write("# 👋 Welcome")

st.markdown(
    """
    Use the **page selector** in the sidebar to switch between the two pages  
    (Streamlit automatically lists any files placed in the **pages/** folder).
    """
)
