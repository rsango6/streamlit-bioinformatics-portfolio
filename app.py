import  streamlit as st

st.set_page_config(
    page_title="Bioinformatics Portfolio",
    page_icon="🧬",
    layout="wide"
)

st.title("💻🧬 Bioinformatics Portfolio")
st.write("A demo with walkthrough over RNA-seq and Clinical ML ready data")

st.markdown(
"""
This application demonstrates two distinct analytical workflows:


* **🧬 RNA-Seq Analysis:** An interactive dashboard for filtering differential gene expression data and rendering Volcano plots.
* **📊 Clinical Data EDA:** Exploratory Data Analysis for a machine learning pipeline predicting clinical outcomes.

**👈 Select a tool from the sidebar to begin.**

""")