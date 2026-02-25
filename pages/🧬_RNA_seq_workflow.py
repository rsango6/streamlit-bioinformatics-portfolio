import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="RNA-seq Analysis",
                   page_icon="🧬",
                   layout="wide")
st.title("Interactive RNA-Seq Volcano Plot")

#1. Load Data (using st.cache_data so it doesn't reload on every click)

@st.cache_data
def load_data():
    df = pd.read_csv("data/shrink_results_LPS.csv")

    df = df.rename(columns={'X': 'ENSEMBL'})
    df['ENSEMBL'] = df['ENSEMBL'].str.split('.').str[0]
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    #Calculate -log10(padj) for Y-axis.
    #We add a tiny number (1e-300) to prevent log(0) errors if padj is exactly zero.
    df['nlog10_padj'] = -np.log10(df['padj'] + 1e-300)
    return df

try:
    df = load_data()

    st.sidebar.header("Filter Thresholds")
    lfc_threshold = st.sidebar.slider("Log2FoldChange (Absolute)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    padj_threshold = st.sidebar.number_input("Adjusted P-Value cutoff", value=0.05, format="%.3f")

    def categorize_gene(row):
        if row['padj'] < padj_threshold and row['log2FoldChange'] > lfc_threshold:
            return 'Upregulated'
        elif row['padj'] < padj_threshold and row['log2FoldChange'] < -lfc_threshold:
            return 'Downregulated'
        else:
            return 'Not Significant'
    
    df['Significance'] = df.apply(categorize_gene, axis=1)

    # 4. Filter the table display based on sidebar

    st.subheader("Significant Genes Data")
    filtered_df = df[df['Significance'] != 'Not Significant']
    st.dataframe(filtered_df[['symbol','baseMean', 'log2FoldChange', 'padj', 'Significance']])
    st.write(f"Found **{len(filtered_df)}** significant genes.")

    #5. Plotly Volcano Plot
    st.subheader("Volcano Plot")

    color_map = {'Upregulated': 'red', 'Downregulated': 'blue', 'Not Significant': 'lightgrey'}

    fig = px.scatter(
        df,
        x='log2FoldChange',
        y='nlog10_padj',
        color='Significance',
        color_discrete_map=color_map,
        hover_name='symbol',
        hover_data={'nlog10_padj': False, 
                    'padj': ':2.e',
                    'log2FoldChange': ':.2f'
                    },
        title=f"Volcano Plot (LFC > {lfc_threshold}, padj < {padj_threshold})",
        range_y=[0, df['nlog10_padj'].max() + 20]
    )

    # Add threshold lines
    fig.add_hline(y=-np.log10(padj_threshold), line_dash="dash", line_color="black")
    fig.add_vline(x=lfc_threshold, line_dash="dash", line_color="black")
    fig.add_vline(x=-lfc_threshold, line_dash="dash", line_color="black")

    st.plotly_chart(fig, width='stretch')

except FileExistsError:
    st.error("Please place the RNA-seq csv file with correct header in the `data/` folder.")
    
