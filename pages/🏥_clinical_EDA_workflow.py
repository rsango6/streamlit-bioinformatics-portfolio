import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Clinical Data EDA", page_icon="📊", layout="wide")
st.title("📊 Clinical Dataset Exploration")
st.subheader("Anonymized data obtained for solving ML problems in clinical settings.")


@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("data/train.csv", index_col=0)
    
    # 1. Drop unwanted columns
    cols_to_drop = ['Education', 'Discharge_Status', 'Current_Work_Status', 'AdmissionType']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # 2. Clean Gender column
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace('Ž', 'F')
        df = df[df['Gender'].isin(['M', 'F'])]
        
    # 3. Drop all columns from the first 'b' column onward
    b_cols = [c for c in df.columns if str(c).lower().startswith('b')]
    if b_cols:
        cutoff = df.columns.get_loc(b_cols[0])
        df = df.iloc[:, :cutoff]
        
    return df

try:
    df = load_and_clean_data()
    
    # --- TOP METRICS ---
    st.markdown("### 🏥 High-Level Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", f"{df.shape[0]:,}")
    col2.metric("Total Features", df.shape[1])
    col3.metric("Avg Length of Stay (LOS)", f"{df['LOS'].mean():.1f} days" if 'LOS' in df.columns else "N/A")
    col4.metric("Avg ICU Days", f"{df['LOS_ICU'].mean():.1f} days" if 'LOS_ICU' in df.columns else "N/A")
    
    st.divider()

    # --- ROW 1: DEMOGRAPHICS ---
    st.markdown("### 👥 Patient Demographics")
    col_demo1, col_demo2 = st.columns(2)
    
    with col_demo1:
        if 'Age_Group' in df.columns and 'Gender' in df.columns:
            # Sort age groups so they display in logical order
            age_order = sorted(df['Age_Group'].dropna().unique())
            fig_age = px.histogram(df, x='Age_Group', color='Gender', barmode='group', 
                                   category_orders={"Age_Group": age_order},
                                   title="Age Group Distribution by Gender",
                                   color_discrete_map={
                                       'F': 'lightblue',
                                       'M': 'darkblue'
                                   })
            st.plotly_chart(fig_age, width='stretch')
            
    with col_demo2:
        if 'Gender' in df.columns:
            fig_gender = px.pie(df, names='Gender', 
                                hole=0.4, 
                                color='Gender',
                                title="Gender Breakdown",
                                color_discrete_map={
                                       'F': 'lightblue',
                                       'M': 'darkblue'
                                   })
            st.plotly_chart(fig_gender, width='stretch')

    st.divider()

    # --- ROW 2: CLINICAL METRICS ---
    st.markdown("### 🛏️ Hospitalization Metrics")
    col_clin1, col_clin2 = st.columns(2)
    
    with col_clin1:
        if 'LOS' in df.columns:
            fig_los = px.box(df, x='Gender' if 'Gender' in df.columns else None, y='LOS', 
                             color='Gender' if 'Gender' in df.columns else None,
                             title="Length of Stay (LOS) Distribution",
                             color_discrete_map={
                                       'F': 'lightblue',
                                       'M': 'darkblue'
                                   })
            st.plotly_chart(fig_los, use_container_width=True)
            
    with col_clin2:
        if 'Surgery_Count' in df.columns:
            fig_surg = px.histogram(df, x='Surgery_Count', 
                                    text_auto=True, 
                                    title="Number of Surgeries per Patient",
                                    )
            # Force x-axis to be integers
            fig_surg.update_xaxes(dtick=1) 
            st.plotly_chart(fig_surg, use_container_width=True)

    st.divider()

    # --- ROW 3: MEDICATION ANALYSIS ---
    st.markdown("### 💊 Medication Usage (Top 15)")
    # Find all columns ending with '_count'
    med_cols = [col for col in df.columns if col.endswith('_count')]
    
    if med_cols:
        # Sum them up and sort them
        med_sums = df[med_cols].sum().sort_values(ascending=False).head(15)
        
        # Clean up the names (remove '_count' for display)
        med_names = [name.replace('_count', '') for name in med_sums.index]
        
        fig_meds = px.bar(x=med_sums.values, y=med_names, orientation='h', 
                          labels={'x': 'Total Times Prescribed', 'y': 'Medication'},
                          title="Most Frequently Prescribed Medications (A-Category)")
        fig_meds.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_meds, use_container_width=True)
    else:
        st.info("No medication count columns found to display.")

    st.divider()

    # --- ROW 4: INTERACTIVE EXPLORER ---
    st.markdown("### 🔍 Deep Dive: Interactive Explorer")
    selected_col = st.selectbox("Select any clinical feature to explore its distribution:", df.columns)
    
    if df[selected_col].nunique() < 20 or df[selected_col].dtype == 'object':
        fig_custom = px.histogram(df, x=selected_col, color='Gender' if 'Gender' in df.columns else None, 
                                  barmode='group', title=f"Distribution of {selected_col}")
    else:
        fig_custom = px.histogram(df, x=selected_col, marginal="box", 
                                  title=f"Distribution and Outliers of {selected_col}")
        
    st.plotly_chart(fig_custom, use_container_width=True)

except FileNotFoundError:
    st.error("Please place your `train.csv` file in the `data/` folder.")