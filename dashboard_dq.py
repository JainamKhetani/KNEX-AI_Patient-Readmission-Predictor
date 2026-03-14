# dashboard_dq.py
# PURPOSE: Interactive Streamlit dashboard for data quality and EDA.
# RUN WITH: streamlit run dashboard_dq.py
# OPENS AT: http://localhost:8501

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Data Quality & EDA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATABASE_URL = "postgresql://admin:admin@localhost:5432/hospital_dw"

# ── Load data ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300)   # Cache for 5 minutes
def load_silver():
    """Load from PostgreSQL silver layer."""
    try:
        eng = create_engine(DATABASE_URL)
        with eng.connect() as conn:
            df = pd.read_sql("SELECT * FROM silver.admissions LIMIT 50000", conn)
        return df, "PostgreSQL"
    except Exception:
        # Fallback to CSV if DB not available
        df = pd.read_csv("data/processed/silver_clean.csv")
        return df, "CSV fallback"

@st.cache_data(ttl=300)
def load_audit():
    """Load quality log from audit schema."""
    try:
        eng = create_engine(DATABASE_URL)
        with eng.connect() as conn:
            return pd.read_sql("SELECT * FROM audit.quality_log ORDER BY log_id", conn)
    except Exception:
        return pd.DataFrame()

df, source = load_silver()
df_audit = load_audit()

# ── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.title("🏥 Dashboard controls")
st.sidebar.caption(f"Data source: {source}")
st.sidebar.caption(f"Rows loaded: {len(df):,}")

page = st.sidebar.radio("Select view", [
    "Overview & KPIs",
    "Data quality report",
    "Missing value analysis",
    "Distributions",
    "Correlation analysis",
    "Outlier analysis",
    "Target variable analysis",
    "Raw data explorer"
])

# ══════════════════════════════════════════════════════════════════════
# PAGE 1: Overview & KPIs
# ══════════════════════════════════════════════════════════════════════
if page == "Overview & KPIs":
    st.title("🏥 Patient Readmission — Data Overview")
    st.caption("Summary of the UCI Diabetic 130-US Hospitals dataset")

    # Row 1: Big KPI numbers
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total encounters",      f"{len(df):,}")
    c2.metric("Unique patients",        f"{df['patient_nbr'].nunique():,}")
    c3.metric("30-day readmit rate",    f"{df['readmitted_30'].mean()*100:.1f}%",
              delta="Target variable")
    c4.metric("Avg hospital stay",      f"{df['time_in_hospital'].mean():.1f} days")
    c5.metric("Avg medications",        f"{df['num_medications'].mean():.0f}")

    st.divider()

    # Row 2: Dataset snapshot
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Dataset snapshot")
        snapshot = pd.DataFrame({
            "Metric": ["Total rows", "Total columns", "Numeric columns",
                       "Categorical columns", "Missing cells (total)",
                       "Missing cells (%)", "Duplicate encounters"],
            "Value": [
                f"{len(df):,}",
                f"{len(df.columns)}",
                f"{df.select_dtypes(include=np.number).shape[1]}",
                f"{df.select_dtypes(include='object').shape[1]}",
                f"{df.isnull().sum().sum():,}",
                f"{df.isnull().sum().sum()/(len(df)*len(df.columns))*100:.2f}%",
                f"{df['encounter_id'].duplicated().sum()}"
            ]
        })
        st.dataframe(snapshot, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("Readmission breakdown")
        readmit_counts = df["readmitted"].value_counts().reset_index()
        readmit_counts.columns = ["Category", "Count"]
        readmit_counts["Pct"] = (readmit_counts["Count"]/len(df)*100).round(1)
        fig = px.pie(readmit_counts, names="Category", values="Count",
                     hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Column list
    st.subheader("All columns in dataset")
    col_info = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str).values,
        "Non-null": df.notnull().sum().values,
        "Null": df.isnull().sum().values,
        "Null %": (df.isnull().sum()/len(df)*100).round(1).values,
        "Unique values": df.nunique().values,
        "Sample value": [str(df[c].dropna().iloc[0])
                         if df[c].notnull().any() else "NaN"
                         for c in df.columns]
    })
    st.dataframe(col_info, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 2: Data quality report
# ══════════════════════════════════════════════════════════════════════
elif page == "Data quality report":
    st.title("📋 Data Quality Report")

    if not df_audit.empty:
        # Quality score card
        total_checks = len(df_audit)
        passed = (df_audit["status"] == "PASS").sum()
        warned = (df_audit["status"] == "WARN").sum()
        failed = (df_audit["status"] == "FAIL").sum()
        overall_score = passed / total_checks * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall quality score", f"{overall_score:.0f}%",
                  delta="Good" if overall_score > 90 else "Needs attention")
        c2.metric("Checks passed", f"{passed}/{total_checks}", delta="PASS")
        c3.metric("Warnings", str(warned), delta="WARN")
        c4.metric("Failures", str(failed), delta="FAIL")

        st.divider()

        # Status bar chart
        status_counts = df_audit["status"].value_counts().reset_index()
        fig = px.bar(status_counts, x="status", y="count",
                     color="status",
                     color_discrete_map={"PASS":"#22c55e","WARN":"#f59e0b","FAIL":"#ef4444"},
                     title="Quality check results")
        st.plotly_chart(fig, use_container_width=True)

        # Full quality log table
        st.subheader("Detailed quality log")
        styled = df_audit[["step_name","check_name","records_total",
                            "records_failed","pass_rate_pct","status","details"]]
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("Run etl_pipeline.py first to generate quality log data.")
        # Compute quality on the fly
        st.subheader("Live quality checks")
        checks = []
        checks.append({"Check": "No duplicate encounters",
                        "Result": "PASS" if df["encounter_id"].duplicated().sum()==0 else "FAIL",
                        "Detail": f"{df['encounter_id'].duplicated().sum()} duplicates"})
        checks.append({"Check": "Encounter ID not null",
                        "Result": "PASS" if df["encounter_id"].notnull().all() else "FAIL",
                        "Detail": f"{df['encounter_id'].isnull().sum()} nulls"})
        checks.append({"Check": "Age in valid range",
                        "Result": "PASS" if (df["age_numeric"].dropna().between(0,100)).all() else "FAIL",
                        "Detail": f"Min={df['age_numeric'].min()}, Max={df['age_numeric'].max()}"})
        checks.append({"Check": "LOS positive",
                        "Result": "PASS" if (df["time_in_hospital"] > 0).all() else "FAIL",
                        "Detail": f"Min={df['time_in_hospital'].min()}"})
        checks.append({"Check": "Target variable binary",
                        "Result": "PASS" if df["readmitted_30"].isin([0,1]).all() else "FAIL",
                        "Detail": f"Values: {df['readmitted_30'].unique().tolist()}"})
        st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 3: Missing value analysis
# ══════════════════════════════════════════════════════════════════════
elif page == "Missing value analysis":
    st.title("🔍 Missing Value Analysis")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing count": df.isnull().sum().values,
        "Missing %": (df.isnull().sum()/len(df)*100).round(2).values
    }).sort_values("Missing %", ascending=False)

    missing_df = missing_df[missing_df["Missing count"] > 0]

    if missing_df.empty:
        st.success("No missing values found! The dataset is complete.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Columns with missing data", len(missing_df))
        c2.metric("Total missing cells",
                  f"{missing_df['Missing count'].sum():,}")

        fig = px.bar(missing_df, x="Missing %", y="Column",
                     orientation="h",
                     color="Missing %",
                     color_continuous_scale="Reds",
                     title="Missing value percentage by column")
        fig.add_vline(x=50, line_dash="dash", line_color="red",
                      annotation_text="50% threshold")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(missing_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 4: Distributions
# ══════════════════════════════════════════════════════════════════════
elif page == "Distributions":
    st.title("📊 Feature Distributions")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = ["race", "gender", "age", "insulin",
                "diabetes_med", "a1c_result"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    st.subheader("Numeric feature distributions")
    selected_num = st.selectbox("Select numeric column", num_cols)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = px.histogram(df, x=selected_num,
                                color="readmitted_30" if "readmitted_30" in df.columns else None,
                                nbins=30, barmode="overlay",
                                title=f"Distribution of {selected_num}",
                                labels={"readmitted_30": "Readmitted <30d"},
                                color_discrete_sequence=["#3b82f6","#ef4444"])
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        fig_box = px.box(df, y=selected_num,
                         x="readmitted_30" if "readmitted_30" in df.columns else None,
                         title=f"Box plot: {selected_num} by readmission",
                         color="readmitted_30" if "readmitted_30" in df.columns else None,
                         color_discrete_sequence=["#3b82f6","#ef4444"])
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Categorical feature distributions")
    selected_cat = st.selectbox("Select categorical column", cat_cols)
    cat_counts = df[selected_cat].value_counts().reset_index()
    cat_counts.columns = ["Value", "Count"]
    fig_cat = px.bar(cat_counts, x="Value", y="Count",
                     color="Count", color_continuous_scale="Blues",
                     title=f"Distribution of {selected_cat}")
    st.plotly_chart(fig_cat, use_container_width=True)

    # All numeric at once
    st.subheader("All numeric columns — descriptive statistics")
    st.dataframe(df[num_cols].describe().round(2), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 5: Correlation analysis
# ══════════════════════════════════════════════════════════════════════
elif page == "Correlation analysis":
    st.title("🔗 Correlation Analysis")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr_matrix = df[num_cols].corr().round(3)

    st.subheader("Pearson correlation heatmap")
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation matrix — all numeric features",
        aspect="auto"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation with target
    if "readmitted_30" in df.columns:
        st.subheader("Correlation with target variable (readmitted_30)")
        target_corr = corr_matrix["readmitted_30"].drop("readmitted_30")\
                        .sort_values(key=abs, ascending=False)
        fig2 = px.bar(
            x=target_corr.values,
            y=target_corr.index,
            orientation="h",
            color=target_corr.values,
            color_continuous_scale="RdBu_r",
            title="Feature correlation with 30-day readmission"
        )
        fig2.add_vline(x=0, line_color="black", line_width=1)
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 6: Outlier analysis
# ══════════════════════════════════════════════════════════════════════
elif page == "Outlier analysis":
    st.title("🎯 Outlier Analysis")

    outlier_cols = ["time_in_hospital","num_lab_procedures",
                    "num_procedures","num_medications","number_diagnoses"]
    outlier_cols = [c for c in outlier_cols if c in df.columns]

    st.subheader("IQR-based outlier detection")
    outlier_report = []
    for col in outlier_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        n_low  = (df[col] < lower).sum()
        n_high = (df[col] > upper).sum()
        outlier_report.append({
            "Column": col, "Q1": Q1, "Q3": Q3, "IQR": IQR,
            "Lower fence": round(lower,2), "Upper fence": round(upper,2),
            "Outliers low": n_low, "Outliers high": n_high,
            "Total outliers": n_low + n_high,
            "Outlier %": round((n_low+n_high)/len(df)*100, 2)
        })
    st.dataframe(pd.DataFrame(outlier_report),
                 use_container_width=True, hide_index=True)

    # Box plots for all outlier columns
    st.subheader("Box plots — all numeric features")
    fig = go.Figure()
    for col in outlier_cols:
        fig.add_trace(go.Box(y=df[col], name=col, boxpoints="outliers"))
    fig.update_layout(title="Outlier detection — box plots",
                      height=450, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 7: Target variable analysis
# ══════════════════════════════════════════════════════════════════════
elif page == "Target variable analysis":
    st.title("🎯 Target Variable Analysis (30-day Readmission)")

    if "readmitted_30" not in df.columns:
        st.error("readmitted_30 column not found. Run etl_pipeline.py first.")
    else:
        rate = df["readmitted_30"].mean() * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall 30-day readmission rate", f"{rate:.2f}%")
        c2.metric("Readmitted patients",  f"{df['readmitted_30'].sum():,}")
        c3.metric("Not readmitted",
                  f"{(df['readmitted_30']==0).sum():,}")

        # Readmission by age
        fig1 = px.bar(
            df.groupby("age")["readmitted_30"].mean().reset_index() \
              .rename(columns={"readmitted_30":"Rate"}),
            x="age", y="Rate",
            title="Readmission rate by age group",
            color="Rate", color_continuous_scale="Reds"
        )
        fig1.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig1, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            # By gender
            if "gender" in df.columns:
                fig2 = px.bar(
                    df.groupby("gender")["readmitted_30"].mean().reset_index(),
                    x="gender", y="readmitted_30",
                    title="Readmission rate by gender",
                    color="readmitted_30", color_continuous_scale="Blues"
                )
                st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            # By insulin
            if "insulin" in df.columns:
                fig3 = px.bar(
                    df.groupby("insulin")["readmitted_30"].mean().reset_index(),
                    x="insulin", y="readmitted_30",
                    title="Readmission rate by insulin type",
                    color="readmitted_30", color_continuous_scale="Oranges"
                )
                st.plotly_chart(fig3, use_container_width=True)

        # LOS vs readmission scatter
        fig4 = px.scatter(
            df.sample(min(5000, len(df))), x="time_in_hospital",
            y="num_medications", color="readmitted_30",
            title="Hospital stay vs medications — coloured by readmission",
            opacity=0.5,
            color_continuous_scale="RdYlGn_r",
            labels={"readmitted_30":"Readmitted <30d"}
        )
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 8: Raw data explorer
# ══════════════════════════════════════════════════════════════════════
elif page == "Raw data explorer":
    st.title("🔎 Raw Data Explorer")

    st.subheader("Filter data")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        readmit_filter = st.selectbox("Readmitted <30d",
                                      ["All","Yes (1)","No (0)"])
    with col_b:
        gender_vals = ["All"] + df["gender"].dropna().unique().tolist() \
            if "gender" in df.columns else ["All"]
        gender_filter = st.selectbox("Gender", gender_vals)
    with col_c:
        min_stay, max_stay = int(df["time_in_hospital"].min()), \
                             int(df["time_in_hospital"].max())
        stay_range = st.slider("Hospital stay (days)",
                               min_stay, max_stay, (min_stay, max_stay))

    # Apply filters
    mask = pd.Series([True] * len(df))
    if readmit_filter == "Yes (1)":
        mask &= df["readmitted_30"] == 1
    elif readmit_filter == "No (0)":
        mask &= df["readmitted_30"] == 0
    if gender_filter != "All" and "gender" in df.columns:
        mask &= df["gender"] == gender_filter
    mask &= df["time_in_hospital"].between(*stay_range)

    filtered = df[mask]
    st.caption(f"Showing {len(filtered):,} of {len(df):,} records")
    st.dataframe(filtered.head(500), use_container_width=True)

    # Download button
    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data as CSV",
                       csv_data, "filtered_patients.csv", "text/csv")