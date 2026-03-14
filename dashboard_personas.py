# dashboard_personas.py
# RUN WITH: streamlit run dashboard_personas.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Persona Dashboards",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load():
    df = pd.read_csv("data/processed/silver_clean.csv")
    np.random.seed(42)
    # Simulate risk scores based on known risk factors
    df["risk_score"] = (
        0.1
        + (df["number_inpatient"] * 0.08)
        + (df["time_in_hospital"] * 0.01)
        + (df["num_medications"] * 0.005)
        + (df["number_diagnoses"] * 0.01)
        + (df["readmitted_30"] * 0.3)
        + np.random.normal(0, 0.05, len(df))
    ).clip(0, 1)
    df["high_risk"] = df["risk_score"] > 0.5
    return df

df = load()

# ── Sidebar navigation ─────────────────────────────────────────────────
st.sidebar.title("Select Persona")
persona = st.sidebar.radio("", [
    "👨‍⚕️  Attending Physician",
    "🗂️  Case Manager",
    "🏥  Hospital Administrator",
    "💼  CFO / Finance"
])

# ══════════════════════════════════════════════════════════════════════
# PERSONA 1: ATTENDING PHYSICIAN
# ══════════════════════════════════════════════════════════════════════
if "Physician" in persona:
    st.title("👨‍⚕️ Physician Console — Patient Risk Monitor")
    st.caption("Identify high-risk patients before discharge for early intervention")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High-risk patients", f"{df['high_risk'].sum():,}",
              delta=f"{df['high_risk'].mean()*100:.1f}% of all")
    c2.metric("30-day readmit rate", f"{df['readmitted_30'].mean()*100:.1f}%")
    c3.metric("Avg risk score (all)", f"{df['risk_score'].mean():.3f}")
    c4.metric("Avg medications (high-risk)",
              f"{df[df['high_risk']]['num_medications'].mean():.1f}")

    st.divider()

    # High-risk patient table
    st.subheader("High-risk patients — sorted by risk score")
    cols_show = ["encounter_id", "patient_nbr", "age", "gender", "race",
                 "time_in_hospital", "num_medications", "number_diagnoses",
                 "diag_1", "insulin", "risk_score"]
    cols_show = [c for c in cols_show if c in df.columns]
    hi_risk_df = df[df["high_risk"]].sort_values(
        "risk_score", ascending=False).head(50)[cols_show]

    st.dataframe(
        hi_risk_df.style.background_gradient(subset=["risk_score"], cmap="Reds"),
        use_container_width=True, hide_index=True
    )

    # Risk score histogram
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(df, x="risk_score", color="readmitted_30",
                           nbins=40, barmode="overlay",
                           title="Risk score distribution",
                           color_discrete_map={0: "#3b82f6", 1: "#ef4444"},
                           labels={"readmitted_30": "Readmitted <30d"})
        fig.add_vline(x=0.5, line_dash="dash", line_color="orange",
                      annotation_text="High-risk threshold")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.box(df, x="insulin", y="risk_score", color="readmitted_30",
                      title="Risk score by insulin type",
                      color_discrete_map={0: "#3b82f6", 1: "#ef4444"})
        st.plotly_chart(fig2, use_container_width=True)

    # Top diagnoses in high-risk patients
    st.subheader("Most common primary diagnoses in high-risk patients")
    top_diag = df[df["high_risk"]]["diag_1"].value_counts().head(15).reset_index()
    top_diag.columns = ["ICD Code", "Count"]
    fig3 = px.bar(top_diag, x="Count", y="ICD Code", orientation="h",
                  color="Count", color_continuous_scale="Reds",
                  title="Top 15 primary diagnoses (high-risk patients)")
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PERSONA 2: CASE MANAGER
# ══════════════════════════════════════════════════════════════════════
elif "Case Manager" in persona:
    st.title("🗂️ Case Manager — Discharge Planning")
    st.caption("Monitor length of stay, medication complexity, and follow-up needs")

    avg_los = df["time_in_hospital"].mean()
    benchmark_los = 4.5

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg length of stay", f"{avg_los:.1f} days",
              delta=f"{avg_los - benchmark_los:+.1f} vs benchmark",
              delta_color="inverse")
    c2.metric("Extended stays (>7 days)",
              f"{(df['time_in_hospital'] > 7).sum():,}")
    c3.metric("High medication load (>20)",
              f"{(df['num_medications'] > 20).sum():,}")
    c4.metric("Multiple prior admissions",
              f"{(df['number_inpatient'] > 1).sum():,}")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        # LOS by age group
        los_age = df.groupby("age")["time_in_hospital"].mean().reset_index()
        los_age.columns = ["Age group", "Avg LOS"]
        los_age = los_age.sort_values("Age group")
        fig = px.bar(los_age, x="Age group", y="Avg LOS",
                     title="Average LOS by age group",
                     color="Avg LOS", color_continuous_scale="Blues")
        fig.add_hline(y=benchmark_los, line_dash="dash",
                      line_color="red", annotation_text="Benchmark")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Readmission by number of prior inpatient visits
        prior = df.groupby("number_inpatient")["readmitted_30"].mean().reset_index()
        prior = prior[prior["number_inpatient"] <= 10]
        fig2 = px.line(prior, x="number_inpatient", y="readmitted_30",
                       markers=True,
                       title="Readmit rate vs prior inpatient visits",
                       labels={"number_inpatient": "Prior inpatient visits",
                               "readmitted_30": "30-day readmit rate"})
        st.plotly_chart(fig2, use_container_width=True)

    # Discharge planning table — patients needing attention
    st.subheader("Patients flagged for discharge review")
    flagged = df[
        (df["time_in_hospital"] > 7) |
        (df["num_medications"] > 20) |
        (df["number_inpatient"] > 2)
    ].sort_values("time_in_hospital", ascending=False).head(30)

    show_cols = ["encounter_id", "age", "gender", "time_in_hospital",
                 "num_medications", "number_inpatient", "number_diagnoses",
                 "readmitted_30"]
    show_cols = [c for c in show_cols if c in flagged.columns]
    st.dataframe(flagged[show_cols], use_container_width=True, hide_index=True)

    # Medication vs LOS scatter
    st.subheader("Medication load vs length of stay")
    fig3 = px.scatter(df.sample(3000), x="num_medications",
                      y="time_in_hospital", color="readmitted_30",
                      color_continuous_scale="RdYlGn_r", opacity=0.5,
                      title="Medications vs LOS — coloured by readmission",
                      trendline="ols")
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PERSONA 3: HOSPITAL ADMINISTRATOR
# ══════════════════════════════════════════════════════════════════════
elif "Administrator" in persona:
    st.title("🏥 Hospital Administrator — Operations Overview")
    st.caption("Hospital-wide readmission trends, admission types, and department performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total admissions", f"{len(df):,}")
    c2.metric("Readmitted <30d", f"{df['readmitted_30'].sum():,}",
              delta=f"{df['readmitted_30'].mean()*100:.1f}% rate")
    c3.metric("Avg procedures/patient", f"{df['num_procedures'].mean():.1f}")
    c4.metric("Emergency admissions",
              f"{(df['admission_type_id'] == 1).sum():,}")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        # Readmit rate by admission type
        adm_type_map = {1: "Emergency", 2: "Urgent", 3: "Elective",
                        4: "Newborn", 5: "Not Available", 6: "NULL",
                        7: "Trauma", 8: "Not Mapped"}
        df["admission_label"] = df["admission_type_id"].map(adm_type_map).fillna("Other")
        adm_rate = df.groupby("admission_label")["readmitted_30"].agg(
            ["mean", "count"]).reset_index()
        adm_rate.columns = ["Admission type", "Readmit rate", "Count"]
        adm_rate = adm_rate.sort_values("Readmit rate", ascending=False)
        fig = px.bar(adm_rate, x="Admission type", y="Readmit rate",
                     color="Readmit rate", color_continuous_scale="Oranges",
                     title="Readmission rate by admission type",
                     text=adm_rate["Count"].apply(lambda x: f"n={x:,}"))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Discharge disposition vs readmission
        disp_rate = df.groupby("discharge_disposition_id")["readmitted_30"].agg(
            ["mean", "count"]).reset_index()
        disp_rate.columns = ["Disposition ID", "Readmit rate", "Count"]
        disp_rate = disp_rate[disp_rate["Count"] > 100].sort_values(
            "Readmit rate", ascending=False).head(10)
        fig2 = px.bar(disp_rate, x="Disposition ID", y="Readmit rate",
                      color="Readmit rate", color_continuous_scale="Reds",
                      title="Readmit rate by discharge disposition (top 10)")
        st.plotly_chart(fig2, use_container_width=True)

    # Race analysis
    col_c, col_d = st.columns(2)
    with col_c:
        race_rate = df.groupby("race")["readmitted_30"].mean().reset_index()
        race_rate.columns = ["Race", "Readmit rate"]
        fig3 = px.bar(race_rate.sort_values("Readmit rate", ascending=False),
                      x="Race", y="Readmit rate",
                      title="Readmission rate by race",
                      color="Readmit rate", color_continuous_scale="Purples")
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        # LOS distribution
        fig4 = px.histogram(df, x="time_in_hospital", nbins=14,
                            title="Length of stay distribution",
                            color_discrete_sequence=["#3b82f6"])
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PERSONA 4: CFO
# ══════════════════════════════════════════════════════════════════════
elif "CFO" in persona:
    st.title("💼 CFO Dashboard — Financial Risk & Penalty Exposure")
    st.caption("CMS readmission penalties and cost drivers")

    PENALTY_PER_READMIT = 15000   # USD per preventable readmission
    PREVENTABLE_RATE    = 0.30    # 30% of readmissions are preventable
    COST_PER_DAY        = 2500    # USD per hospital day

    total_readmits      = df["readmitted_30"].sum()
    preventable         = int(total_readmits * PREVENTABLE_RATE)
    penalty_exposure    = preventable * PENALTY_PER_READMIT
    excess_los_cost     = (df["time_in_hospital"] - df["time_in_hospital"].median())\
                           .clip(lower=0).sum() * COST_PER_DAY

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("30-day readmissions", f"{total_readmits:,}")
    c2.metric("Preventable (est. 30%)", f"{preventable:,}")
    c3.metric("Penalty exposure (USD)", f"${penalty_exposure:,.0f}")
    c4.metric("Excess LOS cost (USD)", f"${excess_los_cost:,.0f}")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        # Cost drivers scatter
        fig = px.scatter(df.sample(3000), x="num_medications",
                         y="time_in_hospital",
                         color="readmitted_30",
                         size="number_diagnoses",
                         color_continuous_scale="RdYlGn_r",
                         opacity=0.6,
                         title="Cost drivers: medications × LOS × diagnoses",
                         labels={"readmitted_30": "Readmitted <30d",
                                 "number_diagnoses": "# diagnoses"})
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Penalty by admission type
        adm_type_map = {1:"Emergency",2:"Urgent",3:"Elective",
                        4:"Newborn",5:"N/A",6:"NULL",7:"Trauma",8:"Other"}
        df["adm_label"] = df["admission_type_id"].map(adm_type_map).fillna("Other")
        pen_df = df.groupby("adm_label").agg(
            readmits=("readmitted_30","sum"),
            total=("readmitted_30","count")
        ).reset_index()
        pen_df["penalty"] = pen_df["readmits"] * PREVENTABLE_RATE * PENALTY_PER_READMIT
        fig2 = px.bar(pen_df.sort_values("penalty", ascending=False),
                      x="adm_label", y="penalty",
                      title="Penalty exposure by admission type (USD)",
                      color="penalty", color_continuous_scale="Reds",
                      labels={"adm_label": "Admission type",
                              "penalty": "Penalty (USD)"})
        st.plotly_chart(fig2, use_container_width=True)

    # Summary table
    st.subheader("Financial summary")
    summary = pd.DataFrame({
        "Metric": [
            "Total 30-day readmissions",
            "Estimated preventable readmissions (30%)",
            "Penalty per readmission (CMS estimate)",
            "Total penalty exposure",
            "Avg cost per admission day",
            "Excess LOS cost",
            "Total financial risk"
        ],
        "Value": [
            f"{total_readmits:,}",
            f"{preventable:,}",
            f"${PENALTY_PER_READMIT:,}",
            f"${penalty_exposure:,.0f}",
            f"${COST_PER_DAY:,}",
            f"${excess_los_cost:,.0f}",
            f"${penalty_exposure + excess_los_cost:,.0f}"
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)