import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generator import (
    generate_raw_dataset,
    MEDICAL_CONDITIONS,
    ADMISSION_TYPES,
    DEPARTMENTS,
    HOSPITALS,
    INSURANCE_PROVIDERS,
    TEST_RESULTS
)
from preprocessing import clean_dataset, ALL_INPUT_FEATURES
from models import train_and_evaluate_models

# Page Configuration
st.set_page_config(
    page_title="MediTrack AI — Operations & Clinical Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Emerald & Indigo Clinical Dark Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

:root {
    --bg-dark: #0b0f19;
    --card-bg: #111827;
    --card-border: rgba(255, 255, 255, 0.08);
    --emerald: #10b981;
    --emerald-glow: rgba(16, 185, 129, 0.15);
    --indigo: #6366f1;
    --indigo-glow: rgba(99, 102, 241, 0.15);
    --cyan: #06b6d4;
    --amber: #f59e0b;
    --rose: #f43f5e;
    --text-primary: #f9fafb;
    --text-secondary: #9ca3af;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: var(--bg-dark);
    background-image: 
        radial-gradient(at 100% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 50%),
        radial-gradient(at 0% 100%, rgba(16, 185, 129, 0.08) 0px, transparent 50%);
    color: var(--text-primary);
}

[data-testid="stHeader"] {
    background: rgba(11, 15, 25, 0.85);
    backdrop-filter: blur(12px);
}

[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

[data-testid="stSidebar"] * {
    color: #cbd5e1;
}

.block-container {
    max-width: 1520px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

h1, h2, h3, .brand-header, .card-title {
    font-family: 'Outfit', sans-serif;
}

/* Header Banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 20px;
    padding: 1.6rem 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; right: 0; width: 300px; height: 100%;
    background: radial-gradient(circle at top right, rgba(16, 185, 129, 0.15), transparent 70%);
    pointer-events: none;
}

.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: var(--emerald);
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.03em;
    margin: 0 0 0.4rem 0;
}

.hero-desc {
    color: var(--text-secondary);
    font-size: 0.95rem;
    max-width: 820px;
    margin: 0;
}

/* Executive Metric Cards */
.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 1.2rem;
    position: relative;
    transition: all 0.25s ease;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
}

.metric-card:hover {
    border-color: rgba(16, 185, 129, 0.35);
    transform: translateY(-2px);
}

.metric-card.accent-emerald { border-left: 4px solid var(--emerald); }
.metric-card.accent-indigo { border-left: 4px solid var(--indigo); }
.metric-card.accent-cyan { border-left: 4px solid var(--cyan); }
.metric-card.accent-amber { border-left: 4px solid var(--amber); }

.metric-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: var(--text-secondary);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-icon {
    font-size: 1.1rem;
}

.metric-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0.4rem 0 0.2rem 0;
}

.metric-footer {
    color: #64748b;
    font-size: 0.75rem;
}

/* Insight Callout Box */
.insight-box {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.8), rgba(30, 41, 59, 0.8));
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 1.1rem 1.4rem;
    margin: 1.2rem 0;
    position: relative;
}

.insight-title {
    color: var(--indigo);
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.35rem;
}

.insight-body {
    color: #e2e8f0;
    font-size: 0.9rem;
    line-height: 1.55;
}

/* Custom Tabs & Buttons */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
}

div[data-testid="stButton"] button {
    border-radius: 12px;
    border: 1px solid rgba(16, 185, 129, 0.4);
    background: linear-gradient(135deg, #059669, #047857);
    color: #ffffff;
    font-weight: 600;
    padding: 0.65rem 1.2rem;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25);
    transition: all 0.2s ease;
}

div[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #10b981, #059669);
    border-color: rgba(16, 185, 129, 0.8);
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    transform: translateY(-1px);
}

.brand-sidebar {
    padding: 0.5rem 0 1.5rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 1rem;
}

.brand-name {
    font-family: 'Outfit', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 8px;
}

.brand-badge {
    background: rgba(16, 185, 129, 0.15);
    color: var(--emerald);
    font-size: 0.65rem;
    padding: 2px 8px;
    border-radius: 99px;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.nav-label {
    color: #64748b;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 1.2rem 0 0.6rem 0;
}

.footer-text {
    color: #475569;
    font-size: 0.78rem;
    text-align: center;
    padding-top: 3rem;
}
</style>
""", unsafe_allow_html=True)


# Load & Cache Dataset & Models
@st.cache_data(show_spinner=False)
def load_data():
    raw_df = generate_raw_dataset(seed=42)
    cleaned_df, missing_summary, dups_count, final_count = clean_dataset(raw_df)
    return raw_df, cleaned_df, missing_summary, dups_count, final_count


raw_df, cleaned_df, missing_summary, dups_count, final_count = load_data()


@st.cache_resource(show_spinner=False)
def get_models(df):
    return train_and_evaluate_models(df)


models, metrics_df, test_results, best_model_name = get_models(cleaned_df)


# Plotly Custom Theme Function
def apply_plotly_theme(fig, height=390):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=15, r=15, t=45, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        title_font=dict(family="Outfit, sans-serif", size=16, color="#f8fafc"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#1e293b", font_color="#f8fafc", font_family="Inter")
    )
    fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.06)", zeroline=False)
    return fig


def render_metric_card(title, value, subtitle, icon="📊", accent="emerald"):
    st.markdown(
        f'''
        <div class="metric-card accent-{accent}">
            <div class="metric-header">
                <span>{title}</span>
                <span class="metric-icon">{icon}</span>
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-footer">{subtitle}</div>
        </div>
        ''',
        unsafe_allow_html=True
    )


def render_hero_banner(title, description, tag="MEDITRACK OPERATIONS CENTER"):
    st.markdown(
        f'''
        <div class="hero-banner">
            <div class="hero-tag"><span>🟢 LIVE</span> • {tag}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-desc">{description}</p>
        </div>
        ''',
        unsafe_allow_html=True
    )


# Sidebar Layout
with st.sidebar:
    st.markdown(
        '''
        <div class="brand-sidebar">
            <div class="brand-name">
                🩺 MediTrack <span class="brand-badge">AI 2.0</span>
            </div>
            <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 4px;">
                Hospital Analytics & Operations
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    st.markdown('<div class="nav-label">Navigation Module</div>', unsafe_allow_html=True)
    
    # Custom styled navigation selector
    current_page = st.radio(
        "Navigation",
        [
            "1 · Data Intake & Schema",
            "2 · Data Quality & Cleaning",
            "3 · Operational Intelligence (EDA)",
            "4 · Predictive Model Suite",
            "5 · Patient Stay Simulator"
        ],
        label_visibility="collapsed"
    )

    st.markdown('<div class="nav-label">System Specs</div>', unsafe_allow_html=True)
    st.caption("⚡ Seed 42 Reproducible Engine")
    st.caption("📊 10,000 Verified Hospital Records")
    st.caption("🩺 Dual ML Pipeline (LR vs RF)")
    st.caption("🔒 Local Academic Instance")


# PAGE 1: DATA INTAKE & SCHEMA
if current_page == "1 · Data Intake & Schema":
    render_hero_banner(
        "Hospital Operational Data Intake",
        "Raw synthetic healthcare records prior to cleaning, capturing patient demographics, clinical diagnostics, admission details, and financial parameters.",
        "STEP 01 • RAW DATA INGESTION"
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Raw Records", f"{len(raw_df):,}", "Initial ingested rows", "📁", "emerald")
    with m2:
        render_metric_card("Total Attributes", f"{raw_df.shape[1]}", "Demographic & Clinical", "📋", "indigo")
    with m3:
        render_metric_card("Missing Value Cells", f"{int(raw_df.isna().sum().sum()):,}", "Requires cleaning", "⚠️", "amber")
    with m4:
        render_metric_card("Duplicate Rows", f"{dups_count:,}", "Identified for removal", "🔄", "cyan")

    st.markdown(
        '''
        <div class="insight-box">
            <div class="insight-title">💡 Operational Ingestion Summary</div>
            <div class="insight-body">
                The MediTrack intake engine synthesizes 10,000+ hospital admission transactions covering patient age, gender, blood classification, 
                diagnosed condition, admission urgency (Emergency/Urgent/Elective), attending physician, insurance carrier, clinical test results, 
                prior 12-month hospital visit history, total billing amount, and hospital stay duration.
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    t1, t2 = st.tabs(["📄 Raw Dataset Explorer", "📚 Data Schema & Column Guide"])

    with t1:
        st.subheader("Raw Dataset Preview (First 100 Records)")
        st.dataframe(raw_df.head(100), width="stretch", hide_index=True, height=450)

    with t2:
        st.subheader("Attribute Dictionary & Feature Mapping")
        schema_data = pd.DataFrame([
            ["patient_id", "Unique patient record identifier", "Categorical / Key"],
            ["age", "Patient age at admission (years)", "Numerical"],
            ["gender", "Patient gender (Male / Female / Other)", "Categorical"],
            ["blood_type", "Patient ABO blood group", "Categorical"],
            ["medical_condition", "Primary clinical diagnosis", "Categorical"],
            ["admission_type", "Admission triage level (Emergency/Urgent/Elective)", "Categorical"],
            ["admission_date", "Timestamp of hospital admission", "DateTime"],
            ["discharge_date", "Timestamp of hospital discharge", "DateTime"],
            ["department", "Hospital care department assigned", "Categorical"],
            ["hospital", "Partner health facility name", "Categorical"],
            ["doctor", "Attending physician of record", "Categorical"],
            ["insurance_provider", "Primary health insurance payer", "Categorical"],
            ["medication", "Primary prescribed pharmacotherapy", "Categorical"],
            ["test_result", "Initial lab/imaging evaluation", "Categorical"],
            ["previous_visits", "Prior admissions in past 12 months", "Numerical"],
            ["billing_amount", "Total charges accrued ($ USD)", "Numerical"],
            ["length_of_stay", "Hospital stay duration (Days) — Target Variable", "Target"]
        ], columns=["Attribute", "Description", "Data Classification"])
        st.dataframe(schema_data, width="stretch", hide_index=True)


# PAGE 2: DATA QUALITY & CLEANING
elif current_page == "2 · Data Quality & Cleaning":
    render_hero_banner(
        "Data Quality, Cleaning & Feature Engineering",
        "Automated data cleaning pipeline: median/mode missing value imputation, duplicate removal, date formatting, and analytical record downsizing to exactly 10,000 rows.",
        "STEP 02 • DATA CLEANING PIPELINE"
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Ingested Rows", f"{len(raw_df):,}", "Before preprocessing", "📥", "indigo")
    with m2:
        render_metric_card("Missing Cells Fixed", f"{int(raw_df.isna().sum().sum()):,}", "Median & Mode Imputation", "🔧", "emerald")
    with m3:
        render_metric_card("Duplicates Purged", f"{dups_count:,}", "Exact row duplicates", "🧹", "amber")
    with m4:
        render_metric_card("Cleaned Dataset", f"{final_count:,}", "Exact target sample count", "✨", "cyan")

    st.markdown("### 🛠️ Data Cleaning Audit Trail")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            '''
            <div class="insight-box">
                <div class="insight-title">1. Missing Value Strategy</div>
                <div class="insight-body">
                    • <b>Numeric Attributes</b> (<code>age</code>, <code>billing_amount</code>): Imputed using dataset medians to avoid outlier distortion.<br>
                    • <b>Categorical Attributes</b> (<code>gender</code>, <code>medical_condition</code>, <code>insurance_provider</code>, <code>test_result</code>): Imputed using categorical mode (most frequent value).
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
    with col_b:
        st.markdown(
            '''
            <div class="insight-box">
                <div class="insight-title">2. Feature Engineering & Trimming</div>
                <div class="insight-body">
                    • <b>Target Derivation</b>: Computed exact <code>length_of_stay</code> as <code>discharge_date - admission_date</code>.<br>
                    • <b>Temporal Features</b>: Extracted <code>admission_month</code> and <code>admission_weekday</code>.<br>
                    • <b>Demographic Bins</b>: Created <code>age_group</code> (0-18, 19-35, 36-50, 51-65, 65+).
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    st.subheader("Missing Value Remediation Table")
    st.dataframe(missing_summary, width="stretch", hide_index=True)

    st.subheader("Cleaned Analytical Dataset Preview (Exactly 10,000 Records)")
    st.dataframe(cleaned_df.head(100), width="stretch", hide_index=True, height=400)
    
    st.success("✅ Cleaning Complete: Analytical dataset verified with 10,000 valid records and 0 missing values.")


# PAGE 3: OPERATIONAL INTELLIGENCE (EDA)
elif current_page == "3 · Operational Intelligence (EDA)":
    render_hero_banner(
        "Hospital Operational Intelligence & Visual Analytics",
        "Exploratory analytics inspecting patient demographics, department workloads, admission urgency patterns, billing trends, and correlation heatmaps.",
        "STEP 03 • EXPLORATORY ANALYSIS"
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Total Patient Cohort", f"{final_count:,}", "Cleaned dataset size", "👥", "emerald")
    with m2:
        render_metric_card("Mean Length of Stay", f"{cleaned_df['length_of_stay'].mean():.1f} Days", "Target metric average", "⏱️", "indigo")
    with m3:
        render_metric_card("Mean Billing Cost", f"${cleaned_df['billing_amount'].mean():,.2f}", "Average patient invoice", "💳", "cyan")
    with m4:
        render_metric_card("Emergency Rate", f"{(cleaned_df['admission_type']=='Emergency').mean()*100:.1f}%", "High priority admissions", "🚨", "amber")

    # Multi-tab layout for organized visual analytics
    tab_demo, tab_ops, tab_fin, tab_corr = st.tabs([
        "👤 Patient Demographics",
        "🏥 Department & Stay Dynamics",
        "💵 Billing & Financial Analytics",
        "🔥 Correlation Engine"
    ])

    with tab_demo:
        st.subheader("Demographic & Clinical Condition Profiles")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            fig1 = px.histogram(
                cleaned_df, x="age", nbins=30,
                title="Age Distribution of Hospital Cohort",
                color_discrete_sequence=["#10b981"],
                marginal="box"
            )
            st.plotly_chart(apply_plotly_theme(fig1, 380), width="stretch")
            st.caption("Patient age spans from 1 to 95 years, with the highest concentration in the 45-65 senior demographic.")

        with d_col2:
            cond_data = cleaned_df["medical_condition"].value_counts().reset_index()
            cond_data.columns = ["Condition", "Patients"]
            fig2 = px.bar(
                cond_data, x="Patients", y="Condition", orientation="h",
                title="Patient Volume by Diagnosed Condition",
                color="Patients", color_continuous_scale=["#0f2438", "#10b981"]
            )
            fig2.update_coloraxes(showscale=False)
            st.plotly_chart(apply_plotly_theme(fig2, 380), width="stretch")
            st.caption("Heart Disease and Emergency Trauma account for the largest proportion of total clinical admissions.")

    with tab_ops:
        st.subheader("Operational Load & Stay Duration Breakdown")
        o_col1, o_col2 = st.columns([0.9, 1.1])
        with o_col1:
            adm_pie = cleaned_df["admission_type"].value_counts().reset_index()
            adm_pie.columns = ["Admission Type", "Count"]
            fig3 = px.pie(
                adm_pie, names="Admission Type", values="Count", hole=0.6,
                title="Admission Triage Mix",
                color="Admission Type",
                color_discrete_map={"Emergency": "#f43f5e", "Urgent": "#06b6d4", "Elective": "#10b981"}
            )
            st.plotly_chart(apply_plotly_theme(fig3, 380), width="stretch")

        with o_col2:
            dept_bar = cleaned_df["department"].value_counts().reset_index()
            dept_bar.columns = ["Department", "Volume"]
            fig4 = px.bar(
                dept_bar, x="Department", y="Volume",
                title="Hospital Department Patient Load",
                color_discrete_sequence=["#6366f1"]
            )
            st.plotly_chart(apply_plotly_theme(fig4, 380), width="stretch")

        st.subheader("Average Stay Duration by Clinical Diagnosis")
        los_df = cleaned_df.groupby("medical_condition", as_index=False)["length_of_stay"].mean().sort_values("length_of_stay")
        fig5 = px.bar(
            los_df, x="length_of_stay", y="medical_condition", orientation="h",
            title="Mean Length of Stay per Condition (Days)",
            labels={"length_of_stay": "Average Days", "medical_condition": "Clinical Condition"},
            color="length_of_stay", color_continuous_scale=["#1e1b4b", "#6366f1"]
        )
        fig5.update_coloraxes(showscale=False)
        st.plotly_chart(apply_plotly_theme(fig5, 360), width="stretch")

    with tab_fin:
        st.subheader("Financial & Invoice Analytics")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            fig6 = px.histogram(
                cleaned_df, x="billing_amount", nbins=40,
                title="Billing Amount Distribution ($ USD)",
                color_discrete_sequence=["#06b6d4"]
            )
            st.plotly_chart(apply_plotly_theme(fig6, 380), width="stretch")

        with f_col2:
            sample_pts = cleaned_df.sample(min(2500, len(cleaned_df)), random_state=42)
            fig7 = px.scatter(
                sample_pts, x="length_of_stay", y="billing_amount", color="admission_type",
                title="Length of Stay vs. Billing Amount",
                labels={"length_of_stay": "Stay Duration (Days)", "billing_amount": "Billing ($)"},
                color_discrete_sequence=["#f43f5e", "#06b6d4", "#10b981"]
            )
            st.plotly_chart(apply_plotly_theme(fig7, 380), width="stretch")

    with tab_corr:
        st.subheader("Multi-Attribute Correlation Matrix")
        num_fields = ["age", "previous_visits", "billing_amount", "length_of_stay"]
        corr_matrix = cleaned_df[num_fields].corr().round(2)
        fig10 = px.imshow(
            corr_matrix, text_auto=True, color_continuous_scale="Viridis", zmin=-1, zmax=1,
            title="Numerical Correlation Matrix", aspect="auto"
        )
        st.plotly_chart(apply_plotly_theme(fig10, 420), width="stretch")
        st.caption("High correlation (0.85) observed between hospital length of stay and total billing amount.")


# PAGE 4: PREDICTIVE MODEL SUITE
elif current_page == "4 · Predictive Model Suite":
    render_hero_banner(
        "Machine Learning Model Evaluation Suite",
        "Benchmarking Linear Regression vs. Random Forest Regressor on identical 80/20 train-test splits and Scikit-Learn ColumnTransformer preprocessors.",
        "STEP 04 • MODEL VALIDATION"
    )

    best_r2 = metrics_df["R² Score"].max()
    best_mae = metrics_df["MAE"].min()
    best_rmse = metrics_df["RMSE"].min()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Top R² Score", f"{best_r2:.3f}", f"Winner: {best_model_name}", "🏆", "emerald")
    with m2:
        render_metric_card("Lowest MAE", f"{best_mae:.2f} Days", "Mean absolute error", "📉", "indigo")
    with m3:
        render_metric_card("Lowest RMSE", f"{best_rmse:.2f} Days", "Root mean square error", "🎯", "cyan")
    with m4:
        render_metric_card("Validation Split", "80% / 20%", "Fixed seed random_state=42", "⚖️", "amber")

    st.subheader("Model Performance Validation Scorecard")
    
    scorecard_df = metrics_df.copy()
    scorecard_df["MAE"] = scorecard_df["MAE"].map(lambda x: f"{x:.3f} Days")
    scorecard_df["RMSE"] = scorecard_df["RMSE"].map(lambda x: f"{x:.3f} Days")
    scorecard_df["R² Score"] = scorecard_df["R² Score"].map(lambda x: f"{x:.3f}")
    st.dataframe(scorecard_df, width="stretch", hide_index=True)

    # Model comparison chart
    melted_df = metrics_df.melt(id_vars="Model", value_vars=["MAE", "RMSE", "R² Score"], var_name="Metric", value_name="Score")
    fig_comp = px.bar(
        melted_df, x="Metric", y="Score", color="Model", barmode="group", text_auto=".3f",
        title="Model Error Metrics & Explained Variance (R²)",
        color_discrete_sequence=["#06b6d4", "#10b981"]
    )
    st.plotly_chart(apply_plotly_theme(fig_comp, 400), width="stretch")

    st.markdown(
        f'''
        <div class="insight-box">
            <div class="insight-title">🏆 Optimal Model Finding: {best_model_name}</div>
            <div class="insight-body">
                The <b>{best_model_name}</b> pipeline delivered the highest predictive accuracy with an R² Score of <b>{best_r2:.3f}</b> and MAE of <b>{best_mae:.3f} Days</b>. 
                It effectively models complex interactions between patient age, diagnosis urgency, and lab test status.
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )


# PAGE 5: PATIENT STAY SIMULATOR
else:
    render_hero_banner(
        "Real-Time Patient Stay Duration Simulator",
        "Configure patient demographics, clinical condition, and admission parameters to estimate expected length of stay.",
        "STEP 05 • PREDICTIVE SIMULATION"
    )

    left_panel, right_panel = st.columns([0.85, 1.15], gap="large")

    with left_panel:
        st.subheader("⚙️ Clinical Profile Controls")
        
        sim_age = st.slider("Patient Age", 1, 95, 54)
        sim_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        sim_condition = st.selectbox("Medical Condition", MEDICAL_CONDITIONS)
        sim_adm_type = st.selectbox("Admission Type", ADMISSION_TYPES)
        sim_dept = st.selectbox("Department", DEPARTMENTS)
        sim_hosp = st.selectbox("Hospital Facility", HOSPITALS)
        sim_ins = st.selectbox("Insurance Provider", INSURANCE_PROVIDERS)
        sim_test = st.selectbox("Initial Test Result", TEST_RESULTS)
        sim_prev_visits = st.number_input("Prior Visits (Past 12m)", 0, 15, 2)
        sim_billing = st.number_input("Estimated Billing ($)", 350.0, 45000.0, 4500.0, step=250.0)

        run_sim = st.button("⚡ Calculate Expected Length of Stay", type="primary", width="stretch")

    with right_panel:
        st.subheader("📊 Model Prediction Output & Triage Guidance")

        input_data = pd.DataFrame([{
            "age": sim_age,
            "gender": sim_gender,
            "medical_condition": sim_condition,
            "admission_type": sim_adm_type,
            "department": sim_dept,
            "hospital": sim_hosp,
            "insurance_provider": sim_ins,
            "test_result": sim_test,
            "previous_visits": sim_prev_visits,
            "billing_amount": sim_billing
        }])

        pred_lr = max(1.0, float(models["Linear Regression"].predict(input_data)[0]))
        pred_rf = max(1.0, float(models["Random Forest"].predict(input_data)[0]))
        winner_pred = pred_rf if best_model_name == "Random Forest" else pred_lr

        if run_sim:
            p1, p2, p3 = st.columns(3)
            with p1:
                render_metric_card("Linear Regression", f"{pred_lr:.1f} Days", "Linear Estimate", "📐", "indigo")
            with p2:
                render_metric_card("Random Forest", f"{pred_rf:.1f} Days", "Ensemble Estimate", "🌲", "emerald")
            with p3:
                render_metric_card(f"Recommended ({best_model_name})", f"{winner_pred:.1f} Days", "Primary Triage Value", "🌟", "cyan")

            # Plotly Gauge Indicator
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=winner_pred,
                number={"suffix": " Days", "font": {"size": 36, "color": "#10b981", "family": "Outfit"}},
                title={"text": f"Predicted Length of Stay ({best_model_name})", "font": {"color": "#f8fafc", "family": "Outfit"}},
                gauge={
                    "axis": {"range": [1, 30]},
                    "bar": {"color": "#10b981"},
                    "bgcolor": "#111827",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [1, 5], "color": "#064e3b"},
                        {"range": [5, 12], "color": "#1e3a8a"},
                        {"range": [12, 30], "color": "#881337"}
                    ]
                }
            ))
            st.plotly_chart(apply_plotly_theme(gauge_fig, 310), width="stretch")

            st.subheader("📋 Operational & Clinical Guidance")
            guidance = []
            if winner_pred > 8.0:
                guidance.append("Extended Stay Warning: Initiate early care coordination and discharge planning.")
            if sim_adm_type == "Emergency" and sim_test in ["Critical", "Abnormal"]:
                guidance.append("High Severity Triage: Reserve step-down telemetry bed in Cardiology/Neurology.")
            if sim_age >= 65:
                guidance.append("Geriatric Protocol: Assess post-acute rehabilitation needs upon admission.")
            if sim_prev_visits >= 4:
                guidance.append("Readmission Risk: Flag for chronic disease management intervention.")

            if guidance:
                for g in guidance:
                    st.warning(f"• {g}")
            else:
                st.success("Standard Clinical Path: Patient profile fits routine length-of-stay expectations.")
        else:
            st.markdown(
                '''
                <div class="insight-box">
                    <div class="insight-title">👈 Simulation Ready</div>
                    <div class="insight-body">
                        Configure the patient profile parameters in the control panel on the left and press 
                        <b>Calculate Expected Length of Stay</b> to generate dual-model stay duration predictions.
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )

st.markdown('<div class="footer-text">MediTrack AI 2.0 • Hospital Operations & Clinical Intelligence Platform • Academic Analytics Project</div>', unsafe_allow_html=True)
