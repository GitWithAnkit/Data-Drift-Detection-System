import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Allow importing from src directory
sys.path.append(os.path.dirname(__file__))

from drift_detector import detect_drift

# ==============================
# STREAMLIT CONFIG
# ==============================

st.set_page_config(
    page_title="Data Drift Detection System",
    layout="centered"
)

st.title("📊 Data Drift Detection System")
st.write(
    "Upload **reference (old)** and **current (new)** datasets to detect data drift "
    "using statistical techniques."
)

# ==============================
# SESSION STATE INITIALIZATION
# ==============================

if "drift_results" not in st.session_state:
    st.session_state.drift_results = None

if "reference_df" not in st.session_state:
    st.session_state.reference_df = None

if "current_df" not in st.session_state:
    st.session_state.current_df = None

# ==============================
# FILE UPLOAD SECTION
# ==============================

ref_file = st.file_uploader(
    "📁 Upload Reference Dataset (CSV)",
    type=["csv"]
)

cur_file = st.file_uploader(
    "📁 Upload Current Dataset (CSV)",
    type=["csv"]
)

if ref_file is not None and cur_file is not None:
    reference_df = pd.read_csv(ref_file)
    current_df = pd.read_csv(cur_file)

    st.session_state.reference_df = reference_df
    st.session_state.current_df = current_df

    st.subheader("🔎 Dataset Preview")
    st.write("Reference Dataset", reference_df.head())
    st.write("Current Dataset", current_df.head())

    # ==============================
    # DRIFT DETECTION BUTTON
    # ==============================

    if st.button("🚀 Detect Data Drift"):
        overall_drift, drift_report = detect_drift(reference_df, current_df)

        st.session_state.drift_results = {
            "overall_drift": overall_drift,
            "drift_report": drift_report
        }

# ==============================
# DISPLAY RESULTS (PERSISTENT)
# ==============================

if st.session_state.drift_results is not None:
    overall_drift = st.session_state.drift_results["overall_drift"]
    drift_report = st.session_state.drift_results["drift_report"]

    st.subheader("🔍 Drift Detection Result")

    # ==============================
    # DRIFT SUMMARY DASHBOARD
    # ==============================

    total_features = len(drift_report)
    drifted_features = len(drift_report[drift_report["drift_detected"] == "Yes"])
    high_drift = len(drift_report[drift_report["severity"] == "High"])

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Features", total_features)
    col2.metric("Drifted Features", drifted_features)
    col3.metric("High Drift Features", high_drift)

    if overall_drift == "Yes":
        st.error("⚠️ Data Drift Detected")
    else:
        st.success("✅ No Data Drift Detected")

    st.subheader("📋 Feature-wise Drift Report")
    st.dataframe(drift_report)

    # ==============================
    # DOWNLOAD REPORT BUTTON
    # ==============================

    csv = drift_report.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Drift Report",
        data=csv,
        file_name="drift_report.csv",
        mime="text/csv"
    )

    # ==============================
    # DRIFT VISUALIZATION
    # ==============================

    st.subheader("📈 Drift Visualization")

    numerical_features = drift_report["feature"].tolist()

    selected_feature = st.selectbox(
        "Select a numerical feature to visualize:",
        numerical_features
    )

    if selected_feature:
        ref_df = st.session_state.reference_df
        cur_df = st.session_state.current_df

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.hist(
            ref_df[selected_feature].dropna(),
            bins=30,
            alpha=0.6,
            label="Reference"
        )

        ax.hist(
            cur_df[selected_feature].dropna(),
            bins=30,
            alpha=0.6,
            label="Current"
        )

        ax.set_title(f"Distribution Comparison: {selected_feature}")
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Frequency")
        ax.legend()

        st.pyplot(fig)
