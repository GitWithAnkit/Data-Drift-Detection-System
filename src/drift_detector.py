import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

# ==============================
# CONFIGURATION
# ==============================

KS_PVALUE_THRESHOLD = 0.05   # KS test significance level
PSI_THRESHOLD = 0.2          # PSI threshold for drift

# ==============================
# PSI CALCULATION
# ==============================

def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index (PSI)
    """
    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.linspace(0, 100, bins + 1)
    expected_percents = np.percentile(expected, breakpoints)
    actual_percents = np.percentile(actual, breakpoints)

    psi_value = 0.0

    for i in range(len(expected_percents) - 1):
        expected_count = np.sum(
            (expected >= expected_percents[i]) & (expected < expected_percents[i + 1])
        ) / len(expected)

        actual_count = np.sum(
            (actual >= actual_percents[i]) & (actual < actual_percents[i + 1])
        ) / len(actual)

        # Avoid division by zero
        if expected_count == 0:
            expected_count = 0.0001
        if actual_count == 0:
            actual_count = 0.0001

        psi_value += (expected_count - actual_count) * np.log(expected_count / actual_count)

    return psi_value

# ==============================
# DRIFT DETECTION FUNCTION
# ==============================

def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    """
    Detect data drift between reference and current datasets.
    Returns:
        overall_drift (str): "Yes" or "No"
        results_df (DataFrame): feature-wise drift report
    """

    # Identify numerical features
    numerical_features = reference_df.select_dtypes(include=[np.number]).columns.tolist()

    drift_results = []

    for feature in numerical_features:
        ref_values = reference_df[feature].dropna()
        cur_values = current_df[feature].dropna()

        # Skip feature if empty
        if len(ref_values) == 0 or len(cur_values) == 0:
            continue

        # KS Test
        ks_stat, ks_pvalue = ks_2samp(ref_values, cur_values)

        # PSI
        psi_score = calculate_psi(ref_values, cur_values)

        # Drift decision
        drift_detected = (ks_pvalue < KS_PVALUE_THRESHOLD) or (psi_score > PSI_THRESHOLD)

        drift_results.append({
            "feature": feature,
            "ks_pvalue": round(ks_pvalue, 5),
            "psi": round(psi_score, 4),
            "drift_detected": "Yes" if drift_detected else "No"
        })

    results_df = pd.DataFrame(drift_results)

    # Overall drift decision
    if not results_df.empty and "Yes" in results_df["drift_detected"].values:
        overall_drift = "Yes"
    else:
        overall_drift = "No"

    return overall_drift, results_df

# ==============================
# SCRIPT MODE (OPTIONAL CLI RUN)
# ==============================

if __name__ == "__main__":
    # This block runs ONLY when file is executed directly
    # It will NOT run when imported into GUI

    REFERENCE_FILE = "../data/reference.csv"
    CURRENT_FILE = "../data/current.csv"

    print("Running drift detector in script mode...\n")

    reference_df = pd.read_csv(REFERENCE_FILE)
    current_df = pd.read_csv(CURRENT_FILE)

    overall_drift, report_df = detect_drift(reference_df, current_df)

    print("📊 Drift Detection Results:")
    print(report_df)

    print("\nOverall Drift Detected:", overall_drift)
