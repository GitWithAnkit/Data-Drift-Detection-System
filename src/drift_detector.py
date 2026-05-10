import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

KS_PVALUE_THRESHOLD = 0.05
PSI_THRESHOLD = 0.2


# ==============================
# PSI NUMERICAL
# ==============================
def calculate_psi_numeric(expected, actual, bins=10):
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

        if expected_count == 0:
            expected_count = 0.0001
        if actual_count == 0:
            actual_count = 0.0001

        psi_value += (expected_count - actual_count) * np.log(expected_count / actual_count)

    return psi_value


# ==============================
# PSI CATEGORICAL
# ==============================
def calculate_psi_categorical(expected, actual):
    expected_counts = expected.value_counts(normalize=True)
    actual_counts = actual.value_counts(normalize=True)

    all_categories = set(expected_counts.index).union(set(actual_counts.index))

    psi_value = 0.0

    for cat in all_categories:
        exp = expected_counts.get(cat, 0.0001)
        act = actual_counts.get(cat, 0.0001)

        psi_value += (exp - act) * np.log(exp / act)

    return psi_value


# ==============================
# MAIN DRIFT FUNCTION
# ==============================
def get_drift_severity(psi):
    if psi < 0.1:
        return "No Drift"
    elif psi < 0.2:
        return "Low"
    elif psi < 0.5:
        return "Medium"
    else:
        return "High"


def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame):

    numerical_features = reference_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = reference_df.select_dtypes(exclude=[np.number]).columns.tolist()

    drift_results = []

    # ===== Numerical Drift =====
    for feature in numerical_features:
        ref = reference_df[feature].dropna()
        cur = current_df[feature].dropna()

        if len(ref) == 0 or len(cur) == 0:
            continue

        ks_stat, ks_pvalue = ks_2samp(ref, cur)
        psi_score = calculate_psi_numeric(ref, cur)

        severity = get_drift_severity(psi_score)

        drift_detected = severity != "No Drift"

        drift_results.append({
            "feature": feature,
            "type": "Numerical",
            "ks_pvalue": round(ks_pvalue, 5),
            "psi": round(psi_score, 4),
            "severity": severity,
            "drift_detected": "Yes" if drift_detected else "No"
        })

    # ===== Categorical Drift =====
    for feature in categorical_features:
        ref = reference_df[feature].dropna()
        cur = current_df[feature].dropna()

        if len(ref) == 0 or len(cur) == 0:
            continue

        psi_score = calculate_psi_categorical(ref, cur)

        severity = get_drift_severity(psi_score)

        drift_detected = severity != "No Drift"

        drift_results.append({
            "feature": feature,
            "type": "Categorical",
            "ks_pvalue": "NA",
            "psi": round(psi_score, 4),
            "severity": severity,
            "drift_detected": "Yes" if drift_detected else "No"
        })

    results_df = pd.DataFrame(drift_results)

    overall_drift = "Yes" if "Yes" in results_df["drift_detected"].values else "No"

    return overall_drift, results_df
