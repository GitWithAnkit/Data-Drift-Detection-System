import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "data", "bank.csv")

REFERENCE_RATIO = 0.7

df = pd.read_csv(INPUT_FILE)

if "y" in df.columns:
    df = df.drop(columns=["y"])

reference_df = df.sample(frac=REFERENCE_RATIO, random_state=42)
current_df = df.drop(reference_df.index).copy()

# Numerical Drift
if "balance" in current_df.columns:
    current_df["balance"] = current_df["balance"] * 1.5

if "age" in current_df.columns:
    current_df = current_df[current_df["age"] < 40]

# Categorical Drift
if "job" in current_df.columns:
    dominant_job = current_df["job"].value_counts().index[0]
    current_df["job"] = dominant_job

if "marital" in current_df.columns:
    current_df["marital"] = "single"

reference_path = os.path.join(BASE_DIR, "data", "reference.csv")
current_path = os.path.join(BASE_DIR, "data", "current_drifted.csv")

reference_df.to_csv(reference_path, index=False)
current_df.to_csv(current_path, index=False)

print("Controlled drift datasets created.")
