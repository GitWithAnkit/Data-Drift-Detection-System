import pandas as pd
import os

INPUT_FILE = "../data/spotify.csv"
OUTPUT_DIR = "../data"
REFERENCE_RATIO = 0.7
RANDOM_STATE = 42

# Load dataset (comma-separated)
df = pd.read_csv(INPUT_FILE)

# Drop target column if present
if "y" in df.columns:
    df = df.drop(columns=["y"])

print("Original dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Split
reference_df = df.sample(frac=REFERENCE_RATIO, random_state=RANDOM_STATE)
current_df = df.drop(reference_df.index)

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
reference_df.to_csv(os.path.join(OUTPUT_DIR, "old.csv"), index=False)
current_df.to_csv(os.path.join(OUTPUT_DIR, "new.csv"), index=False)

print("Reference shape:", reference_df.shape)
print("Current shape:", current_df.shape)
print("✅ Split completed correctly")
