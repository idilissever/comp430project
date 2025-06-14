import os
import pandas as pd

# Config
RAW_PATH = "adult-over-simplified.csv"
ANON_FOLDER = "adult_over_simplified_anonymized"
OUTPUT_FOLDER = "adult_over_simplified_clean"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load raw dataset
raw_df = pd.read_csv(RAW_PATH)
raw_df.columns = raw_df.columns.str.strip()
raw_df["index"] = raw_df.index  # Add index column explicitly
raw_indices = set(raw_df["index"])
all_columns = list(raw_df.columns)

# Process each anonymized file
for filename in os.listdir(ANON_FOLDER):
    if not filename.endswith(".csv"):
        continue

    anon_path = os.path.join(ANON_FOLDER, filename)
    anon_df = pd.read_csv(anon_path)
    anon_df.columns = anon_df.columns.str.strip()

    # Remove unnecessary level_0 column if present
    if "level_0" in anon_df.columns:
        anon_df.drop(columns=["level_0"], inplace=True)

    # Ensure "index" column exists
    if "index" not in anon_df.columns:
        raise ValueError(f'"index" column missing in {filename}')

    anon_indices = set(anon_df["index"])
    missing_indices = sorted(raw_indices - anon_indices)

    suppressed_rows = []
    for idx in missing_indices:
        label = raw_df.loc[idx, "income"]
        suppressed = {col: "*" for col in all_columns if col not in ["index", "income"]}
        suppressed["index"] = idx
        suppressed["income"] = label
        suppressed_rows.append(suppressed)

    suppressed_df = pd.DataFrame(suppressed_rows, columns=all_columns)
    full_df = pd.concat([anon_df, suppressed_df], ignore_index=True)
    full_df = full_df.sort_values(by="index").reset_index(drop=True)

    out_path = os.path.join(OUTPUT_FOLDER, filename)
    full_df.to_csv(out_path, index=False)

    print(f"✅ Cleaned: {filename}")

print("🎉 All cleaned datasets saved to:", OUTPUT_FOLDER)
