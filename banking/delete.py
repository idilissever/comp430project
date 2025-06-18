import pandas as pd

df = pd.read_csv("banking_simplified.csv")

columns_and_dgh = {
	"age": "age.csv",
	"balance": "balance.csv",
	"duration": "duration.csv",
	"previous": "previous.csv"
}

# Process each column
for col, dgh_file in columns_and_dgh.items():
	print(f"Processing {col}...")

	# Load original DGH
	dgh = pd.read_csv(f"hierarchies/{dgh_file}")

	# Get unique values from the dataset
	used_values = set(df[col].unique())

	# Filter DGH: first column should match used values
	dgh_filtered = dgh[dgh.iloc[:, 0].isin(used_values)]

	# Save filtered DGH
	output_file = dgh_file.replace(".csv", "_filtered.csv")
	dgh_filtered.to_csv(output_file, index=False)
	print(f"Saved filtered DGH to {output_file}")
