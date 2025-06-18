import os
import pandas as pd
from anjana.anonymity import k_anonymity

# Read and process the data
data = pd.read_csv("adult-over-simplified.csv")
data.columns = data.columns.str.strip()

str_cols = [
	"workclass",
	"education",
	"marital_status",
	"occupation",
	"sex",
]
for col in str_cols:
	data[col] = data[col].str.strip()

# Define the identifiers, quasi-identifiers and the sensitive attribute
quasi_ident = [
	"age",
	"workclass",
	"education",
	"occupation",
	"marital_status",
	"sex",
]
ident = []
sens_att = "income"

# Select the suppression limit allowed
supp_level = 20

# Import the hierarchies for each quasi-identifier. Define a dictionary containing them
hierarchies = {
	"age": dict(pd.read_csv("hierarchies/age.csv", header=None)),
	"workclass": dict(pd.read_csv("hierarchies/workclass.csv", header=None)),
	"education": dict(pd.read_csv("hierarchies/education.csv", header=None)),
	"occupation": dict(pd.read_csv("hierarchies/occupation.csv", header=None)),
	"marital_status": dict(pd.read_csv("hierarchies/marital.csv", header=None)),
	"sex": dict(pd.read_csv("hierarchies/sex.csv", header=None)),
}

output_dir = "adult_over_simplified_anonymized"
os.makedirs(output_dir, exist_ok=True)

# Apply k-anonymity and save
for i in range(0, 11):  # 2^1 to 2^10 = 2 to 1024
	k = 2 ** i
	anon_data = k_anonymity(data, ident, quasi_ident, k, supp_level, hierarchies)
	anon_data.to_csv(os.path.join(output_dir, f"adult_k{k}.csv"), index=False)
