import os

import pandas as pd
from anjana.anonymity import k_anonymity

# Read and process the data
data = pd.read_csv("banking-simplified.csv")
data.columns = data.columns.str.strip()

str_cols = [
	"education",
	"job",
	"marital",
	"month",
]
for col in str_cols:
	data[col] = data[col].str.strip()

# Define the identifiers, quasi-identifiers and the sensitive attribute
quasi_ident = [
	"age",
	"balance",
	"education",
	"job",
	"marital",
	"month",
	"previous"
]
ident = []
sens_att = "y"

# Select the suppression limit allowed
supp_level = 20

# Import the hierarchies for each quasi-identifier. Define a dictionary containing them
hierarchies = {
	"age": dict(pd.read_csv("banking_hierarchies/age.csv", header=None)),
	"balance": dict(pd.read_csv("banking_hierarchies/balance.csv", header=None)),
	"education": dict(pd.read_csv("banking_hierarchies/education.csv", header=None)),
	"job": dict(pd.read_csv("banking_hierarchies/job.csv", header=None)),
	"marital": dict(pd.read_csv("banking_hierarchies/marital.csv", header=None)),
	"month": dict(pd.read_csv("banking_hierarchies/month.csv", header=None)),
	"previous": dict(pd.read_csv("banking_hierarchies/previous.csv", header=None))
}

output_dir = "banking_simplified_anonymized"
os.makedirs(output_dir, exist_ok=True)

# Apply k-anonymity and save
for i in range(0, 11):  # 2^1 to 2^10 = 2 to 1024
	k = 2 ** i
	anon_data = k_anonymity(data, ident, quasi_ident, k, supp_level, hierarchies)
	anon_data.to_csv(os.path.join(output_dir, f"banking_k{k}.csv"), index=False)
