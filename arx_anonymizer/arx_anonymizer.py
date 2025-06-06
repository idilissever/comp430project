import os
import pandas as pd
from pyarxaas import ARXaaS, AttributeType
from pyarxaas.privacy_models import KAnonymity

# Initialize ARXaaS client
arxaas = ARXaaS("https://arx.deidentifier.org/api")

# Paths
input_csv = "adult-over-simplified.csv"
output_dir = "../adult_over_simplified_anonymized_arx"
os.makedirs(output_dir, exist_ok=True)

# Load and preprocess data
data = pd.read_csv(input_csv)
data.columns = data.columns.str.strip()

str_cols = ["workclass", "education", "marital_status", "occupation", "sex"]
for col in str_cols:
    data[col] = data[col].str.strip()

# Define attribute types
attribute_types = {
    "index": AttributeType.IDENTIFYING,
    "age": AttributeType.QUASIIDENTIFYING,
    "education": AttributeType.QUASIIDENTIFYING,
    "occupation": AttributeType.QUASIIDENTIFYING,
    "marital_status": AttributeType.QUASIIDENTIFYING,
    "sex": AttributeType.QUASIIDENTIFYING,
    "income": AttributeType.SENSITIVE
}

# Load generalization hierarchies from CSV files
def load_hierarchy(path):
    df = pd.read_csv(path, header=None, dtype=str)
    return df.values.tolist()

hierarchies = {
    "age": load_hierarchy("hierarchies/age.csv"),
    "education": load_hierarchy("hierarchies/education.csv"),
    "occupation": load_hierarchy("hierarchies/occupation.csv"),
    "marital_status": load_hierarchy("hierarchies/marital.csv"),
    "sex": load_hierarchy("hierarchies/sex.csv"),
}

# Apply k-anonymity for k = 2^0 to 2^10
for i in range(0, 11):
    k = 2 ** i
    print(f"Anonymizing with k={k}...")

    dataset = arxaas.dataset(data, attribute_types=attribute_types, hierarchies=hierarchies)
    privacy_model = KAnonymity(k)
    result = arxaas.anonymize(dataset, [privacy_model])

    anon_df = result.anonymized_dataset.to_dataframe()
    anon_df.to_csv(os.path.join(output_dir, f"adult_k{k}.csv"), index=False)

    print(f"Saved: adult_k{k}.csv")
