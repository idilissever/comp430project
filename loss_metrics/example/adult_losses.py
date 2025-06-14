from pathlib import Path

import pandas as pd

from dgh.anjana.adult_dgh import get_csv_adult_dghs
from loss_metrics.loss_calculator import *


def main():
    RAW_DATASET_PATH = Path("../../anjana_anonymizer/adult-over-simplified.csv")
    raw_dataset = pd.read_csv(RAW_DATASET_PATH)
    raw_dataset = raw_dataset.to_dict(orient="records")
    raw_dataset = [
        {k.strip(): v for k, v in row.items()} for row in raw_dataset
    ]  # Clean column names

    adult_dghs = get_csv_adult_dghs()
    k_values = [2**i for i in range(0, 11)]  # k = 1, 2, 4, ..., 1024
    results = []

    for k in k_values:
        ANONYMIZED_DATASET_PATH = Path(
            f"../../anjana_anonymizer/adult_over_simplified_clean/adult_k{k}.csv"
        )
        anonymized_dataset = pd.read_csv(ANONYMIZED_DATASET_PATH)
        anonymized_dataset = anonymized_dataset.to_dict(orient="records")
        md_loss = calculate_md_loss(adult_dghs, raw_dataset, anonymized_dataset)
        lm_loss = calculate_lm_loss(adult_dghs, anonymized_dataset)
        print(f"k={k}, MD Loss: {md_loss}, LM Loss: {lm_loss}")
        results.append({"k": k, "md_loss": round(md_loss, 4), "lm_loss": round(lm_loss, 4)})

    result_df = pd.DataFrame(results)
    result_df.to_csv("loss_metrics_report.csv", index=False)
    print("📁 Saved: loss_metrics_report.csv")

if __name__ == "__main__":
    main()
