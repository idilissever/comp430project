import pandas as pd
import difflib
from pathlib import Path


def reduce_to_matching_columns(
    detailed_csv_path: str, reference_csv_path: str, output_csv_path: str
) -> None:
    """
    Removes columns from detailed CSV that don't closely match those in the reference CSV.
    Saves the reduced DataFrame to output_csv_path.
    """
    detailed_df = pd.read_csv(detailed_csv_path)
    reference_df = pd.read_csv(reference_csv_path)

    reference_cols = set(reference_df.columns)
    matched_cols = [
        col
        for col in detailed_df.columns
        if (
            difflib.get_close_matches(col, reference_cols, n=1, cutoff=0.9)
            or col in ["income", "salary-class", "gender", "sex"]
        )
    ]

    reduced_df = detailed_df[matched_cols]
    reduced_df.to_csv(output_csv_path, index=False)


# Example usage:
reduce_to_matching_columns(
    "../raw_data/adult.csv",
    "../anjana_anonymizer/adult-simplified.csv",
    "../anjana_anonymizer/adult-simplified.csv",
)
