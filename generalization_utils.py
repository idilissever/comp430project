import pandas as pd
from sklearn.preprocessing import LabelEncoder


def is_interval(value):
    return isinstance(value, str) and "," in value and all(part.strip().isdigit() for part in value.split(","))


def convert_interval_to_mean(value):
    """Convert '30,40' → 35.0"""
    if is_interval(value):
        a, b = value.split(",")
        return (float(a) + float(b)) / 2
    try:
        return float(value)
    except:
        return None


def split_interval(value):
    """Convert '30,40' → (30.0, 40.0)"""
    if is_interval(value):
        a, b = value.split(",")
        return float(a), float(b)
    try:
        v = float(value)
        return v, v
    except:
        return None, None


def process_numeric_intervals(df, columns, keep_width=False):
    """
    For each column in 'columns', splits 'col' into 'col_lower' and 'col_upper'.
    Optionally adds 'col_range_width'.
    """
    for col in columns:
        df[[f"{col}_lower", f"{col}_upper"]] = df[col].apply(lambda x: pd.Series(split_interval(x)))
        if keep_width:
            df[f"{col}_range_width"] = df[f"{col}_upper"] - df[f"{col}_lower"]
        df.drop(col, axis=1, inplace=True)
    return df


def encode_categoricals(df, columns):
    """
    Label-encode the specified categorical columns.
    '*' and missing values are treated as separate valid categories.
    """
    for col in columns:
        df[col] = df[col].astype(str).fillna("UNKNOWN")
        df[col] = LabelEncoder().fit_transform(df[col])
    return df


def encode_target(df, target_column):
    """
    Encode the target (class/label) column into 0/1 format.
    Example: '<=50K' → 0, '>50K' → 1
    """
    df[target_column] = LabelEncoder().fit_transform(df[target_column].astype(str))
    return df


def preprocess_anonymized_dataframe(
    df,
    numeric_interval_columns=[],
    categorical_columns=[],
    target_column=None,
    keep_width=False
):
    """
    One-shot preprocessing for anonymized datasets.
    - Splits numeric intervals
    - Encodes categoricals
    - Encodes target
    """
    df = process_numeric_intervals(df, numeric_interval_columns, keep_width=keep_width)
    df = encode_categoricals(df, categorical_columns)
    if target_column:
        df = encode_target(df, target_column)
    return df
