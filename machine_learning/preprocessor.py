from __future__ import annotations

"""Preprocessing utilities for k-anonymized datasets.

This module assumes that Domain Generalization Hierarchies (DGH) have already
been instantiated **per column** and passed in at construction time.

Supported encoding strategies per column (configure via `strategy` argument):
    * "interval"        – Numeric intervals → midpoint, width, depth
    * "one_hot"         – Categorical value (string) → one-hot vector
    * "ordinal"         – Categorical value → integer code
    * "path"            – DGH path → list[int] of node indices (variable length)
    * "level_indicator" – Categorical value → (code, depth) two-tuple

Suppressed values are denoted by "*" and treated as the DGH root (no specificity).
Unseen or suppressed tokens at inference map to UNK (index 0).

`Preprocessor.fit_transform(df)` → `pd.DataFrame` suitable for model training.
`encode_features(preprocessor, df)` → `np.ndarray` ready for scikit-learn.
"""

import re
import pickle
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

SUPPRESS_TOKEN = "*"
_INTERVAL_RE = re.compile(r"[\[]?(?P<low>-?\d+\.?\d*),\s*(?P<high>-?\d+\.?\d*)[\[]?")

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

LOWER_BOUND = -1e6
UPPER_BOUND = 1e6

def _parse_interval(interval_str: str) -> tuple[float, float]:
    """Extract numeric (low, high) from interval notation like "[20, 40["."""
    s = interval_str.strip()
    m = _INTERVAL_RE.match(s)

    if m:
        return float(m.group("low")), float(m.group("high"))

        # Handle >=X form
    if s.startswith(">="):
        value = float(s[2:])
        return value, UPPER_BOUND

        # Handle <X form
    if s.startswith("<"):
        value = float(s[1:])
        return LOWER_BOUND, value

        # Fallback: single number
    try:
        value = float(s)
        return value, value
    except ValueError:
        raise ValueError(f"Invalid interval format: {interval_str}")


# ----------------------------------------------------------------------------
# Encoder base
# ----------------------------------------------------------------------------


class ColumnEncoder:
    def fit(self, series: pd.Series, dgh) -> "ColumnEncoder":
        raise NotImplementedError

    def transform(self, series: pd.Series, dgh) -> pd.DataFrame:  # noqa: D401
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Concrete encoders
# ----------------------------------------------------------------------------


class IntervalEncoder(ColumnEncoder):
    def fit(self, series: pd.Series, dgh) -> "IntervalEncoder":
        return self

    def transform(self, series: pd.Series, dgh):  # type: ignore[override]
        mids, widths, depths = [], [], []
        for v in series.astype(str):
            if v == SUPPRESS_TOKEN:
                mids.append(0.0)
                widths.append(0.0)
                depths.append(0)
                continue
            low, high = _parse_interval(v)
            mids.append(0.5 * (low + high))
            widths.append(high - low)
            node = dgh.find_node_by_value(v) or dgh.root
            depths.append(node.depth())
        name = series.name
        return pd.DataFrame(
            {
                f"{name}_mid": mids,
                f"{name}_width": widths,
                f"{name}_depth": depths,
            },
            index=series.index,
        )


class OneHotEncoder(ColumnEncoder):
    def __init__(self):
        self.vocab: dict[str, int] = {}

    def fit(self, series: pd.Series, dgh) -> "OneHotEncoder":
        uniques = sorted(set(series.astype(str)) - {SUPPRESS_TOKEN})
        self.vocab = {
            tok: i for i, tok in enumerate(uniques, start=1)
        }  # reserve 0 for UNK/suppress
        return self

    def transform(self, series: pd.Series, dgh):  # type: ignore[override]
        n = len(self.vocab) + 1  # include UNK column at idx 0
        idx = series.astype(str).map(self.vocab).fillna(0).astype(int)
        one_hot = np.zeros((len(series), n), dtype=np.float32)
        one_hot[np.arange(len(series)), idx] = 1.0
        cols = [f"{series.name}_{cat}" for cat in sorted(self.vocab)]
        cols = [f"{series.name}_UNK"] + cols
        return pd.DataFrame(one_hot, columns=cols, index=series.index)


class OrdinalEncoder(ColumnEncoder):
    def __init__(self):
        self.vocab: dict[str, int] = {}

    def fit(self, series: pd.Series, dgh) -> "OrdinalEncoder":
        uniques = sorted(set(series.astype(str)) - {SUPPRESS_TOKEN})
        self.vocab = {tok: i + 1 for i, tok in enumerate(uniques)}  # 0 for UNK/suppress
        return self

    def transform(self, series: pd.Series, dgh):  # type: ignore[override]
        codes = series.astype(str).map(self.vocab).fillna(0).astype(int)
        return pd.DataFrame({series.name: codes}, index=series.index)


class PathEncoder(ColumnEncoder):
    def __init__(self):
        self.node_index: dict[Any, int] = {}
        self.max_depth: int = 0

    def fit(self, series: pd.Series, dgh) -> "PathEncoder":
        for node in dgh.get_all_nodes():
            if node.value not in self.node_index:
                self.node_index[node.value] = len(self.node_index) + 1
        self.max_depth = max(node.depth() for node in dgh.get_all_nodes())
        return self

    def transform(self, series: pd.Series, dgh):  # type: ignore[override]
        mat = np.zeros((len(series), self.max_depth + 1), dtype=np.int32)
        for i, v in enumerate(series.astype(str)):
            node = None
            if v == SUPPRESS_TOKEN:
                node = dgh.root
            else:
                node = dgh.find_node_by_value(v) or dgh.root
            path_vals = list(reversed([n.value for n in node.ancestors()]))
            codes = [self.node_index.get(val, 0) for val in path_vals]
            pad = self.max_depth + 1 - len(codes)
            mat[i, pad:] = codes
        cols = [f"{series.name}_p{i}" for i in range(self.max_depth + 1)]
        return pd.DataFrame(mat, columns=cols, index=series.index)


class LevelIndicatorEncoder(ColumnEncoder):
    def __init__(self):
        self.vocab: dict[str, int] = {}

    def fit(self, series: pd.Series, dgh) -> "LevelIndicatorEncoder":
        uniques = sorted(set(series.astype(str)) - {SUPPRESS_TOKEN})
        self.vocab = {tok: i + 1 for i, tok in enumerate(uniques)}
        return self

    def transform(self, series: pd.Series, dgh):  # type: ignore[override]
        codes = series.astype(str).map(self.vocab).fillna(0).astype(int)
        depths = []
        for v in series.astype(str):
            if v == SUPPRESS_TOKEN:
                depths.append(0)
            else:
                node = dgh.find_node_by_value(v) or dgh.root
                depths.append(node.depth())
        return pd.DataFrame(
            {
                f"{series.name}_code": codes,
                f"{series.name}_depth": depths,
            },
            index=series.index,
        )


# ----------------------------------------------------------------------------
# Encoder factory
# ----------------------------------------------------------------------------

_ENCODER_FACTORY = {
    "interval": IntervalEncoder,
    "one_hot": OneHotEncoder,
    "ordinal": OrdinalEncoder,
    "path": PathEncoder,
    "level_indicator": LevelIndicatorEncoder,
}

# ----------------------------------------------------------------------------
# Preprocessor orchestrator
# ----------------------------------------------------------------------------


class Preprocessor:
    def __init__(self, dghs: dict[str, Any], strategy: dict[str, str]):
        self.dghs = dghs
        self.strategy = strategy
        self.encoders: dict[str, ColumnEncoder] = {}

    def fit(self, df: pd.DataFrame):
        for col in df.columns:
            enc_key = self.strategy[col]
            encoder = _ENCODER_FACTORY[enc_key]()
            self.encoders[col] = encoder.fit(df[col], self.dghs[col])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for col in df.columns:
            parts.append(self.encoders[col].transform(df[col], self.dghs[col]))
        return pd.concat(parts, axis=1)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "dghs": self.dghs,
                    "strategy": self.strategy,
                    "encoders": self.encoders,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls(payload["dghs"], payload["strategy"])
        obj.encoders = payload["encoders"]
        return obj


# ----------------------------------------------------------------------------
# Public helper for scikit-learn
# ----------------------------------------------------------------------------


def encode_features(
    prep: Preprocessor, df: pd.DataFrame, *, y_col: Optional[str] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    feature_df = df.drop(columns=[y_col]) if y_col else df.copy()
    X_df = prep.transform(feature_df)
    X = X_df.to_numpy(dtype=np.float32)
    y = df[y_col].to_numpy() if y_col else None
    return X, y
