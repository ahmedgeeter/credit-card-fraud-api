"""Data preprocessing module for fraud detection dataset."""

from __future__ import annotations

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV into a pandas DataFrame.

    Args:
        path: Absolute or relative path to the raw CSV file.

    Returns:
        Loaded DataFrame.
    """
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame, target_col: str = "Class") -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features X and target y."""
    x = df.drop(columns=[target_col])
    y = df[target_col]
    return x, y


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Run baseline cleaning logic.

    TODO:
    - Add missing value handling strategy if needed.
    - Add scaling strategy for time/amount features in Phase 2.
    """
    cleaned_df = df.copy()
    return cleaned_df
