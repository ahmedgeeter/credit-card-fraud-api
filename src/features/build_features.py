"""Feature engineering functions."""

from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate model-ready features from input DataFrame.

    TODO:
    - Add optional log-transform for `Amount`.
    - Add robust scaling if model choice requires normalization.
    """
    feature_df = df.copy()
    return feature_df
