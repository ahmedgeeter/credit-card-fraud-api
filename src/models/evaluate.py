"""Model evaluation helpers."""

from __future__ import annotations

from typing import Any


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute evaluation metrics.

    TODO:
    - Add ROC-AUC, PR-AUC, recall, precision, F1.
    - Support threshold analysis for imbalanced data.
    """
    _ = (y_true, y_pred)
    return {}
