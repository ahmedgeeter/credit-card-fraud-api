"""Inference helpers for fraud prediction."""

from __future__ import annotations

from typing import Any


def predict_proba(model: Any, features: Any) -> float:
    """Return fraud probability for a single sample.

    TODO:
    - Validate input feature ordering.
    - Return calibrated probability if calibration is added.
    """
    proba = model.predict_proba(features)[0][1]
    return float(proba)
