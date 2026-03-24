"""Pydantic schemas for API IO contracts."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, create_model


def _transaction_feature_field_definitions() -> dict[str, tuple]:
    """Field definitions in creditcard.csv column order (excluding Class)."""
    fields: dict[str, tuple] = {
        "Time": (
            float,
            Field(description="Seconds elapsed between this transaction and the first in the dataset."),
        ),
    }
    for i in range(1, 29):
        fields[f"V{i}"] = (float, Field(description=f"Transformed feature V{i} (PCA / confidential)."))
    fields["Amount"] = (float, Field(description="Transaction amount."))
    return fields


class _TransactionFeaturesBase(BaseModel):
    """Reject unknown fields so payloads stay aligned with training columns."""

    model_config = ConfigDict(extra="forbid")


TransactionFeatures = create_model(
    "TransactionFeatures",
    __base__=_TransactionFeaturesBase,
    __doc__="Named input features matching `creditcard.csv` (excluding Class).",
    **_transaction_feature_field_definitions(),
)


FEATURE_NAMES: tuple[str, ...] = (
    "Time",
    *(f"V{i}" for i in range(1, 29)),
    "Amount",
)


def transaction_features_to_matrix(body: BaseModel) -> np.ndarray:
    """Build a single-row feature matrix in training column order."""
    row = body.model_dump()
    return np.array([[row[name] for name in FEATURE_NAMES]], dtype=np.float64)


class FraudPredictResponse(BaseModel):
    """Prediction result: fraud flag and fraud probability (class 1)."""

    fraud: bool
    risk_score: float = Field(..., description="Probability of fraud (positive class).")


# Backward compatibility with older imports (optional).
class PredictionRequest(BaseModel):
    """Deprecated: use ``TransactionFeatures``."""

    features: list[float] = Field(..., description="Ordered numeric feature vector.")


class PredictionResponse(BaseModel):
    """Deprecated: use ``FraudPredictResponse``."""

    fraud_probability: float
    predicted_class: int
