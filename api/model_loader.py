"""Model loading utility for FastAPI application."""

from __future__ import annotations

from pathlib import Path

import joblib


def load_model(model_path: Path):
    """Load model artifact from disk.

    Args:
        model_path: Path to serialized model artifact.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    return joblib.load(model_path)
