"""Full training pipeline for credit card fraud detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, RAW_DATA_PATH, ensure_directories


@dataclass
class PreprocessedData:
    """Train/validation splits (scaled) plus full raw feature matrix for refit."""

    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    X_full: pd.DataFrame
    y_full: pd.Series


def load_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load raw Kaggle credit card fraud CSV.

    Args:
        path: CSV path. Defaults to ``data/raw/creditcard.csv`` under project root.

    Returns:
        Raw dataframe including ``Class`` target column.
    """
    csv_path = Path(path) if path is not None else RAW_DATA_PATH
    return pd.read_csv(csv_path)


def _drop_irrelevant_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Drop non-feature columns if present (e.g. row id)."""
    out = df.copy()
    drop: list[str] = []
    for col in ("Id", "ID", "index"):
        if col in out.columns and col != target_col:
            drop.append(col)
    if drop:
        out = out.drop(columns=drop)
    return out


def preprocess(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreprocessedData:
    """Prepare stratified train/validation splits with imputation and scaling.

    Fits ``SimpleImputer`` and ``StandardScaler`` on the training split only.

    Args:
        df: Raw dataframe with features and target.
        target_col: Name of binary target (1=fraud).
        test_size: Validation fraction.
        random_state: RNG seed for splitting.

    Returns:
        Scaled arrays for training and validation, plus full (pre-split) X/y for
        refitting the production pipeline on all data.
    """
    cleaned = _drop_irrelevant_columns(df, target_col)
    if target_col not in cleaned.columns:
        raise ValueError(f"Target column {target_col!r} not found.")

    X = cleaned.drop(columns=[target_col])
    y = cleaned[target_col]

    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train_df)
    X_val_imp = imputer.transform(X_val_df)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)

    return PreprocessedData(
        X_train=X_train_scaled.astype(np.float64),
        X_val=X_val_scaled.astype(np.float64),
        y_train=np.asarray(y_train),
        y_val=np.asarray(y_val),
        X_full=X,
        y_full=y,
    )


def train_models(X_train: np.ndarray, y_train: np.ndarray) -> dict[str, BaseEstimator]:
    """Train logistic regression and random forest on scaled training features.

    Args:
        X_train: Scaled feature matrix.
        y_train: Binary labels.

    Returns:
        Mapping model name -> fitted estimator.
    """
    models: dict[str, BaseEstimator] = {
        # sklearn uses class_weight='balanced' (equivalent intent to "balanced").
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        ),
        "RandomForestClassifier": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }
    fitted: dict[str, BaseEstimator] = {}
    for name, est in models.items():
        est.fit(X_train, y_train)
        fitted[name] = est
    return fitted


def evaluate(model: BaseEstimator, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
    """Compute ROC-AUC, PR-AUC, and F1 on validation data."""
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    return {
        "roc_auc": float(roc_auc_score(y_val, y_proba)),
        "pr_auc": float(average_precision_score(y_val, y_proba)),
        "f1": float(f1_score(y_val, y_pred)),
    }


def _metrics_comparison_table(results: dict[str, dict[str, float]]) -> str:
    """Format metrics as an aligned table for stdout."""
    rows: list[dict[str, Any]] = []
    for name, m in results.items():
        rows.append(
            {
                "Model": name,
                "ROC-AUC": round(m["roc_auc"], 4),
                "PR-AUC": round(m["pr_auc"], 4),
                "F1": round(m["f1"], 4),
            }
        )
    out = pd.DataFrame(rows)
    return out.to_string(index=False)


def save_best(
    results: dict[str, dict[str, float]],
    models: dict[str, BaseEstimator],
    X_full: pd.DataFrame,
    y_full: pd.Series,
    models_dir: Path | None = None,
) -> tuple[str, dict[str, float]]:
    """Persist best model (by validation ROC-AUC) and preprocessing artifacts.

    Saves (joblib, ``.pkl`` suffix):

    - ``full_pipeline.pkl``: imputer + scaler + best classifier (end-to-end).
    - ``preprocessing_pipeline.pkl``: imputer + scaler only.
    - ``scaler.pkl``: fitted ``StandardScaler``.
    - ``best_model.pkl``: fitted classifier step only.

    Refits the full pipeline on the entire dataset (all rows in ``X_full``).

    Args:
        results: Metric dict per model name from ``evaluate``.
        models: Fitted estimators keyed by name (used to choose cloned type/params).
        X_full: Full feature dataframe (same columns as training).
        y_full: Full target series.
        models_dir: Directory for artifacts. Defaults to ``models/`` under project root.

    Returns:
        Tuple of (best model name, that model's validation metrics dict).
    """
    out_dir = models_dir if models_dir is not None else MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_fitted = models[best_name]

    full_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clone(best_fitted)),
        ]
    )
    full_pipeline.fit(X_full, y_full)

    preprocessing_only = Pipeline(
        [
            ("imputer", full_pipeline.named_steps["imputer"]),
            ("scaler", full_pipeline.named_steps["scaler"]),
        ]
    )

    joblib.dump(full_pipeline, out_dir / "full_pipeline.pkl")
    joblib.dump(preprocessing_only, out_dir / "preprocessing_pipeline.pkl")
    joblib.dump(full_pipeline.named_steps["scaler"], out_dir / "scaler.pkl")
    joblib.dump(full_pipeline.named_steps["clf"], out_dir / "best_model.pkl")

    return best_name, results[best_name]


def main() -> None:
    """Run EDA, train two models, compare metrics, save best artifacts."""
    ensure_directories()
    df = load_data()

    print("=== EDA ===")
    print(f"Shape: {df.shape}")
    print("\nClass distribution (count):")
    print(df["Class"].value_counts().to_string())
    print("\nClass distribution (proportion):")
    print(df["Class"].value_counts(normalize=True).round(6).to_string())
    missing_total = int(df.isnull().sum().sum())
    print(f"\nTotal missing values (all columns): {missing_total}")
    if missing_total:
        print("\nMissing values per column (non-zero only):")
        miss = df.isnull().sum()
        print(miss[miss > 0].to_string())

    prep = preprocess(df)
    fitted = train_models(prep.X_train, prep.y_train)
    results = {name: evaluate(model, prep.X_val, prep.y_val) for name, model in fitted.items()}

    print("\n=== Validation metrics (comparison) ===")
    print(_metrics_comparison_table(results))

    best_name, best_metrics = save_best(results, fitted, prep.X_full, prep.y_full)

    print("\n=== Best model (by validation ROC-AUC) ===")
    print(f"Selected: {best_name}")
    print(f"  ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:  {best_metrics['pr_auc']:.4f}")
    print(f"  F1:      {best_metrics['f1']:.4f}")
    print(f"\nArtifacts written to: {MODELS_DIR.resolve()}")
    print("  - full_pipeline.pkl")
    print("  - preprocessing_pipeline.pkl")
    print("  - scaler.pkl")
    print("  - best_model.pkl")


if __name__ == "__main__":
    main()
