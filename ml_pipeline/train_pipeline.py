"""Reproducible entry point for the canonical V4 model training pipeline."""

from pathlib import Path
import json
from datetime import datetime, timezone

import pandas as pd

from .features import (
    ALL_FEATURES,
    ALL_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    DERIVED_FEATURES,
    engineer_features,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "dataset" / "student_placement_synthetic.csv"
ARTIFACT_DIR = PROJECT_DIR / "trained_models_v4"
MANIFEST_PATH = ARTIFACT_DIR / "model_manifest.json"


def validate_dataset(frame: pd.DataFrame) -> None:
    required = set(ALL_FEATURES) - set(DERIVED_FEATURES)
    required |= {"placement_status", "salary_package_lpa"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if frame["placement_status"].isna().any():
        raise ValueError("placement_status contains missing values.")


def write_manifest(frame: pd.DataFrame) -> dict:
    """Write metadata consumed by operators and the API model loader."""
    metrics_path = ARTIFACT_DIR / "tech_model_metrics_v4.json"
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = {
        "model_version": "V4",
        "model_name": "XGBoost Technical Classifier",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(DATASET_PATH),
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "synthetic": True,
        },
        "features": {
            "all": ALL_FEATURES,
            "numeric": ALL_NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "derived": DERIVED_FEATURES,
        },
        "targets": {
            "classification": "placement_status",
            "regression": "salary_package_lpa",
        },
        "metrics": metrics,
        "artifacts": {
            "preprocessor": "tech_preprocessor_v4.joblib",
            "classifier": "best_tech_model_xgboost_v4.joblib",
            "regressor": "salary_regressor_v4.joblib",
            "feature_names": "feature_names_v4.json",
            "metrics": "tech_model_metrics_v4.json",
        },
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    frame = pd.read_csv(DATASET_PATH)
    validate_dataset(frame)
    print(f"Validated {len(frame):,} rows. Starting canonical V4 training...")

    # Keep the existing, tested V4 trainer as the implementation while exposing
    # one reproducible command for operators.
    from model_training_tech_v4 import train_models

    train_models()

    required_artifacts = [
        "tech_preprocessor_v4.joblib",
        "best_tech_model_xgboost_v4.joblib",
        "salary_regressor_v4.joblib",
        "feature_names_v4.json",
        "tech_model_metrics_v4.json",
        "model_manifest.json",
    ]
    missing = [name for name in required_artifacts if not (ARTIFACT_DIR / name).exists()]
    if missing:
        raise RuntimeError(f"Training completed without required artifacts: {', '.join(missing)}")
    print(f"Canonical V4 training completed. Artifacts: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
