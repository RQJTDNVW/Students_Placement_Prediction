"""Focused checks for the canonical V4 feature and artifact contract."""

import json
from pathlib import Path

import pandas as pd

from ml_pipeline.features import ALL_FEATURES, engineer_features


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "dataset" / "student_placement_synthetic.csv"
MANIFEST_PATH = PROJECT_DIR / "trained_models_v4" / "model_manifest.json"


def run_tests() -> None:
    raw = pd.read_csv(DATASET_PATH, nrows=3)
    features = engineer_features(raw)
    assert list(features.columns) == ALL_FEATURES
    assert features.shape == (3, len(ALL_FEATURES))
    assert features[["tech_score", "experience_score", "skill_score"]].notna().all().all()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["model_version"] == "V4"
    assert manifest["features"]["all"] == ALL_FEATURES
    for artifact in manifest["artifacts"].values():
        assert (MANIFEST_PATH.parent / artifact).exists(), artifact

    print("ML pipeline contract checks passed.")


if __name__ == "__main__":
    run_tests()
