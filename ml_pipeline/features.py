"""Single source of truth for V4 training and inference feature engineering."""

from __future__ import annotations

import pandas as pd

NUMERICAL_BASE_FEATURES = [
    "cgpa",
    "backlogs",
    "coding_skills",
    "dsa_score",
    "aptitude_score",
    "communication_skills",
    "ml_knowledge",
    "system_design",
    "internships",
    "projects_count",
    "certifications",
    "hackathons",
    "open_source_contributions",
    "extracurriculars",
]

CATEGORICAL_FEATURES = ["branch", "college_tier"]
DERIVED_FEATURES = ["tech_score", "experience_score", "skill_score"]
ALL_NUMERICAL_FEATURES = NUMERICAL_BASE_FEATURES + DERIVED_FEATURES
ALL_FEATURES = ALL_NUMERICAL_FEATURES + CATEGORICAL_FEATURES

DEFAULTS = {
    "cgpa": 7.0,
    "backlogs": 0,
    "coding_skills": 6.0,
    "dsa_score": 5.5,
    "aptitude_score": 65.0,
    "communication_skills": 6.0,
    "ml_knowledge": 4.5,
    "system_design": 4.0,
    "internships": 1,
    "projects_count": 2,
    "certifications": 1,
    "hackathons": 1,
    "open_source_contributions": 0,
    "extracurriculars": 1,
    "branch": "CSE",
    "college_tier": "Tier-2",
}


def engineer_features(data: pd.DataFrame, fill_defaults: bool = False) -> pd.DataFrame:
    """Return the canonical V4 feature frame without mutating ``data``."""
    frame = data.copy()
    if fill_defaults:
        for column, value in DEFAULTS.items():
            if column not in frame.columns:
                frame[column] = value

    missing = [column for column in NUMERICAL_BASE_FEATURES + CATEGORICAL_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing V4 feature columns: {', '.join(missing)}")

    frame["tech_score"] = frame[["coding_skills", "dsa_score", "system_design", "ml_knowledge"]].mean(axis=1)
    frame["experience_score"] = (
        frame["internships"] * 2.0
        + frame["projects_count"]
        + frame["certifications"] * 0.5
        + frame["hackathons"] * 1.5
        + frame["open_source_contributions"] * 1.5
    )
    frame["skill_score"] = (
        (frame["aptitude_score"] / 10.0)
        + frame["communication_skills"]
        + frame["extracurriculars"]
    ) / 3.0
    return frame[ALL_FEATURES]
