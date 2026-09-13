# ==========================================================================================
# Student Placement Prediction — Model Interface
# ==========================================================================================
#
# Purpose:
#     Connector between dashboard.py and the trained machine learning model.
#
# This file:
#     1. Loads preprocessing artifacts
#     2. Loads the trained XGBoost model
#     3. Recreates the feature engineering used during training
#     4. Accepts student information
#     5. Returns placement prediction and probability
#
# This file does NOT contain frontend or dashboard code.
# ==========================================================================================

from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd


# ==========================================================================================
# 1. PROJECT PATHS
# ==========================================================================================

PROJECT_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = PROJECT_DIR / "processed_data"
MODELS_DIR = PROJECT_DIR / "trained_models"

PREPROCESSOR_PATH = PROCESSED_DIR / "preprocessor.joblib"
FEATURE_SELECTOR_PATH = PROCESSED_DIR / "feature_selector.joblib"
MODEL_PATH = MODELS_DIR / "xgboost.joblib"


# ==========================================================================================
# 2. ARTIFACT LOADING
# ==========================================================================================

_model = None
_preprocessor = None
_feature_selector = None


def load_artifacts():
    """
    Load the preprocessing pipeline, feature selector, and trained model.

    Returns:
        tuple:
            model,
            preprocessor,
            feature_selector
    """

    global _model
    global _preprocessor
    global _feature_selector

    if _model is not None:
        return _model, _preprocessor, _feature_selector

    missing_files = []

    if not PREPROCESSOR_PATH.exists():
        missing_files.append(str(PREPROCESSOR_PATH))

    if not FEATURE_SELECTOR_PATH.exists():
        missing_files.append(str(FEATURE_SELECTOR_PATH))

    if not MODEL_PATH.exists():
        missing_files.append(str(MODEL_PATH))

    if missing_files:
        raise FileNotFoundError(
            "The following required model files are missing:\n\n"
            + "\n".join(missing_files)
            + "\n\n"
            "Please run data_preprocessing.py and model_training.py first."
        )

    try:
        _preprocessor = joblib.load(PREPROCESSOR_PATH)
        _feature_selector = joblib.load(FEATURE_SELECTOR_PATH)
        _model = joblib.load(MODEL_PATH)

    except Exception as error:
        raise RuntimeError(
            f"Could not load the trained model artifacts.\n"
            f"Original error: {error}"
        ) from error

    return _model, _preprocessor, _feature_selector


# ==========================================================================================
# 3. FEATURE ENGINEERING
# ==========================================================================================

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate the same feature engineering used during model training.

    Important:
        The feature names and calculations must remain consistent with
        data_preprocessing.py.
    """

    df = data.copy()

    # ------------------------------------------------------------------
    # Academic score
    # ------------------------------------------------------------------

    academic_columns = [
        "cgpa",
        "dsa_score",
        "aptitude_score"
    ]

    available_academic_columns = [
        column for column in academic_columns
        if column in df.columns
    ]

    if available_academic_columns:
        df["academic_score"] = df[
            available_academic_columns
        ].mean(axis=1)

    # ------------------------------------------------------------------
    # Skill score
    # ------------------------------------------------------------------

    skill_columns = [
        "coding_skills",
        "dsa_score",
        "ml_knowledge",
        "system_design"
    ]

    available_skill_columns = [
        column for column in skill_columns
        if column in df.columns
    ]

    if available_skill_columns:
        df["skill_score"] = df[
            available_skill_columns
        ].mean(axis=1)

    # ------------------------------------------------------------------
    # Experience score
    # ------------------------------------------------------------------

    experience_columns = [
        "internships",
        "projects_count",
        "certifications",
        "hackathons",
        "open_source_contributions"
    ]

    available_experience_columns = [
        column for column in experience_columns
        if column in df.columns
    ]

    if available_experience_columns:
        df["experience_score"] = df[
            available_experience_columns
        ].sum(axis=1)

    # ------------------------------------------------------------------
    # Backlog category
    # ------------------------------------------------------------------

    if "backlogs" in df.columns:
        df["backlog_category"] = pd.cut(
            df["backlogs"],
            bins=[-np.inf, 0, 2, np.inf],
            labels=[
                "No Backlogs",
                "Low Backlogs",
                "High Backlogs"
            ]
        )

    # ------------------------------------------------------------------
    # Employability score
    # ------------------------------------------------------------------

    employability_columns = [
        "cgpa",
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
        "extracurriculars"
    ]

    available_employability_columns = [
        column for column in employability_columns
        if column in df.columns
    ]

    if available_employability_columns:
        df["employability_score"] = df[
            available_employability_columns
        ].mean(axis=1)

    return df


# ==========================================================================================
# 4. INPUT VALIDATION
# ==========================================================================================

def validate_student_data(student_data: Dict[str, Any]) -> None:
    """
    Validate the student input before prediction.
    """

    required_columns = [
        "branch",
        "college_tier",
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
        "extracurriculars"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in student_data
    ]

    if missing_columns:
        raise ValueError(
            "Missing required student fields:\n"
            + ", ".join(missing_columns)
        )

    numeric_columns = [
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
        "extracurriculars"
    ]

    for column in numeric_columns:
        value = student_data[column]

        if value is None:
            raise ValueError(f"{column} cannot be empty.")

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{column} must be a numeric value."
            )

        if not np.isfinite(numeric_value):
            raise ValueError(
                f"{column} must contain a valid finite number."
            )

        if numeric_value < 0:
            raise ValueError(
                f"{column} cannot be negative."
            )

    if not 0 <= float(student_data["cgpa"]) <= 10:
        raise ValueError("CGPA must be between 0 and 10.")

    score_columns = [
        "coding_skills",
        "dsa_score",
        "aptitude_score",
        "communication_skills",
        "ml_knowledge",
        "system_design",
        "extracurriculars"
    ]

    for column in score_columns:
        if float(student_data[column]) > 100:
            raise ValueError(
                f"{column} must be between 0 and 100."
            )

    if not str(student_data["branch"]).strip():
        raise ValueError("Branch cannot be empty.")

    if not str(student_data["college_tier"]).strip():
        raise ValueError("College tier cannot be empty.")


# ==========================================================================================
# 5. PREPARE INPUT DATA
# ==========================================================================================

def prepare_student_data(student_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert a student dictionary into the DataFrame format expected
    by the preprocessing pipeline.
    """

    validate_student_data(student_data)

    df = pd.DataFrame([student_data])

    # Ensure numeric values have the correct type
    numeric_columns = [
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
        "extracurriculars"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Recreate training features
    df = create_features(df)

    return df


# ==========================================================================================
# 6. PREDICTION FUNCTION
# ==========================================================================================

def predict_student(
    student_data: Dict[str, Any],
    model_name: str = "XGBoost"
) -> Dict[str, Any]:
    """
    Predict whether a student is likely to be placed.

    Args:
        student_data:
            Dictionary containing student information.

        model_name:
            Currently uses the trained XGBoost model.

    Returns:
        Dictionary containing:
            prediction
            status
            probability
            model
    """

    model, preprocessor, feature_selector = load_artifacts()

    df = prepare_student_data(student_data)

    try:
        # Apply the exact preprocessing used during training
        transformed_data = preprocessor.transform(df)

        # Apply the trained feature selector
        selected_data = feature_selector.transform(
            transformed_data
        )

        # Generate prediction
        prediction = int(model.predict(selected_data)[0])

        # Generate probability if supported
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(
                selected_data
            )[0]

            placement_probability = float(
                probabilities[1] * 100
            )

        else:
            decision_score = float(
                model.decision_function(selected_data)[0]
            )

            placement_probability = float(
                1 / (1 + np.exp(-decision_score)) * 100
            )

        status = (
            "Placed"
            if prediction == 1
            else "Not Placed"
        )

        return {
            "prediction": prediction,
            "status": status,
            "probability": round(
                placement_probability,
                2
            ),
            "model": model_name,
            "processed_feature_count": int(
                selected_data.shape[1]
            )
        }

    except Exception as error:
        raise RuntimeError(
            "Prediction failed while processing the student data.\n"
            f"Original error: {error}"
        ) from error


# ==========================================================================================
# 7. SIMPLE ALIAS FOR DASHBOARD CONNECTION
# ==========================================================================================

def predict_from_dict(
    student_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Alias function for dashboard.py.

    Example:
        result = predict_from_dict(student_data)
    """

    return predict_student(student_data)


# ==========================================================================================
# 8. MODEL STATUS CHECK
# ==========================================================================================

def get_model_status() -> Dict[str, Any]:
    """
    Return information about whether the model files are available.
    """

    return {
        "project_directory": str(PROJECT_DIR),
        "model_exists": MODEL_PATH.exists(),
        "preprocessor_exists": PREPROCESSOR_PATH.exists(),
        "feature_selector_exists": FEATURE_SELECTOR_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "preprocessor_path": str(PREPROCESSOR_PATH),
        "feature_selector_path": str(FEATURE_SELECTOR_PATH)
    }


# ==========================================================================================
# 9. TESTING
# ==========================================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("STUDENT PLACEMENT PREDICTION — INTERFACE TEST")
    print("=" * 80)

    print("\nChecking model files...")

    status = get_model_status()

    for key, value in status.items():
        print(f"{key}: {value}")

    sample_student = {
        "branch": "Computer Science",
        "college_tier": "Tier 1",
        "cgpa": 8.5,
        "backlogs": 0,
        "coding_skills": 85,
        "dsa_score": 80,
        "aptitude_score": 82,
        "communication_skills": 78,
        "ml_knowledge": 75,
        "system_design": 70,
        "internships": 2,
        "projects_count": 4,
        "certifications": 3,
        "hackathons": 2,
        "open_source_contributions": 1,
        "extracurriculars": 80
    }

    print("\nRunning sample prediction...")

    try:
        result = predict_student(sample_student)

        print("\nPrediction Result")
        print("-" * 40)

        for key, value in result.items():
            print(f"{key}: {value}")

    except Exception as error:
        print("\nERROR:")
        print(error)






    #to run the frontend server, use the following command in PowerShell:
    # ========================================================================================== 
    # cd "C:\AI\Student Placement Prediction\frontend"; $env:Path="C:\Program Files\nodejs;$env:Path"; & "C:\Program Files\nodejs\npm.cmd" run dev