# ==========================================================================================
# STUDENT PLACEMENT PREDICTION - INTERFACE V3
# ==========================================================================================
# Run:
# python interface_v3.py
#
# Required files:
# processed_data_v3/preprocessor_v3.joblib
# trained_models_v3/best_model_xgboost.joblib
# ==========================================================================================

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


# ==========================================================================================
# 1. PROJECT PATHS
# ==========================================================================================

PROJECT_DIR = Path(__file__).resolve().parent

PREPROCESSOR_PATH = (
    PROJECT_DIR / "processed_data_v3" / "preprocessor_v3.joblib"
)

MODEL_PATH = (
    PROJECT_DIR / "trained_models_v3" / "best_model_xgboost.joblib"
)


# ==========================================================================================
# 2. REQUIRED FEATURES
# ==========================================================================================

BASE_FEATURES = [
    "cgpa",
    "internships",
    "projects_count",
    "certifications",
    "aptitude_score",
    "communication_skills",
    "extracurricular_activities",
]

DERIVED_FEATURES = [
    "experience_score",
    "skill_score",
]

MODEL_FEATURES = BASE_FEATURES + DERIVED_FEATURES


# ==========================================================================================
# 3. LOAD SAVED MODEL AND PREPROCESSOR
# ==========================================================================================

def load_artifacts():
    """Load the saved preprocessing object and trained model."""

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found:\n{PREPROCESSOR_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)

    return preprocessor, model


# ==========================================================================================
# 4. FEATURE ENGINEERING
# ==========================================================================================

def create_features(student_data: dict) -> pd.DataFrame:
    """
    Reproduce the feature engineering used in data_preprocessing_v3.py.
    """

    df = pd.DataFrame([student_data])

    # Ensure all expected base columns exist
    for column in BASE_FEATURES:
        if column not in df.columns:
            df[column] = np.nan

    # Experience score
    # Same logical features used during V3 preprocessing
    df["experience_score"] = (
        df["internships"]
        + df["projects_count"]
        + df["certifications"]
    )

    # Skill score
    df["skill_score"] = (
        df["aptitude_score"]
        + df["communication_skills"]
        + df["extracurricular_activities"]
    ) / 3.0

    # Keep the exact feature order used by the model
    df = df[MODEL_FEATURES]

    return df


# ==========================================================================================
# 5. VALIDATE INPUTS
# ==========================================================================================

def validate_input(student_data: dict):
    """Validate student input values."""

    errors = []

    if not 0 <= student_data["cgpa"] <= 10:
        errors.append("CGPA must be between 0 and 10.")

    if student_data["internships"] < 0:
        errors.append("Internships cannot be negative.")

    if student_data["projects_count"] < 0:
        errors.append("Projects count cannot be negative.")

    if student_data["certifications"] < 0:
        errors.append("Certifications cannot be negative.")

    if not 0 <= student_data["aptitude_score"] <= 100:
        errors.append("Aptitude score must be between 0 and 100.")

    if not 0 <= student_data["communication_skills"] <= 10:
        errors.append("Communication skills must be between 0 and 10.")

    if not 0 <= student_data["extracurricular_activities"] <= 10:
        errors.append(
            "Extracurricular activities score must be between 0 and 10."
        )

    if errors:
        raise ValueError("\n".join(errors))


# ==========================================================================================
# 6. PREDICTION FUNCTION
# ==========================================================================================

def predict_student(student_data: dict, preprocessor, model):
    """Preprocess student data and generate prediction."""

    validate_input(student_data)

    # Create the same features used during training
    feature_df = create_features(student_data)

    # Apply the saved preprocessing pipeline
    processed_features = preprocessor.transform(feature_df)

    # Generate prediction
    prediction = int(model.predict(processed_features)[0])

    # Generate probability if supported
    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(processed_features)
        probability = float(probabilities[0][1])

    # Convert prediction to readable status
    if prediction == 1:
        status = "Placed"
    else:
        status = "Not Placed"

    # Risk interpretation
    if probability is not None:
        if probability >= 0.75:
            risk_level = "Low Risk"
        elif probability >= 0.50:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"
    else:
        risk_level = "Not Available"

    return {
        "status": status,
        "prediction": prediction,
        "probability": probability,
        "risk_level": risk_level,
        "feature_data": feature_df,
        "processed_features": processed_features,
    }


# ==========================================================================================
# 7. DISPLAY RESULTS
# ==========================================================================================

def display_result(student_data: dict, result: dict):
    """Display prediction results in the terminal."""

    print("\n" + "=" * 90)
    print("STUDENT PLACEMENT PREDICTION RESULT")
    print("=" * 90)

    print("\nStudent Information")
    print("-" * 40)

    print(f"CGPA:                         {student_data['cgpa']:.2f}")
    print(f"Internships:                  {student_data['internships']}")
    print(f"Projects:                     {student_data['projects_count']}")
    print(f"Certifications:               {student_data['certifications']}")
    print(f"Aptitude Score:               {student_data['aptitude_score']:.2f}")
    print(
        f"Communication Skills:         "
        f"{student_data['communication_skills']:.2f}"
    )
    print(
        f"Extracurricular Activities:   "
        f"{student_data['extracurricular_activities']:.2f}"
    )

    print("\nPrediction")
    print("-" * 40)

    print(f"Predicted Status:             {result['status']}")

    if result["probability"] is not None:
        print(
            f"Placement Probability:        "
            f"{result['probability'] * 100:.2f}%"
        )

    print(f"Risk Level:                   {result['risk_level']}")

    print("\nDerived Scores")
    print("-" * 40)

    feature_df = result["feature_data"]

    print(
        f"Experience Score:             "
        f"{feature_df['experience_score'].iloc[0]:.2f}"
    )

    print(
        f"Skill Score:                  "
        f"{feature_df['skill_score'].iloc[0]:.2f}"
    )

    print("\nModel Information")
    print("-" * 40)
    print("Model:                        XGBoost")
    print("Version:                      V3")
    print(f"Processed Features:           {result['processed_features'].shape[1]}")

    print("\n" + "=" * 90)


# ==========================================================================================
# 8. GET STUDENT INPUT
# ==========================================================================================

def get_student_input():
    """Collect student information from the terminal."""

    print("\n" + "=" * 90)
    print("ENTER STUDENT INFORMATION")
    print("=" * 90)

    print("\nAcademic Information")
    print("-" * 40)

    cgpa = float(input("CGPA (0-10): "))
    aptitude_score = float(input("Aptitude score (0-100): "))

    print("\nExperience Information")
    print("-" * 40)

    internships = int(input("Number of internships: "))
    projects_count = int(input("Number of projects: "))
    certifications = int(input("Number of certifications: "))

    print("\nSkill Information")
    print("-" * 40)

    communication_skills = float(
        input("Communication skills score (0-10): ")
    )

    extracurricular_activities = float(
        input("Extracurricular activities score (0-10): ")
    )

    student_data = {
        "cgpa": cgpa,
        "internships": internships,
        "projects_count": projects_count,
        "certifications": certifications,
        "aptitude_score": aptitude_score,
        "communication_skills": communication_skills,
        "extracurricular_activities": extracurricular_activities,
    }

    return student_data


# ==========================================================================================
# 9. MAIN PROGRAM
# ==========================================================================================

def main():
    print("=" * 90)
    print("STUDENT PLACEMENT PREDICTION - INTERFACE V3")
    print("=" * 90)

    print("\nLoading saved model and preprocessor...")

    try:
        preprocessor, model = load_artifacts()

        print("Model loaded successfully.")
        print("Preprocessor loaded successfully.")

    except Exception as error:
        print("\nERROR WHILE LOADING MODEL")
        print("-" * 40)
        print(error)
        sys.exit(1)

    while True:
        try:
            student_data = get_student_input()

            result = predict_student(
                student_data=student_data,
                preprocessor=preprocessor,
                model=model,
            )

            display_result(student_data, result)

        except ValueError as error:
            print("\nINPUT ERROR")
            print("-" * 40)
            print(error)

        except Exception as error:
            print("\nPREDICTION ERROR")
            print("-" * 40)
            print(error)

        print("\nWhat would you like to do?")
        print("1. Predict another student")
        print("2. Exit")

        choice = input("\nEnter choice: ").strip()

        if choice != "1":
            print("\nExiting Student Placement Prediction Interface.")
            print("Thank you!")
            break


# ==========================================================================================
# 10. RUN
# ==========================================================================================

if __name__ == "__main__":
    main() 
