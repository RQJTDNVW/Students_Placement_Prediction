# ==========================================================================================
# STUDENT PLACEMENT PREDICTION — DATA INTEGRATION V2
# ==========================================================================================
# Combines the current synthetic dataset with the external placement dataset.
#
# Important:
#   - Salary is removed because it causes target leakage.
#   - Student IDs are removed because they are not predictive features.
#   - The synthetic "extracurriculars" column is mapped to
#     "extracurricular_activities".
#   - Dataset-specific columns are retained only for integration compatibility.
#   - Model training should later exclude columns that are completely missing.
# ==========================================================================================

from pathlib import Path
import json
import re
import urllib.request

import numpy as np
import pandas as pd


# ==========================================================================================
# 1. PROJECT PATHS
# ==========================================================================================

PROJECT_DIR = Path(r"C:\AI\Student Placement Prediction")

CURRENT_DATASET_PATH = (
    PROJECT_DIR
    / "dataset"
    / "student_placement_synthetic.csv"
)

EXTERNAL_DATASET_PATH = (
    PROJECT_DIR
    / "dataset"
    / "external"
    / "placement_prediction_10000.csv"
)

EXTERNAL_DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "bankystack/Placement-Prediction/master/placementdata.csv"
)

OUTPUT_DIR = PROJECT_DIR / "integrated_data_v2"

SYNTHETIC_OUTPUT_PATH = (
    OUTPUT_DIR / "synthetic_normalized_v2.csv"
)

EXTERNAL_OUTPUT_PATH = (
    OUTPUT_DIR / "external_normalized_v2.csv"
)

INTEGRATED_OUTPUT_PATH = (
    OUTPUT_DIR / "integrated_dataset_v2.csv"
)

REPORT_OUTPUT_PATH = (
    OUTPUT_DIR / "integration_report_v2.json"
)


# ==========================================================================================
# 2. CONFIGURATION
# ==========================================================================================

TARGET_COLUMN = "placement_status"

SHARED_FEATURES = [
    "cgpa",
    "ssc_marks",
    "hsc_marks",
    "internships",
    "projects_count",
    "certifications",
    "aptitude_score",
    "communication_skills",
    "extracurricular_activities",
    "placement_training",
]

DERIVED_FEATURES = [
    "academic_score",
    "experience_score",
    "skill_score",
]

FINAL_COLUMNS = (
    SHARED_FEATURES
    + DERIVED_FEATURES
    + [
        TARGET_COLUMN,
        "source_dataset",
    ]
)


# ==========================================================================================
# 3. COLUMN HELPERS
# ==========================================================================================

def normalize_column_name(column_name):
    """
    Converts different column-name formats into one consistent format.

    Examples:
        SSC_Marks                    -> ssc_marks
        Workshops/Certifications     -> workshops_certifications
        ExtracurricularActivities    -> extracurricularactivities
    """

    column_name = str(column_name).strip().lower()

    column_name = re.sub(
        r"[^a-z0-9]+",
        "_",
        column_name,
    )

    column_name = re.sub(
        r"_+",
        "_",
        column_name,
    )

    return column_name.strip("_")


def build_column_map(df):
    """
    Creates a normalized-name-to-original-name mapping.
    """

    return {
        normalize_column_name(column): column
        for column in df.columns
    }


def find_column(df, possible_names):
    """
    Finds the first existing column from a list of possible names.
    """

    column_map = build_column_map(df)

    for possible_name in possible_names:
        normalized_name = normalize_column_name(
            possible_name
        )

        if normalized_name in column_map:
            return column_map[normalized_name]

    return None


def get_column(df, possible_names, default_value=np.nan):
    """
    Returns a matching column.

    If no matching column exists, returns a Series filled
    with the supplied default value.
    """

    found_column = find_column(
        df,
        possible_names,
    )

    if found_column is not None:
        return df[found_column].copy()

    print(
        "WARNING: Column not found:",
        ", ".join(possible_names),
        f". Using default value: {default_value}",
    )

    return pd.Series(
        default_value,
        index=df.index,
    )


# ==========================================================================================
# 4. VALUE-CONVERSION HELPERS
# ==========================================================================================

def convert_numeric(series):
    """
    Converts values to numeric values.
    Invalid values become NaN.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def clean_target_value(value):
    """
    Converts placement-status values into:
        1 = Placed
        0 = Not placed
        NaN = Unknown
    """

    if pd.isna(value):
        return np.nan

    # Numeric values
    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        if value == 1:
            return 1

        if value == 0:
            return 0

        return np.nan

    text = str(value).strip().lower()

    # Normalize separators and spaces
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)

    positive_values = {
        "1",
        "yes",
        "y",
        "true",
        "placed",
        "place",
        "selected",
        "select",
        "pass",
        "passed",
        "successful",
        "success",
        "accepted",
        "hired",
        "employed",
    }

    negative_values = {
        "0",
        "no",
        "n",
        "false",
        "not placed",
        "notplaced",
        "not place",
        "not selected",
        "notselected",
        "not select",
        "unplaced",
        "unselected",
        "rejected",
        "reject",
        "fail",
        "failed",
        "unsuccessful",
        "unsuccess",
        "not hired",
        "unemployed",
    }

    if text in positive_values:
        return 1

    if text in negative_values:
        return 0

    # Catch phrases such as "Not Placed"
    if "not" in text and (
        "place" in text
        or "select" in text
        or "hire" in text
    ):
        return 0

    if (
        "place" in text
        or "select" in text
        or "hire" in text
    ):
        return 1

    # Handle numeric strings
    try:
        numeric_value = float(text)

        if numeric_value == 1:
            return 1

        if numeric_value == 0:
            return 0

    except ValueError:
        pass

    return np.nan


def clean_activity_value(value):
    """
    Converts extracurricular activity values into numeric scores.

    Supported examples:
        0, 1, 2, 3
        Yes, No
        Low, Medium, High
    """

    if pd.isna(value):
        return np.nan

    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        return float(value)

    text = str(value).strip().lower()

    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)

    activity_mapping = {
        "no": 0,
        "none": 0,
        "false": 0,
        "0": 0,
        "yes": 1,
        "true": 1,
        "1": 1,
        "low": 1,
        "medium": 2,
        "moderate": 2,
        "high": 3,
        "excellent": 3,
        "2": 2,
        "3": 3,
    }

    if text in activity_mapping:
        return activity_mapping[text]

    try:
        return float(text)

    except ValueError:
        return np.nan


# ==========================================================================================
# 5. DOWNLOAD EXTERNAL DATASET
# ==========================================================================================

def download_external_dataset():
    """
    Downloads the external dataset if it does not already exist.
    """

    if EXTERNAL_DATASET_PATH.exists():
        print("\nExternal dataset already exists.")
        print(f"Path: {EXTERNAL_DATASET_PATH}")
        return

    EXTERNAL_DATASET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nDownloading external dataset...")

    try:
        urllib.request.urlretrieve(
            EXTERNAL_DATASET_URL,
            EXTERNAL_DATASET_PATH,
        )

        print("External dataset downloaded successfully.")
        print(f"Saved to: {EXTERNAL_DATASET_PATH}")

    except Exception as error:
        raise RuntimeError(
            f"Could not download external dataset: {error}"
        )


# ==========================================================================================
# 6. LOAD DATASETS
# ==========================================================================================

def load_current_dataset():
    """
    Loads the synthetic dataset.
    """

    if not CURRENT_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Current dataset not found:\n{CURRENT_DATASET_PATH}"
        )

    df = pd.read_csv(
        CURRENT_DATASET_PATH
    )

    print("\nCurrent dataset loaded:")
    print(f"Shape: {df.shape}")

    print("\nCurrent dataset columns:")
    print(df.columns.tolist())

    return df


def load_external_dataset():
    """
    Loads the external dataset.
    """

    if not EXTERNAL_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"External dataset not found:\n{EXTERNAL_DATASET_PATH}"
        )

    df = pd.read_csv(
        EXTERNAL_DATASET_PATH
    )

    print("\nExternal dataset loaded:")
    print(f"Shape: {df.shape}")

    print("\nExternal dataset columns:")
    print(df.columns.tolist())

    return df


# ==========================================================================================
# 7. NORMALIZE SYNTHETIC DATASET
# ==========================================================================================

def normalize_current_dataset(df):
    """
    Converts the synthetic dataset into the shared schema.
    """

    print("\nNormalizing current synthetic dataset...")

    normalized = pd.DataFrame(
        index=df.index
    )

    normalized["cgpa"] = convert_numeric(
        get_column(
            df,
            [
                "cgpa",
                "CGPA",
            ],
        )
    )

    normalized["ssc_marks"] = convert_numeric(
        get_column(
            df,
            [
                "ssc_marks",
                "ssc_percentage",
                "ssc",
                "10th_marks",
                "tenth_marks",
            ],
        )
    )

    normalized["hsc_marks"] = convert_numeric(
        get_column(
            df,
            [
                "hsc_marks",
                "hsc_percentage",
                "hsc",
                "12th_marks",
                "twelfth_marks",
            ],
        )
    )

    normalized["internships"] = convert_numeric(
        get_column(
            df,
            [
                "internships",
                "internship",
            ],
        )
    )

    normalized["projects_count"] = convert_numeric(
        get_column(
            df,
            [
                "projects_count",
                "projects",
                "project_count",
            ],
        )
    )

    normalized["certifications"] = convert_numeric(
        get_column(
            df,
            [
                "certifications",
                "workshops_certifications",
                "workshops",
                "certification_count",
            ],
        )
    )

    normalized["aptitude_score"] = convert_numeric(
        get_column(
            df,
            [
                "aptitude_score",
                "aptitude_test_score",
                "aptitudetestscore",
                "aptitude",
            ],
        )
    )

    normalized["communication_skills"] = convert_numeric(
        get_column(
            df,
            [
                "communication_skills",
                "soft_skills_rating",
                "softskillsrating",
                "soft_skill_rating",
                "communication",
            ],
        )
    )

    # IMPORTANT FIX:
    # Synthetic dataset uses "extracurriculars".
    # It is mapped to "extracurricular_activities".
    normalized["extracurricular_activities"] = (
        get_column(
            df,
            [
                "extracurriculars",
                "extracurricular_activities",
                "extracurricular",
                "extra_curricular_activities",
                "extra_curricular",
                "extracurricular_score",
            ],
        ).apply(clean_activity_value)
    )

    normalized["placement_training"] = convert_numeric(
        get_column(
            df,
            [
                "placement_training",
                "placementtraining",
                "training",
                "placement_preparation",
            ],
        )
    )

    normalized[TARGET_COLUMN] = (
        get_column(
            df,
            [
                "placement_status",
                "placementstatus",
                "placed",
                "placement",
            ],
        ).apply(clean_target_value)
    )

    normalized["source_dataset"] = "synthetic"

    return normalized


# ==========================================================================================
# 8. NORMALIZE EXTERNAL DATASET
# ==========================================================================================

def normalize_external_dataset(df):
    """
    Converts the external dataset into the shared schema.
    """

    print("\nNormalizing external dataset...")

    normalized = pd.DataFrame(
        index=df.index
    )

    normalized["cgpa"] = convert_numeric(
        get_column(
            df,
            [
                "CGPA",
                "cgpa",
            ],
        )
    )

    normalized["ssc_marks"] = convert_numeric(
        get_column(
            df,
            [
                "SSC_Marks",
                "ssc_marks",
                "ssc_percentage",
                "ssc",
            ],
        )
    )

    normalized["hsc_marks"] = convert_numeric(
        get_column(
            df,
            [
                "HSC_Marks",
                "hsc_marks",
                "hsc_percentage",
                "hsc",
            ],
        )
    )

    normalized["internships"] = convert_numeric(
        get_column(
            df,
            [
                "Internships",
                "internships",
                "internship",
            ],
        )
    )

    normalized["projects_count"] = convert_numeric(
        get_column(
            df,
            [
                "Projects",
                "projects",
                "projects_count",
                "project_count",
            ],
        )
    )

    normalized["certifications"] = convert_numeric(
        get_column(
            df,
            [
                "Workshops/Certifications",
                "workshops_certifications",
                "workshops",
                "certifications",
            ],
        )
    )

    normalized["aptitude_score"] = convert_numeric(
        get_column(
            df,
            [
                "AptitudeTestScore",
                "aptitude_test_score",
                "aptitude_score",
                "aptitude",
            ],
        )
    )

    normalized["communication_skills"] = convert_numeric(
        get_column(
            df,
            [
                "SoftSkillsRating",
                "soft_skills_rating",
                "softskillsrating",
                "communication_skills",
                "communication",
            ],
        )
    )

    normalized["extracurricular_activities"] = (
        get_column(
            df,
            [
                "ExtracurricularActivities",
                "ExtraCurricularActivities",
                "extracurricular_activities",
                "extracurriculars",
                "extracurricular",
                "extra_curricular_activities",
            ],
        ).apply(clean_activity_value)
    )

    normalized["placement_training"] = convert_numeric(
        get_column(
            df,
            [
                "PlacementTraining",
                "placement_training",
                "placementtraining",
                "training",
            ],
        )
    )

    # Print the original target values for debugging.
    original_target_column = find_column(
        df,
        [
            "PlacementStatus",
            "placement_status",
            "placementstatus",
            "placed",
            "placement",
        ],
    )

    if original_target_column is not None:
        print("\nExternal target values before cleaning:")
        print(
            df[original_target_column]
            .value_counts(dropna=False)
            .head(30)
        )

    normalized[TARGET_COLUMN] = (
        get_column(
            df,
            [
                "PlacementStatus",
                "placement_status",
                "placementstatus",
                "placed",
                "placement",
            ],
        ).apply(clean_target_value)
    )

    normalized["source_dataset"] = "external_10000"

    return normalized


# ==========================================================================================
# 9. ADD DERIVED FEATURES
# ==========================================================================================

def add_derived_features(df):
    """
    Creates common derived features.

    Missing values are ignored during calculations.
    """

    df = df.copy()

    df["academic_score"] = df[
        [
            "cgpa",
            "ssc_marks",
            "hsc_marks",
        ]
    ].mean(
        axis=1,
        skipna=True,
    )

    df["experience_score"] = df[
        [
            "internships",
            "projects_count",
            "certifications",
        ]
    ].sum(
        axis=1,
        skipna=True,
    )

    df["skill_score"] = df[
        [
            "aptitude_score",
            "communication_skills",
            "extracurricular_activities",
        ]
    ].mean(
        axis=1,
        skipna=True,
    )

    return df


# ==========================================================================================
# 10. CLEAN DATASET
# ==========================================================================================

def clean_dataset(df, dataset_name):
    """
    Removes duplicates and rows with unknown target values.
    """

    print(f"\nCleaning {dataset_name} dataset...")

    df = df.copy()

    initial_rows = len(df)

    df = df.drop_duplicates()

    removed_duplicates = (
        initial_rows - len(df)
    )

    print(
        f"Removed duplicate rows: {removed_duplicates}"
    )

    missing_target_count = (
        df[TARGET_COLUMN]
        .isna()
        .sum()
    )

    if missing_target_count > 0:
        print(
            "Removing rows with missing target:",
            missing_target_count,
        )

        df = df.dropna(
            subset=[TARGET_COLUMN]
        )

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(int)
    )

    return df


# ==========================================================================================
# 11. REMOVE LEAKAGE AND ID COLUMNS
# ==========================================================================================

def remove_leakage_columns(df):
    """
    Removes salary and student-ID columns.
    """

    df = df.copy()

    leakage_columns = {
        "salary_package_lpa",
        "salary",
        "salary_package",
        "package",
        "salarypackage",
    }

    id_columns = {
        "studentid",
        "student_id",
        "slno",
        "serial_no",
        "id",
    }

    columns_to_remove = []

    for column in df.columns:
        normalized_column = normalize_column_name(
            column
        )

        if normalized_column in leakage_columns:
            columns_to_remove.append(column)

        elif normalized_column in id_columns:
            columns_to_remove.append(column)

    columns_to_remove = list(
        dict.fromkeys(columns_to_remove)
    )

    if columns_to_remove:
        print(
            "\nRemoving leakage/ID columns:",
            columns_to_remove,
        )

        df = df.drop(
            columns=columns_to_remove,
            errors="ignore",
        )

    return df


# ==========================================================================================
# 12. ALIGN FINAL COLUMNS
# ==========================================================================================

def align_final_columns(df):
    """
    Ensures that every dataset has the same columns and order.
    """

    df = df.copy()

    for column in FINAL_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    df = df[FINAL_COLUMNS]

    return df


# ==========================================================================================
# 13. CREATE REPORT
# ==========================================================================================

def create_integration_report(
    synthetic_df,
    external_df,
    integrated_df,
):
    """
    Saves an integration report as JSON.
    """

    report = {
        "project": "Student Placement Prediction",
        "integration_version": "v2",
        "current_dataset_path": str(
            CURRENT_DATASET_PATH
        ),
        "external_dataset_path": str(
            EXTERNAL_DATASET_PATH
        ),
        "external_dataset_url": EXTERNAL_DATASET_URL,
        "target_column": TARGET_COLUMN,
        "shared_features": SHARED_FEATURES,
        "derived_features": DERIVED_FEATURES,
        "final_columns": FINAL_COLUMNS,
        "dataset_shapes": {
            "synthetic": list(
                synthetic_df.shape
            ),
            "external": list(
                external_df.shape
            ),
            "integrated": list(
                integrated_df.shape
            ),
        },
        "source_distribution": (
            integrated_df["source_dataset"]
            .value_counts()
            .to_dict()
        ),
        "target_distribution": (
            integrated_df[TARGET_COLUMN]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "missing_values": (
            integrated_df.isna()
            .sum()
            .loc[
                lambda values: values > 0
            ]
            .to_dict()
        ),
    }

    with open(
        REPORT_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            default=str,
        )

    return report


# ==========================================================================================
# 14. MAIN PIPELINE
# ==========================================================================================

def run_integration():
    print("=" * 100)
    print("STUDENT PLACEMENT PREDICTION — DATA INTEGRATION V2")
    print("=" * 100)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Download and load datasets
    # ------------------------------------------------------------------

    download_external_dataset()

    current_df = load_current_dataset()
    external_df = load_external_dataset()

    # ------------------------------------------------------------------
    # Normalize datasets
    # ------------------------------------------------------------------

    synthetic_normalized = (
        normalize_current_dataset(
            current_df
        )
    )

    external_normalized = (
        normalize_external_dataset(
            external_df
        )
    )

    # ------------------------------------------------------------------
    # Clean datasets
    # ------------------------------------------------------------------

    synthetic_normalized = clean_dataset(
        synthetic_normalized,
        "synthetic",
    )

    external_normalized = clean_dataset(
        external_normalized,
        "external",
    )

    # ------------------------------------------------------------------
    # Add derived features
    # ------------------------------------------------------------------

    synthetic_normalized = add_derived_features(
        synthetic_normalized
    )

    external_normalized = add_derived_features(
        external_normalized
    )

    # ------------------------------------------------------------------
    # Remove leakage and ID columns
    # ------------------------------------------------------------------

    synthetic_normalized = (
        remove_leakage_columns(
            synthetic_normalized
        )
    )

    external_normalized = (
        remove_leakage_columns(
            external_normalized
        )
    )

    # ------------------------------------------------------------------
    # Align columns
    # ------------------------------------------------------------------

    synthetic_normalized = align_final_columns(
        synthetic_normalized
    )

    external_normalized = align_final_columns(
        external_normalized
    )

    # ------------------------------------------------------------------
    # Save normalized datasets
    # ------------------------------------------------------------------

    synthetic_normalized.to_csv(
        SYNTHETIC_OUTPUT_PATH,
        index=False,
    )

    external_normalized.to_csv(
        EXTERNAL_OUTPUT_PATH,
        index=False,
    )

    # ------------------------------------------------------------------
    # Combine datasets
    # ------------------------------------------------------------------

    print("\nCleaning integrated dataset...")

    integrated_df = pd.concat(
        [
            synthetic_normalized,
            external_normalized,
        ],
        ignore_index=True,
    )

    before_duplicates = len(
        integrated_df
    )

    integrated_df = (
        integrated_df.drop_duplicates()
    )

    removed_integrated_duplicates = (
        before_duplicates
        - len(integrated_df)
    )

    print(
        "Removed duplicate rows:",
        removed_integrated_duplicates,
    )

    integrated_df = align_final_columns(
        integrated_df
    )

    # ------------------------------------------------------------------
    # Save integrated dataset
    # ------------------------------------------------------------------

    integrated_df.to_csv(
        INTEGRATED_OUTPUT_PATH,
        index=False,
    )

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------

    create_integration_report(
        synthetic_normalized,
        external_normalized,
        integrated_df,
    )

    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("INTEGRATION COMPLETED SUCCESSFULLY")
    print("=" * 100)

    print("\nSaved files:")

    print(
        "Synthetic normalized:",
        SYNTHETIC_OUTPUT_PATH,
    )

    print(
        "External normalized: ",
        EXTERNAL_OUTPUT_PATH,
    )

    print(
        "Integrated dataset:  ",
        INTEGRATED_OUTPUT_PATH,
    )

    print(
        "Integration report:  ",
        REPORT_OUTPUT_PATH,
    )

    print("\nDataset shapes:")

    print(
        "Synthetic: ",
        synthetic_normalized.shape,
    )

    print(
        "External:  ",
        external_normalized.shape,
    )

    print(
        "Integrated:",
        integrated_df.shape,
    )

    print("\nSource distribution:")

    print(
        integrated_df["source_dataset"]
        .value_counts()
    )

    print("\nTarget distribution:")

    print(
        integrated_df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    print("\nMissing values by column:")

    missing_values = (
        integrated_df.isna()
        .sum()
        .loc[
            lambda values: values > 0
        ]
    )

    if missing_values.empty:
        print("No missing values found.")

    else:
        print(missing_values)

    print("\nFinal columns:")

    for index, column in enumerate(
        integrated_df.columns,
        start=1,
    ):
        print(
            f"{index:02d}. {column}"
        )

    print("\n" + "=" * 100)


# ==========================================================================================
# 15. RUN
# ==========================================================================================

if __name__ == "__main__":
    run_integration()