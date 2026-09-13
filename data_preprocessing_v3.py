# ==========================================================================================
# Student Placement Prediction
# Data Preprocessing V3
# ==========================================================================================
#
# Purpose:
#   1. Load the integrated synthetic + external dataset
#   2. Select shared features
#   3. Remove leakage columns
#   4. Split into train and test sets
#   5. Preprocess numerical and categorical features
#   6. Apply SMOTE only to training data
#   7. Save all artifacts for model training and inference
#
# Run:
#   python data_preprocessing_v3.py
#
# ==========================================================================================

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")


# ==========================================================================================
# 1. PROJECT PATHS
# ==========================================================================================

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_DIR / "integrated_data_v2"
OUTPUT_DIR = PROJECT_DIR / "processed_data_v3"

INPUT_FILE = INPUT_DIR / "integrated_dataset_v2.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================================================
# 2. CONFIGURATION
# ==========================================================================================

TARGET_COLUMN = "placement_status"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# These are the features that can be shared between:
#   - Your synthetic dataset
#   - The external 10,000-row dataset
#
# We intentionally do not use:
#   - ssc_marks
#   - hsc_marks
#   - placement_training
#
# because those columns are missing from the synthetic dataset.

NUMERICAL_FEATURES = [
    "cgpa",
    "internships",
    "projects_count",
    "certifications",
    "aptitude_score",
    "communication_skills",
    "extracurricular_activities",
    "experience_score",
    "skill_score",
]

CATEGORICAL_FEATURES = []

SELECTED_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

LEAKAGE_COLUMNS = [
    "salary_package_lpa",
    "salary",
    "salary_package",
    "placement_probability",
    "placement_prediction",
]

NON_FEATURE_COLUMNS = [
    "source_dataset",
    "student_id",
    "studentid",
    "id",
    "slno",
]


# ==========================================================================================
# 3. HELPER FUNCTIONS
# ==========================================================================================

def print_section(title: str):
    """Print a formatted section heading."""
    print("\n" + "=" * 95)
    print(title)
    print("=" * 95)


def convert_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Convert selected columns to numeric values.

    Invalid values are converted to NaN and handled later
    by the training-only imputer.
    """

    df = df.copy()

    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def clean_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean placement_status into binary integer values.

    Supported values include:
        Placed, NotPlaced, Yes, No, True, False, 1, 0
    """

    df = df.copy()

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' was not found in the dataset."
        )

    def convert_target(value):
        if pd.isna(value):
            return np.nan

        value = str(value).strip().lower()

        if value in {
            "placed",
            "yes",
            "y",
            "true",
            "1",
            "selected",
            "successful",
        }:
            return 1

        if value in {
            "notplaced",
            "not placed",
            "not_placed",
            "no",
            "n",
            "false",
            "0",
            "not selected",
            "unsuccessful",
        }:
            return 0

        try:
            numeric_value = float(value)

            if numeric_value == 1:
                return 1

            if numeric_value == 0:
                return 0

        except ValueError:
            pass

        return np.nan

    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(convert_target)

    missing_targets = df[TARGET_COLUMN].isna().sum()

    if missing_targets > 0:
        print(
            f"Removing {missing_targets:,} rows with invalid or missing target values."
        )
        df = df.dropna(subset=[TARGET_COLUMN])

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df


def add_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add missing shared feature columns as NaN.

    This makes the script robust if one of the datasets does not
    contain a particular feature.
    """

    df = df.copy()

    for column in SELECTED_FEATURES:
        if column not in df.columns:
            print(
                f"WARNING: Missing feature '{column}'. "
                f"Creating it with NaN values."
            )
            df[column] = np.nan

    return df


def calculate_shared_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate shared derived features.

    experience_score:
        internships + projects_count + certifications

    skill_score:
        aptitude_score + communication_skills +
        extracurricular_activities

    Important:
        Missing values are not filled here.
        Imputation is fitted later using training data only.
    """

    df = df.copy()

    experience_columns = [
        "internships",
        "projects_count",
        "certifications",
    ]

    skill_columns = [
        "aptitude_score",
        "communication_skills",
        "extracurricular_activities",
    ]

    for column in experience_columns + skill_columns:
        if column not in df.columns:
            df[column] = np.nan

        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Recalculate experience score from shared features.
    df["experience_score"] = df[experience_columns].sum(
        axis=1,
        min_count=1,
    )

    # Recalculate skill score from shared features.
    df["skill_score"] = df[skill_columns].mean(
        axis=1,
        skipna=True,
    )

    return df


def remove_leakage_and_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove target leakage and metadata columns.

    The target is kept separately.
    """

    df = df.copy()

    columns_to_remove = []

    for column in LEAKAGE_COLUMNS + NON_FEATURE_COLUMNS:
        if column in df.columns:
            columns_to_remove.append(column)

    if columns_to_remove:
        print("Removing columns:")
        for column in columns_to_remove:
            print(f"  - {column}")

        df = df.drop(columns=columns_to_remove, errors="ignore")

    return df


def save_json(data: dict, path: Path):
    """Save a dictionary as formatted JSON."""

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)


# ==========================================================================================
# 4. LOAD DATASET
# ==========================================================================================

def load_dataset() -> pd.DataFrame:
    """Load the integrated dataset."""

    print_section("LOADING INTEGRATED DATASET")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Integrated dataset was not found:\n{INPUT_FILE}\n\n"
            "Run data_integration_v2.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Input file: {INPUT_FILE}")
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {list(df.columns)}")

    return df


# ==========================================================================================
# 5. PREPARE DATA
# ==========================================================================================

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the integrated dataset before splitting."""

    print_section("PREPARING DATASET")

    df = df.copy()

    # Remove duplicate rows.
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        print(f"Removing duplicate rows: {duplicate_count:,}")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("No duplicate rows found.")

    # Add any missing shared columns.
    df = add_missing_features(df)

    # Convert numerical columns.
    df = convert_numeric_columns(
        df,
        NUMERICAL_FEATURES,
    )

    # Recalculate shared derived features.
    df = calculate_shared_features(df)

    # Clean target.
    df = clean_target_column(df)

    # Remove leakage and metadata columns.
    df = remove_leakage_and_metadata(df)

    # Verify required columns.
    missing_required = [
        column
        for column in SELECTED_FEATURES + [TARGET_COLUMN]
        if column not in df.columns
    ]

    if missing_required:
        raise ValueError(
            "The following required columns are missing:\n"
            + "\n".join(missing_required)
        )

    # Keep only selected features and target.
    final_columns = SELECTED_FEATURES + [TARGET_COLUMN]

    df = df[final_columns].copy()

    print(f"Prepared dataset shape: {df.shape}")

    print("\nTarget distribution:")
    print(df[TARGET_COLUMN].value_counts().sort_index())

    print("\nMissing values:")
    missing_values = df.isna().sum()
    missing_values = missing_values[missing_values > 0]

    if missing_values.empty:
        print("No missing values found.")
    else:
        print(missing_values)

    return df


# ==========================================================================================
# 6. BUILD PREPROCESSOR
# ==========================================================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing transformer.

    Numerical:
        Median imputation
        StandardScaler

    Categorical:
        Most frequent imputation
        One-hot encoding
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    transformers = [
        (
            "numerical",
            numerical_pipeline,
            NUMERICAL_FEATURES,
        )
    ]

    if CATEGORICAL_FEATURES:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor


# ==========================================================================================
# 7. PREPROCESS AND APPLY SMOTE
# ==========================================================================================

def preprocess_and_balance(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """
    Fit preprocessing only on training data.

    SMOTE is applied only to the training data.

    The test set remains untouched so that evaluation represents
    real-world class distribution.
    """

    print_section("PREPROCESSING DATA")

    preprocessor = build_preprocessor()

    print("Fitting preprocessor on training data only...")

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
    X_test_processed = np.asarray(X_test_processed, dtype=np.float32)

    y_train_array = np.asarray(y_train, dtype=np.int32)
    y_test_array = np.asarray(y_test, dtype=np.int32)

    print(f"Processed training shape: {X_train_processed.shape}")
    print(f"Processed testing shape: {X_test_processed.shape}")

    print("\nTraining target distribution before SMOTE:")
    print(pd.Series(y_train_array).value_counts().sort_index())

    print("\nTesting target distribution:")
    print(pd.Series(y_test_array).value_counts().sort_index())

    # Apply SMOTE only to the training data.
    print("\nApplying SMOTE to training data only...")

    smote = SMOTE(
        random_state=RANDOM_STATE,
        sampling_strategy="auto",
    )

    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train_processed,
        y_train_array,
    )

    X_train_balanced = np.asarray(
        X_train_balanced,
        dtype=np.float32,
    )

    y_train_balanced = np.asarray(
        y_train_balanced,
        dtype=np.int32,
    )

    print(
        f"Balanced training shape: {X_train_balanced.shape}"
    )

    print("\nTraining target distribution after SMOTE:")
    print(pd.Series(y_train_balanced).value_counts().sort_index())

    return (
        preprocessor,
        smote,
        X_train_processed,
        X_test_processed,
        X_train_balanced,
        y_train_balanced,
        y_test_array,
    )


# ==========================================================================================
# 8. SAVE PROCESSED DATA
# ==========================================================================================

def save_processed_data(
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor,
    smote,
    X_train_processed: np.ndarray,
    X_test_processed: np.ndarray,
    X_train_balanced: np.ndarray,
    y_train_balanced: np.ndarray,
    y_test_array: np.ndarray,
):
    """Save all processed data and preprocessing artifacts."""

    print_section("SAVING PROCESSED DATA")

    # Save original prepared dataset.
    df.to_csv(
        OUTPUT_DIR / "prepared_dataset_v3.csv",
        index=False,
    )

    # Save raw train/test splits.
    X_train.to_csv(
        OUTPUT_DIR / "X_train_raw_v3.csv",
        index=False,
    )

    X_test.to_csv(
        OUTPUT_DIR / "X_test_raw_v3.csv",
        index=False,
    )

    pd.DataFrame(
        {TARGET_COLUMN: y_train}
    ).to_csv(
        OUTPUT_DIR / "y_train_raw_v3.csv",
        index=False,
    )

    pd.DataFrame(
        {TARGET_COLUMN: y_test}
    ).to_csv(
        OUTPUT_DIR / "y_test_raw_v3.csv",
        index=False,
    )

    # Save processed arrays.
    np.save(
        OUTPUT_DIR / "X_train_processed_v3.npy",
        X_train_processed,
    )

    np.save(
        OUTPUT_DIR / "X_test_processed_v3.npy",
        X_test_processed,
    )

    # Save SMOTE-balanced training arrays.
    np.save(
        OUTPUT_DIR / "X_train_balanced_v3.npy",
        X_train_balanced,
    )

    np.save(
        OUTPUT_DIR / "y_train_balanced_v3.npy",
        y_train_balanced,
    )

    np.save(
        OUTPUT_DIR / "y_test_v3.npy",
        y_test_array,
    )

    # Save preprocessing artifacts.
    joblib.dump(
        preprocessor,
        OUTPUT_DIR / "preprocessor_v3.joblib",
    )

    joblib.dump(
        smote,
        OUTPUT_DIR / "smote_v3.joblib",
    )

    # Extract processed feature names.
    feature_names = []

    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "remainder":
            continue

        if hasattr(transformer, "get_feature_names_out"):
            names = transformer.get_feature_names_out(columns)
            feature_names.extend(names.tolist())
        else:
            feature_names.extend(columns)

    feature_names = [
        str(name).replace("numerical__", "")
        .replace("categorical__", "")
        for name in feature_names
    ]

    pd.DataFrame(
        {
            "feature_index": range(len(feature_names)),
            "feature_name": feature_names,
        }
    ).to_csv(
        OUTPUT_DIR / "processed_feature_names_v3.csv",
        index=False,
    )

    # Save metadata.
    metadata = {
        "project": "Student Placement Prediction",
        "version": "V3",
        "input_file": str(INPUT_FILE),
        "target_column": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "selected_features": SELECTED_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "leakage_columns_removed": LEAKAGE_COLUMNS,
        "non_feature_columns_removed": NON_FEATURE_COLUMNS,
        "original_dataset_shape": list(df.shape),
        "raw_train_shape": list(X_train.shape),
        "raw_test_shape": list(X_test.shape),
        "processed_train_shape": list(X_train_processed.shape),
        "processed_test_shape": list(X_test_processed.shape),
        "balanced_train_shape": list(X_train_balanced.shape),
        "train_target_distribution_before_smote": {
            str(key): int(value)
            for key, value in y_train.value_counts().to_dict().items()
        },
        "train_target_distribution_after_smote": {
            str(key): int(value)
            for key, value in pd.Series(
                y_train_balanced
            ).value_counts().to_dict().items()
        },
        "test_target_distribution": {
            str(key): int(value)
            for key, value in pd.Series(
                y_test_array
            ).value_counts().to_dict().items()
        },
        "processed_feature_count": len(feature_names),
    }

    save_json(
        metadata,
        OUTPUT_DIR / "preprocessing_metadata_v3.json",
    )

    print("Saved files:")

    for file in sorted(OUTPUT_DIR.iterdir()):
        if file.is_file():
            print(f"  - {file.name}")


# ==========================================================================================
# 9. SAVE EXTERNAL VALIDATION DATA
# ==========================================================================================

def save_external_validation_data(df: pd.DataFrame):
    """
    Save external rows separately when source_dataset is available.

    This is useful for checking whether the model generalizes
    to data from a different source.
    """

    print_section("SAVING EXTERNAL VALIDATION DATA")

    original_file = INPUT_FILE

    try:
        original_df = pd.read_csv(original_file)
    except Exception:
        print("Could not reload original integrated dataset.")
        return

    if "source_dataset" not in original_df.columns:
        print(
            "source_dataset column not found. "
            "Skipping external validation split."
        )
        return

    original_df = add_missing_features(original_df)
    original_df = convert_numeric_columns(
        original_df,
        NUMERICAL_FEATURES,
    )
    original_df = calculate_shared_features(original_df)
    original_df = clean_target_column(original_df)

    external_mask = (
        original_df["source_dataset"]
        .astype(str)
        .str.lower()
        .str.contains("external")
    )

    external_df = original_df.loc[external_mask].copy()

    if external_df.empty:
        print("No external rows found.")
        return

    external_df = external_df[
        SELECTED_FEATURES + [TARGET_COLUMN]
    ]

    external_df.to_csv(
        OUTPUT_DIR / "external_validation_dataset_v3.csv",
        index=False,
    )

    print(
        f"External validation rows saved: {len(external_df):,}"
    )


# ==========================================================================================
# 10. MAIN PIPELINE
# ==========================================================================================

def run_preprocessing():
    """Run the complete V3 preprocessing pipeline."""

    print_section("STUDENT PLACEMENT PREDICTION - PREPROCESSING V3")

    # Load.
    df = load_dataset()

    # Prepare.
    df = prepare_dataset(df)

    # Separate features and target.
    X = df[SELECTED_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    print_section("TRAIN / TEST SPLIT")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples: {len(X_test):,}")

    print("\nTraining target distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting target distribution:")
    print(y_test.value_counts().sort_index())

    # Preprocess and balance.
    (
        preprocessor,
        smote,
        X_train_processed,
        X_test_processed,
        X_train_balanced,
        y_train_balanced,
        y_test_array,
    ) = preprocess_and_balance(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    # Save artifacts.
    save_processed_data(
        df=df,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        smote=smote,
        X_train_processed=X_train_processed,
        X_test_processed=X_test_processed,
        X_train_balanced=X_train_balanced,
        y_train_balanced=y_train_balanced,
        y_test_array=y_test_array,
    )

    # Save external validation data.
    save_external_validation_data(df)

    print_section("PREPROCESSING V3 COMPLETED SUCCESSFULLY")

    print(f"Output directory:\n{OUTPUT_DIR}")

    print("\nNext step:")
    print("Run model_training_v3.py")


# ==========================================================================================
# 11. SCRIPT ENTRY POINT
# ==========================================================================================

if __name__ == "__main__":
    run_preprocessing()