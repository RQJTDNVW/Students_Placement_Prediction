# =============================================================================
# STUDENT PLACEMENT PREDICTION
# COMPLETE DATA PREPROCESSING PIPELINE
# =============================================================================

from pathlib import Path
import warnings
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_DIR = Path(r"C:\AI\Student Placement Prediction")

DATASET_PATH = PROJECT_DIR / "dataset" / "student_placement_synthetic.csv"

OUTPUT_DIR = PROJECT_DIR / "processed_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# =============================================================================
# 2. LOAD DATA
# =============================================================================

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print("\n" + "=" * 80)
    print("DATASET LOADED")
    print("=" * 80)

    print(f"Dataset shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    return df


# =============================================================================
# 3. BASIC DATA CLEANING
# =============================================================================

def clean_dataset(df):

    df = df.copy()

    print("\n" + "=" * 80)
    print("BASIC DATA CLEANING")
    print("=" * 80)

    # Remove duplicate rows
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    print(f"Duplicates removed: {duplicate_count}")

    # Remove rows where target is missing
    target_column = "placement_status"

    missing_target = df[target_column].isna().sum()

    if missing_target > 0:
        df = df.dropna(subset=[target_column]).reset_index(drop=True)

    print(f"Rows with missing target removed: {missing_target}")

    # Convert target to integer
    df[target_column] = pd.to_numeric(
        df[target_column],
        errors="coerce"
    )

    # Remove invalid target rows
    df = df[df[target_column].isin([0, 1])].copy()

    df[target_column] = df[target_column].astype(int)

    # Remove salary because it is only known after placement.
    # Keeping it would cause target leakage.
    salary_column = "salary_package_lpa"

    if salary_column in df.columns:
        df = df.drop(columns=[salary_column])

    print(f"Final cleaned shape: {df.shape}")
    print("\nTarget distribution:")
    print(df[target_column].value_counts())

    return df


# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

def create_features(df):

    df = df.copy()

    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Academic score
    # -------------------------------------------------------------------------

    academic_columns = [
        "cgpa",
        "dsa_score",
        "aptitude_score",
    ]

    available_academic_columns = [
        column
        for column in academic_columns
        if column in df.columns
    ]

    if available_academic_columns:

        df["academic_score"] = df[
            available_academic_columns
        ].mean(axis=1)

    # -------------------------------------------------------------------------
    # Technical skill score
    # -------------------------------------------------------------------------

    skill_columns = [
        "coding_skills",
        "dsa_score",
        "ml_knowledge",
        "system_design",
    ]

    available_skill_columns = [
        column
        for column in skill_columns
        if column in df.columns
    ]

    if available_skill_columns:

        df["skill_score"] = df[
            available_skill_columns
        ].mean(axis=1)

    # -------------------------------------------------------------------------
    # Experience score
    # -------------------------------------------------------------------------

    experience_columns = [
        "internships",
        "projects_count",
        "certifications",
        "hackathons",
        "open_source_contributions",
    ]

    available_experience_columns = [
        column
        for column in experience_columns
        if column in df.columns
    ]

    if available_experience_columns:

        df["experience_score"] = df[
            available_experience_columns
        ].sum(axis=1)

    # -------------------------------------------------------------------------
    # Backlog category
    # -------------------------------------------------------------------------

    if "backlogs" in df.columns:

        df["backlog_category"] = pd.cut(
            df["backlogs"],
            bins=[-np.inf, 0, 2, np.inf],
            labels=[
                "No Backlogs",
                "Low Backlogs",
                "High Backlogs",
            ]
        )

    # -------------------------------------------------------------------------
    # Employability score
    # -------------------------------------------------------------------------

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
        "extracurriculars",
    ]

    available_employability_columns = [
        column
        for column in employability_columns
        if column in df.columns
    ]

    if available_employability_columns:

        df["employability_score"] = df[
            available_employability_columns
        ].mean(axis=1)

    print(f"Feature-engineered shape: {df.shape}")

    return df


# =============================================================================
# 5. TRAIN/TEST SPLIT
# =============================================================================

def split_data(df):

    print("\n" + "=" * 80)
    print("TRAIN/TEST SPLIT")
    print("=" * 80)

    target_column = "placement_status"

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    print("\nOriginal training distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting distribution:")
    print(y_test.value_counts().sort_index())

    return X_train, X_test, y_train, y_test


# =============================================================================
# 6. PREPROCESS NUMERIC AND CATEGORICAL FEATURES
# =============================================================================

def preprocess_features(X_train, X_test):

    print("\n" + "=" * 80)
    print("FEATURE PREPROCESSING")
    print("=" * 80)

    numeric_columns = X_train.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print(f"Numeric columns: {len(numeric_columns)}")
    print(f"Categorical columns: {len(categorical_columns)}")

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            ),
        ],
        remainder="drop",
    )

    # Fit only on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data using the training-fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)

    X_train_processed = np.asarray(
        X_train_processed,
        dtype=np.float32
    )

    X_test_processed = np.asarray(
        X_test_processed,
        dtype=np.float32
    )

    # Get real feature names
    feature_names = preprocessor.get_feature_names_out()

    feature_names = [
        str(name).replace("numeric__", "")
        .replace("categorical__", "")
        for name in feature_names
    ]

    print(f"Processed training shape: {X_train_processed.shape}")
    print(f"Processed testing shape: {X_test_processed.shape}")

    return (
        X_train_processed,
        X_test_processed,
        preprocessor,
        feature_names,
    )


# =============================================================================
# 7. FEATURE SELECTION USING RFE
# =============================================================================

def select_features(
    X_train,
    X_test,
    y_train,
    feature_names,
):

    print("\n" + "=" * 80)
    print("FEATURE SELECTION USING RFE")
    print("=" * 80)

    total_features = X_train.shape[1]

    # Keep approximately 50% of the features
    selected_feature_count = max(
        1,
        total_features // 2
    )

    estimator = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="liblinear",
    )

    selector = RFE(
        estimator=estimator,
        n_features_to_select=selected_feature_count,
        step=1,
    )

    X_train_selected = selector.fit_transform(
        X_train,
        y_train
    )

    X_test_selected = selector.transform(X_test)

    selected_feature_names = [
        feature_names[index]
        for index, selected in enumerate(selector.support_)
        if selected
    ]

    print(f"Original feature count: {total_features}")
    print(f"Selected feature count: {len(selected_feature_names)}")

    print("\nSelected features:")
    for feature in selected_feature_names:
        print(f" - {feature}")

    return (
        X_train_selected,
        X_test_selected,
        selector,
        selected_feature_names,
    )


# =============================================================================
# 8. APPLY SMOTE CORRECTLY
# =============================================================================

def apply_smote(X_train, y_train):

    print("\n" + "=" * 80)
    print("SMOTE OVERSAMPLING")
    print("=" * 80)

    print("Class distribution before SMOTE:")
    print(pd.Series(y_train).value_counts().sort_index())

    smote = SMOTE(
        random_state=RANDOM_STATE,
        sampling_strategy="auto",
    )

    # IMPORTANT:
    # The returned arrays must be used for saving and training.
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_train_resampled).value_counts().sort_index())

    print(f"\nTraining shape after SMOTE: {X_train_resampled.shape}")

    return X_train_resampled, y_train_resampled, smote


# =============================================================================
# 9. SAVE PROCESSED DATA
# =============================================================================

def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    selector,
    smote,
    selected_feature_names,
):

    print("\n" + "=" * 80)
    print("SAVING PROCESSED DATA")
    print("=" * 80)

    # Save arrays
    np.save(
        OUTPUT_DIR / "X_train.npy",
        X_train
    )

    np.save(
        OUTPUT_DIR / "X_test.npy",
        X_test
    )

    np.save(
        OUTPUT_DIR / "y_train.npy",
        np.asarray(y_train)
    )

    np.save(
        OUTPUT_DIR / "y_test.npy",
        np.asarray(y_test)
    )

    # Save preprocessing objects
    joblib.dump(
        preprocessor,
        OUTPUT_DIR / "preprocessor.joblib"
    )

    joblib.dump(
        selector,
        OUTPUT_DIR / "feature_selector.joblib"
    )

    joblib.dump(
        smote,
        OUTPUT_DIR / "smote.joblib"
    )

    # Save selected feature names
    feature_names_df = pd.DataFrame({
        "feature_index": range(len(selected_feature_names)),
        "feature_name": selected_feature_names,
    })

    feature_names_df.to_csv(
        OUTPUT_DIR / "selected_feature_names.csv",
        index=False
    )

    # Save metadata
    metadata = {
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "smote_applied": True,
        "feature_selection": "RFE",
        "selected_feature_count": len(selected_feature_names),
        "train_rows_after_smote": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "feature_count": int(X_train.shape[1]),
    }

    with open(
        OUTPUT_DIR / "preprocessing_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("Saved files:")

    saved_files = [
        "X_train.npy",
        "X_test.npy",
        "y_train.npy",
        "y_test.npy",
        "preprocessor.joblib",
        "feature_selector.joblib",
        "smote.joblib",
        "selected_feature_names.csv",
        "preprocessing_metadata.json",
    ]

    for filename in saved_files:
        print(f" - {filename}")


# =============================================================================
# 10. VALIDATE FINAL DATA
# =============================================================================

def validate_processed_data(
    X_train,
    X_test,
    y_train,
    y_test,
    selected_feature_names,
):

    print("\n" + "=" * 80)
    print("FINAL VALIDATION")
    print("=" * 80)

    assert X_train.shape[1] == X_test.shape[1]

    assert X_train.shape[1] == len(
        selected_feature_names
    )

    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()

    assert not np.isinf(X_train).any()
    assert not np.isinf(X_test).any()

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    train_distribution = pd.Series(
        y_train
    ).value_counts().sort_index()

    print(f"Final X_train shape: {X_train.shape}")
    print(f"Final X_test shape: {X_test.shape}")

    print("\nFinal training class distribution:")
    print(train_distribution)

    print("\nFinal testing class distribution:")
    print(pd.Series(y_test).value_counts().sort_index())

    print("\nValidation completed successfully.")


# =============================================================================
# 11. MAIN PIPELINE
# =============================================================================

def run_preprocessing_pipeline():

    print("\n")
    print("=" * 80)
    print("STUDENT PLACEMENT PREDICTION")
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 80)

    # Step 1: Load
    df = load_dataset()

    # Step 2: Clean
    df = clean_dataset(df)

    # Step 3: Feature engineering
    df = create_features(df)

    # Step 4: Split
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 5: Preprocess
    (
        X_train_processed,
        X_test_processed,
        preprocessor,
        feature_names,
    ) = preprocess_features(
        X_train,
        X_test
    )

    # Step 6: RFE
    (
        X_train_selected,
        X_test_selected,
        selector,
        selected_feature_names,
    ) = select_features(
        X_train_processed,
        X_test_processed,
        y_train,
        feature_names,
    )

    # Step 7: SMOTE
    (
        X_train_smote,
        y_train_smote,
        smote,
    ) = apply_smote(
        X_train_selected,
        y_train
    )

    # Step 8: Validate
    validate_processed_data(
        X_train_smote,
        X_test_selected,
        y_train_smote,
        y_test,
        selected_feature_names,
    )

    # Step 9: Save
    save_processed_data(
        X_train_smote,
        X_test_selected,
        y_train_smote,
        y_test,
        preprocessor,
        selector,
        smote,
        selected_feature_names,
    )

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print("\nImportant expected output:")
    print("The training classes should be balanced after SMOTE.")
    print("For your dataset, it should be approximately:")
    print("Class 0: 54,780")
    print("Class 1: 54,780")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    run_preprocessing_pipeline()