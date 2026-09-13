# ==========================================================================================
# STUDENT PLACEMENT INTELLIGENCE — ADVANCED TECH MODEL & SALARY REGRESSION (V4)
# ==========================================================================================
# Purpose:
#   1. Train a dedicated technical-hiring placement classifier on the full synthetic dataset
#      incorporating coding_skills, dsa_score, college_tier, branch, backlogs, etc.
#   2. Train a multi-stage salary regressor predicting expected salary package (LPA) for
#      placed candidates.
#   3. Save serialized pipelines for instant inference and TreeSHAP explainability in FastAPI.
# ==========================================================================================

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
from ml_pipeline.features import (
    ALL_FEATURES,
    ALL_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    engineer_features as canonical_engineer_features,
)
from ml_pipeline.train_pipeline import write_manifest

warnings.filterwarnings("ignore")

# ==========================================================================================
# 1. DIRECTORIES & PATHS
# ==========================================================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "dataset" / "student_placement_synthetic.csv"

TRAINED_MODELS_V4 = PROJECT_DIR / "trained_models_v4"
MODEL_RESULTS_V4 = PROJECT_DIR / "model_results_v4"

TRAINED_MODELS_V4.mkdir(parents=True, exist_ok=True)
MODEL_RESULTS_V4.mkdir(parents=True, exist_ok=True)

# ==========================================================================================
# 2. FEATURE DEFINITIONS
# ==========================================================================================

TARGET_CLASSIFICATION = "placement_status"
TARGET_REGRESSION = "salary_package_lpa"

# ==========================================================================================
# 3. FEATURE ENGINEERING FUNCTION
# ==========================================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the canonical V4 feature set."""
    return canonical_engineer_features(df, fill_defaults=True)


# ==========================================================================================
# 4. DATA LOADING & PREPARATION
# ==========================================================================================

def load_and_prepare_data():
    print(f"Loading synthetic dataset from:\n  {DATASET_PATH}")
    raw = pd.read_csv(DATASET_PATH)
    print(f"Raw shape: {raw.shape}")

    features = engineer_features(raw)
    features[TARGET_CLASSIFICATION] = pd.to_numeric(
        raw[TARGET_CLASSIFICATION], errors="raise"
    ).astype(int)
    features[TARGET_REGRESSION] = pd.to_numeric(
        raw[TARGET_REGRESSION], errors="coerce"
    )
    return features


# ==========================================================================================
# 5. BUILD PREPROCESSOR
# ==========================================================================================

def create_preprocessor():
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, ALL_NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


# ==========================================================================================
# 6. TRAIN CLASSIFIER & REGRESSOR
# ==========================================================================================

def train_models():
    df = load_and_prepare_data()

    X = df[ALL_FEATURES].copy()
    y_class = df[TARGET_CLASSIFICATION].astype(int)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.20, random_state=42, stratify=y_class
    )

    print("\nTraining preprocessor on X_train...")
    preprocessor = create_preprocessor()
    preprocessor.fit(X_train)

    # Get feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    processed_feature_names = ALL_NUMERICAL_FEATURES + cat_feature_names

    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # --------------------------------------------------------------------------------------
    # STAGE 1: PLACEMENT CLASSIFIER (XGBOOST)
    # --------------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 1: TRAINING PLACEMENT CLASSIFIER (XGBOOST)")
    print("=" * 80)

    clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )

    clf.fit(X_train_proc, y_train)

    y_pred = clf.predict(X_test_proc)
    y_prob = clf.predict_proba(X_test_proc)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"Classification Test Accuracy: {acc * 100:.2f}%")
    print(f"ROC-AUC:                     {roc_auc:.4f}")
    print(f"F1-Score:                    {f1:.4f}")
    print(f"Precision:                   {prec:.4f}")
    print(f"Recall:                      {rec:.4f}")
    print(f"PR-AUC:                      {pr_auc:.4f}")

    clf_metrics = {
        "model_name": "XGBoost Technical Classifier",
        "version": "V4",
        "feature_count": len(processed_feature_names),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
    }

    # Feature Importance
    fi = pd.DataFrame({
        "feature": processed_feature_names,
        "importance": clf.feature_importances_,
    }).sort_values(by="importance", ascending=False)

    fi.to_csv(MODEL_RESULTS_V4 / "feature_importance.csv", index=False)

    # --------------------------------------------------------------------------------------
    # STAGE 2: SALARY REGRESSOR (FOR PLACED CANDIDATES)
    # --------------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 2: TRAINING SALARY REGRESSOR (XGBOOST)")
    print("=" * 80)

    # Filter to placed students with valid salary
    df_placed = df[df[TARGET_CLASSIFICATION] == 1].dropna(subset=[TARGET_REGRESSION])
    X_reg = df_placed[ALL_FEATURES].copy()
    y_reg = df_placed[TARGET_REGRESSION].astype(float)

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.20, random_state=42
    )

    X_train_reg_proc = preprocessor.transform(X_train_reg)
    X_test_reg_proc = preprocessor.transform(X_test_reg)

    reg = xgb.XGBRegressor(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train_reg_proc, y_train_reg)

    y_reg_pred = reg.predict(X_test_reg_proc)
    r2 = float(r2_score(y_test_reg, y_reg_pred))
    mae = float(mean_absolute_error(y_test_reg, y_reg_pred))
    rmse = float(root_mean_squared_error(y_test_reg, y_reg_pred))

    print(f"Salary Regression R2 Score:   {r2:.4f}")
    print(f"Salary Mean Absolute Error:   {mae:.2f} LPA")
    print(f"Salary Root Mean Sq Error:    {rmse:.2f} LPA")
    print(f"Average Actual Salary:        {y_test_reg.mean():.2f} LPA")

    reg_metrics = {
        "model_name": "XGBoost Salary Regressor",
        "version": "V4",
        "target": "salary_package_lpa",
        "r2_score": r2,
        "mae_lpa": mae,
        "rmse_lpa": rmse,
        "mean_actual_lpa": float(y_test_reg.mean()),
    }

    # --------------------------------------------------------------------------------------
    # SERIALIZE ARTIFACTS
    # --------------------------------------------------------------------------------------
    print("\nSaving trained models and preprocessor artifacts...")

    joblib.dump(preprocessor, TRAINED_MODELS_V4 / "tech_preprocessor_v4.joblib")
    joblib.dump(clf, TRAINED_MODELS_V4 / "best_tech_model_xgboost_v4.joblib")
    joblib.dump(reg, TRAINED_MODELS_V4 / "salary_regressor_v4.joblib")

    with open(TRAINED_MODELS_V4 / "feature_names_v4.json", "w") as f:
        json.dump(processed_feature_names, f, indent=2)

    with open(TRAINED_MODELS_V4 / "tech_model_metrics_v4.json", "w") as f:
        json.dump({
            "classification": clf_metrics,
            "regression": reg_metrics,
        }, f, indent=2)

    with open(MODEL_RESULTS_V4 / "classification_metrics.json", "w") as f:
        json.dump(clf_metrics, f, indent=2)

    with open(MODEL_RESULTS_V4 / "salary_regression_metrics.json", "w") as f:
        json.dump(reg_metrics, f, indent=2)

    write_manifest(df)

    print("All V4 artifacts saved successfully to:")
    print(f"  {TRAINED_MODELS_V4}")
    print(f"  {MODEL_RESULTS_V4}")


if __name__ == "__main__":
    train_models()
