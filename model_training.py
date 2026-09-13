# =============================================================================
# Student Placement Prediction
# Complete Model Training and Evaluation
# =============================================================================

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = PROJECT_DIR / "processed_data"
MODELS_DIR = PROJECT_DIR / "trained_models"
RESULTS_DIR = PROJECT_DIR / "model_results"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# 2. LOAD PROCESSED DATA
# =============================================================================

def load_processed_data():
    print("\n" + "=" * 80)
    print("LOADING PROCESSED DATA")
    print("=" * 80)

    required_files = [
        "X_train.npy",
        "X_test.npy",
        "y_train.npy",
        "y_test.npy",
    ]

    for filename in required_files:
        file_path = PROCESSED_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"\nMissing file: {file_path}\n"
                "Please run data_preprocessing.py first."
            )

    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")

    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape : {y_test.shape}")

    print("\nTraining class distribution:")
    print(pd.Series(y_train).value_counts().sort_index())

    print("\nTesting class distribution:")
    print(pd.Series(y_test).value_counts().sort_index())

    return X_train, X_test, y_train, y_test


# =============================================================================
# 3. DEFINE MODELS
# =============================================================================

def create_models():
    """
    Create all classification models.

    LinearSVC is used instead of the normal RBF SVC because the dataset
    contains approximately 100,000 samples. LinearSVC is much faster
    and is suitable for large datasets.
    """

    models = {
        # ---------------------------------------------------------------------
        # Logistic Regression
        # ---------------------------------------------------------------------
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
            n_jobs=-1,
        ),

        # ---------------------------------------------------------------------
        # K-Nearest Neighbors
        # ---------------------------------------------------------------------
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=-1,
        ),

        # ---------------------------------------------------------------------
        # Naive Bayes
        # ---------------------------------------------------------------------
        "Naive Bayes": GaussianNB(),

        # ---------------------------------------------------------------------
        # Decision Tree
        # ---------------------------------------------------------------------
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
        ),

        # ---------------------------------------------------------------------
        # Random Forest
        # ---------------------------------------------------------------------
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),

        # ---------------------------------------------------------------------
        # Fast SVM
        #
        # LinearSVC is significantly faster than RBF SVC for 100,000 samples.
        # It does not calculate probabilities, so decision_function() is
        # used later for ROC-AUC.
        # ---------------------------------------------------------------------
        "SVM": LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=42,
        ),

        # ---------------------------------------------------------------------
        # XGBoost
        # ---------------------------------------------------------------------
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    return models


# =============================================================================
# 4. GET MODEL SCORES
# =============================================================================

def get_prediction_scores(model, X):
    """
    Return probability-like prediction scores for ROC-AUC.

    Models such as Logistic Regression, KNN, Random Forest, and XGBoost
    provide predict_proba().

    LinearSVC does not provide predict_proba(), so decision_function()
    is used instead.
    """

    # Models that support probability prediction
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

        # Binary classification:
        # return probability of class 1
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            return probabilities[:, 1]

        return probabilities

    # LinearSVC uses decision_function()
    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    return None


# =============================================================================
# 5. EVALUATE MODEL
# =============================================================================

def evaluate_model(model_name, model, X_test, y_test):
    print(f"\nEvaluating {model_name}...")

    predictions = model.predict(X_test)
    scores = get_prediction_scores(model, X_test)

    # -------------------------------------------------------------------------
    # Classification metrics
    # -------------------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    # -------------------------------------------------------------------------
    # ROC-AUC
    # -------------------------------------------------------------------------

    try:
        roc_auc = roc_auc_score(
            y_test,
            scores,
        )
    except Exception:
        roc_auc = None

    # -------------------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    # -------------------------------------------------------------------------
    # Classification report
    # -------------------------------------------------------------------------

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(report)

    # -------------------------------------------------------------------------
    # Return results
    # -------------------------------------------------------------------------

    results = {
        "model": model_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    return results


# =============================================================================
# 6. TRAIN ALL MODELS
# =============================================================================

def train_models(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 80)
    print("MODEL TRAINING STARTED")
    print("=" * 80)

    models = create_models()

    print(f"\nTotal models created: {len(models)}")

    all_results = []

    for model_name, model in models.items():

        print("\n" + "-" * 80)
        print(f"TRAINING: {model_name}")
        print("-" * 80)

        try:
            # -----------------------------------------------------------------
            # Train model
            # -----------------------------------------------------------------

            model.fit(
                X_train,
                y_train,
            )

            print(f"{model_name} training completed.")

            # -----------------------------------------------------------------
            # Evaluate model
            # -----------------------------------------------------------------

            results = evaluate_model(
                model_name=model_name,
                model=model,
                X_test=X_test,
                y_test=y_test,
            )

            all_results.append(results)

            # -----------------------------------------------------------------
            # Save trained model
            # -----------------------------------------------------------------

            safe_model_name = (
                model_name.lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            model_path = MODELS_DIR / f"{safe_model_name}.joblib"

            joblib.dump(
                model,
                model_path,
            )

            print(f"Model saved to: {model_path}")

        except Exception as error:
            print(f"\nERROR while training {model_name}:")
            print(error)

    return all_results


# =============================================================================
# 7. SAVE RESULTS
# =============================================================================

def save_results(all_results):
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    if not all_results:
        print("No results were generated.")
        return

    results_dataframe = pd.DataFrame(all_results)

    # -------------------------------------------------------------------------
    # Save model comparison CSV
    # -------------------------------------------------------------------------

    comparison_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]

    comparison_dataframe = results_dataframe[
        comparison_columns
    ].copy()

    comparison_dataframe = comparison_dataframe.sort_values(
        by="f1_score",
        ascending=False,
    )

    comparison_path = RESULTS_DIR / "model_comparison.csv"

    comparison_dataframe.to_csv(
        comparison_path,
        index=False,
    )

    print(f"Model comparison saved to: {comparison_path}")

    # -------------------------------------------------------------------------
    # Save complete results as JSON
    # -------------------------------------------------------------------------

    json_path = RESULTS_DIR / "complete_model_results.json"

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            all_results,
            file,
            indent=4,
        )

    print(f"Complete results saved to: {json_path}")

    # -------------------------------------------------------------------------
    # Print model comparison
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print(
        comparison_dataframe.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # -------------------------------------------------------------------------
    # Select best model based on F1 score
    # -------------------------------------------------------------------------

    best_model_name = comparison_dataframe.iloc[0]["model"]

    best_model_info = {
        "best_model": best_model_name,
        "selection_metric": "f1_score",
        "reason": "Highest F1 score on the test dataset",
    }

    best_model_path = RESULTS_DIR / "best_model.json"

    with open(
        best_model_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            best_model_info,
            file,
            indent=4,
        )

    print(f"\nBest model information saved to: {best_model_path}")
    print(f"Best model: {best_model_name}")


# =============================================================================
# 8. MAIN FUNCTION
# =============================================================================

def main():
    print("\n" + "=" * 80)
    print("STUDENT PLACEMENT PREDICTION")
    print("MODEL TRAINING PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load processed data
    # -------------------------------------------------------------------------

    X_train, X_test, y_train, y_test = load_processed_data()

    # -------------------------------------------------------------------------
    # Train models
    # -------------------------------------------------------------------------

    all_results = train_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------

    save_results(all_results)

    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 80)

    print("\nGenerated folders:")
    print(f"Models : {MODELS_DIR}")
    print(f"Results: {RESULTS_DIR}")


# =============================================================================
# RUN PROGRAM
# =============================================================================

if __name__ == "__main__":
    main()