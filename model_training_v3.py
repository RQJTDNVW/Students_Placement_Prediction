# ==========================================================================================
# Student Placement Prediction
# Model Training V3
# ==========================================================================================
#
# Purpose:
#   1. Load preprocessed V3 data
#   2. Train multiple classification models
#   3. Evaluate models on untouched test data
#   4. Select the best model using ROC-AUC and F1 score
#   5. Save all trained models
#   6. Evaluate the best model on external validation data
#
# Run:
#   python model_training_v3.py
#
# ==========================================================================================

from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.svm import SVC

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ==========================================================================================
# 1. PROJECT PATHS
# ==========================================================================================

PROJECT_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = PROJECT_DIR / "processed_data_v3"
MODELS_DIR = PROJECT_DIR / "trained_models_v3"
RESULTS_DIR = PROJECT_DIR / "model_results_v3"

EXTERNAL_FILE = (
    PROCESSED_DIR / "external_validation_dataset_v3.csv"
)

PREPROCESSOR_FILE = (
    PROCESSED_DIR / "preprocessor_v3.joblib"
)

FEATURE_NAMES_FILE = (
    PROCESSED_DIR / "processed_feature_names_v3.csv"
)

X_TRAIN_FILE = (
    PROCESSED_DIR / "X_train_balanced_v3.npy"
)

Y_TRAIN_FILE = (
    PROCESSED_DIR / "y_train_balanced_v3.npy"
)

X_TEST_FILE = (
    PROCESSED_DIR / "X_test_processed_v3.npy"
)

Y_TEST_FILE = (
    PROCESSED_DIR / "y_test_v3.npy"
)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================================================
# 2. CONFIGURATION
# ==========================================================================================

RANDOM_STATE = 42

TARGET_COLUMN = "placement_status"

# The model-selection priority.
# ROC-AUC is used first because it measures ranking quality.
# F1 score is used as a secondary criterion.
PRIMARY_METRIC = "roc_auc"


# ==========================================================================================
# 3. UTILITY FUNCTIONS
# ==========================================================================================

def print_section(title: str):
    """Print a formatted section heading."""

    print("\n" + "=" * 95)
    print(title)
    print("=" * 95)


def save_json(data: dict, path: Path):
    """Save a dictionary as formatted JSON."""

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)


def safe_model_name(model_name: str) -> str:
    """Convert a model name into a safe filename."""

    return (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def get_probability_predictions(model, X):
    """
    Return probability for class 1.

    Models with predict_proba:
        use predict_proba

    Models without predict_proba:
        use decision_function and convert scores
        into a 0-1 range using the logistic function.
    """

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

        if probabilities.ndim == 2:
            return probabilities[:, 1]

        return probabilities

    if hasattr(model, "decision_function"):
        decision_scores = model.decision_function(X)

        return 1.0 / (1.0 + np.exp(-decision_scores))

    predictions = model.predict(X)

    return np.asarray(predictions, dtype=float)


def calculate_metrics(
    y_true,
    y_pred,
    y_probability,
) -> dict:
    """Calculate classification metrics."""

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Not Placed", "Placed"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_probability,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                y_probability,
            )
        ),
        "confusion_matrix": matrix.tolist(),
        "not_placed_precision": float(
            report["Not Placed"]["precision"]
        ),
        "not_placed_recall": float(
            report["Not Placed"]["recall"]
        ),
        "not_placed_f1": float(
            report["Not Placed"]["f1-score"]
        ),
        "placed_precision": float(
            report["Placed"]["precision"]
        ),
        "placed_recall": float(
            report["Placed"]["recall"]
        ),
        "placed_f1": float(
            report["Placed"]["f1-score"]
        ),
    }

    return metrics


def print_metrics(model_name: str, metrics: dict):
    """Print model metrics."""

    print(f"\n{model_name}")

    print("-" * len(model_name))

    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"F1 Score:           {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:             {metrics['pr_auc']:.4f}")

    print("\nNot Placed:")
    print(
        f"  Precision: {metrics['not_placed_precision']:.4f}"
    )
    print(
        f"  Recall:    {metrics['not_placed_recall']:.4f}"
    )
    print(
        f"  F1 Score:  {metrics['not_placed_f1']:.4f}"
    )

    print("\nPlaced:")
    print(
        f"  Precision: {metrics['placed_precision']:.4f}"
    )
    print(
        f"  Recall:    {metrics['placed_recall']:.4f}"
    )
    print(
        f"  F1 Score:  {metrics['placed_f1']:.4f}"
    )

    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))


# ==========================================================================================
# 4. LOAD PROCESSED DATA
# ==========================================================================================

def load_processed_data():
    """Load training and testing arrays."""

    print_section("LOADING PROCESSED V3 DATA")

    required_files = [
        X_TRAIN_FILE,
        Y_TRAIN_FILE,
        X_TEST_FILE,
        Y_TEST_FILE,
        PREPROCESSOR_FILE,
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"Required file was not found:\n{file}\n\n"
                "Run data_preprocessing_v3.py first."
            )

    X_train = np.load(X_TRAIN_FILE)
    y_train = np.load(Y_TRAIN_FILE)

    X_test = np.load(X_TEST_FILE)
    y_test = np.load(Y_TEST_FILE)

    preprocessor = joblib.load(PREPROCESSOR_FILE)

    print(f"Training data shape: {X_train.shape}")
    print(f"Training target shape: {y_train.shape}")

    print(f"Testing data shape: {X_test.shape}")
    print(f"Testing target shape: {y_test.shape}")

    print("\nTraining target distribution:")
    print(pd.Series(y_train).value_counts().sort_index())

    print("\nTesting target distribution:")
    print(pd.Series(y_test).value_counts().sort_index())

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        preprocessor,
    )


# ==========================================================================================
# 5. BUILD MODELS
# ==========================================================================================

def build_models():
    """
    Build the classification models.

    The training data has already been balanced using SMOTE.
    Therefore, class_weight is not used for most models.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),

        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),

        "Support Vector Machine": SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
        ),
    }

    return models


# ==========================================================================================
# 6. TRAIN AND EVALUATE MODELS
# ==========================================================================================

def train_and_evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """Train and evaluate all models."""

    print_section("TRAINING AND EVALUATING MODELS")

    models = build_models()

    all_metrics = {}
    trained_models = {}

    for model_name, model in models.items():

        print_section(f"TRAINING: {model_name}")

        start_time = time.perf_counter()

        try:
            model.fit(X_train, y_train)

            training_time = time.perf_counter() - start_time

            y_pred = model.predict(X_test)
            y_probability = get_probability_predictions(
                model,
                X_test,
            )

            metrics = calculate_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_probability=y_probability,
            )

            metrics["training_time_seconds"] = float(
                training_time
            )

            all_metrics[model_name] = metrics
            trained_models[model_name] = model

            print_metrics(
                model_name,
                metrics,
            )

            print(
                f"\nTraining time: "
                f"{training_time:.2f} seconds"
            )

        except Exception as error:
            print(
                f"\nERROR while training {model_name}:"
            )
            print(error)

    return trained_models, all_metrics


# ==========================================================================================
# 7. SELECT BEST MODEL
# ==========================================================================================

def select_best_model(
    trained_models: dict,
    all_metrics: dict,
):
    """
    Select the best model.

    Ranking:
        1. ROC-AUC
        2. F1 score
        3. Accuracy
    """

    if not all_metrics:
        raise RuntimeError(
            "No models were successfully trained."
        )

    ranking = sorted(
        all_metrics.keys(),
        key=lambda name: (
            all_metrics[name]["roc_auc"],
            all_metrics[name]["f1_score"],
            all_metrics[name]["accuracy"],
        ),
        reverse=True,
    )

    best_model_name = ranking[0]
    best_model = trained_models[best_model_name]

    print_section("MODEL COMPARISON")

    comparison_rows = []

    for model_name in ranking:
        metrics = all_metrics[model_name]

        comparison_rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "training_time_seconds": metrics[
                    "training_time_seconds"
                ],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(
        f"\nBest model based on ROC-AUC: "
        f"{best_model_name}"
    )

    return (
        best_model_name,
        best_model,
        comparison_df,
    )


# ==========================================================================================
# 8. SAVE TRAINED MODELS AND RESULTS
# ==========================================================================================

def save_models_and_results(
    trained_models: dict,
    all_metrics: dict,
    best_model_name: str,
    best_model,
    comparison_df: pd.DataFrame,
):
    """Save models, metrics, and comparison results."""

    print_section("SAVING MODELS AND RESULTS")

    for model_name, model in trained_models.items():

        filename = (
            safe_model_name(model_name)
            + ".joblib"
        )

        model_path = MODELS_DIR / filename

        joblib.dump(
            model,
            model_path,
        )

        print(f"Saved: {model_path.name}")

    best_model_path = (
        MODELS_DIR
        / f"best_model_{safe_model_name(best_model_name)}.joblib"
    )

    joblib.dump(
        best_model,
        best_model_path,
    )

    comparison_df.to_csv(
        RESULTS_DIR / "model_comparison_v3.csv",
        index=False,
    )

    save_json(
        all_metrics,
        RESULTS_DIR / "all_model_metrics_v3.json",
    )

    best_model_metrics = {
        "best_model_name": best_model_name,
        "best_model_file": best_model_path.name,
        "selection_metric": PRIMARY_METRIC,
        "metrics": all_metrics[best_model_name],
    }

    save_json(
        best_model_metrics,
        RESULTS_DIR / "best_model_metrics_v3.json",
    )

    print(f"Saved best model: {best_model_path.name}")


# ==========================================================================================
# 9. SAVE TEST PREDICTIONS
# ==========================================================================================

def save_test_predictions(
    best_model_name: str,
    best_model,
    X_test,
    y_test,
):
    """Save predictions for the untouched test set."""

    print_section("SAVING TEST PREDICTIONS")

    predictions = best_model.predict(X_test)

    probabilities = get_probability_predictions(
        best_model,
        X_test,
    )

    prediction_df = pd.DataFrame(
        {
            "actual_placement_status": y_test,
            "predicted_placement_status": predictions,
            "placement_probability": probabilities,
            "predicted_status": np.where(
                predictions == 1,
                "Placed",
                "Not Placed",
            ),
        }
    )

    prediction_file = (
        RESULTS_DIR / "best_model_test_predictions_v3.csv"
    )

    prediction_df.to_csv(
        prediction_file,
        index=False,
    )

    print(f"Saved: {prediction_file}")


# ==========================================================================================
# 10. EXTERNAL VALIDATION
# ==========================================================================================

def evaluate_external_dataset(
    best_model_name: str,
    best_model,
    preprocessor,
):
    """
    Evaluate the best model on the external dataset.

    The external dataset is transformed using the preprocessor
    fitted on the training data.
    """

    print_section("EXTERNAL VALIDATION")

    if not EXTERNAL_FILE.exists():
        print(
            "External validation file was not found."
        )
        print(EXTERNAL_FILE)
        return

    external_df = pd.read_csv(EXTERNAL_FILE)

    print(
        f"External dataset shape: "
        f"{external_df.shape}"
    )

    if TARGET_COLUMN not in external_df.columns:
        print(
            f"Target column '{TARGET_COLUMN}' "
            "not found in external dataset."
        )
        return

    y_external = external_df[TARGET_COLUMN].astype(int)

    X_external = external_df.drop(
        columns=[TARGET_COLUMN],
        errors="ignore",
    )

    # Keep only the columns expected by the preprocessor.
    expected_columns = []

    for transformer_name, transformer, columns in (
        preprocessor.transformers_
    ):
        if transformer_name == "remainder":
            continue

        expected_columns.extend(list(columns))

    for column in expected_columns:
        if column not in X_external.columns:
            X_external[column] = np.nan

    X_external = X_external[expected_columns]

    # Transform external data using the already-fitted preprocessor.
    X_external_processed = preprocessor.transform(
        X_external
    )

    X_external_processed = np.asarray(
        X_external_processed,
        dtype=np.float32,
    )

    print(
        f"External processed shape: "
        f"{X_external_processed.shape}"
    )

    predictions = best_model.predict(
        X_external_processed
    )

    probabilities = get_probability_predictions(
        best_model,
        X_external_processed,
    )

    external_metrics = calculate_metrics(
        y_true=y_external,
        y_pred=predictions,
        y_probability=probabilities,
    )

    external_metrics["model_name"] = best_model_name
    external_metrics["external_sample_count"] = int(
        len(y_external)
    )

    print_metrics(
        f"{best_model_name} - External Validation",
        external_metrics,
    )

    external_prediction_df = pd.DataFrame(
        {
            "actual_placement_status": y_external,
            "predicted_placement_status": predictions,
            "placement_probability": probabilities,
            "predicted_status": np.where(
                predictions == 1,
                "Placed",
                "Not Placed",
            ),
        }
    )

    external_prediction_file = (
        RESULTS_DIR
        / "best_model_external_predictions_v3.csv"
    )

    external_prediction_df.to_csv(
        external_prediction_file,
        index=False,
    )

    save_json(
        external_metrics,
        RESULTS_DIR / "external_validation_metrics_v3.json",
    )

    print(
        f"\nSaved external predictions: "
        f"{external_prediction_file}"
    )


# ==========================================================================================
# 11. MAIN TRAINING PIPELINE
# ==========================================================================================

def run_training_pipeline():
    """Run the complete V3 model training pipeline."""

    print_section(
        "STUDENT PLACEMENT PREDICTION - MODEL TRAINING V3"
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        preprocessor,
    ) = load_processed_data()

    trained_models, all_metrics = (
        train_and_evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
    )

    (
        best_model_name,
        best_model,
        comparison_df,
    ) = select_best_model(
        trained_models=trained_models,
        all_metrics=all_metrics,
    )

    save_models_and_results(
        trained_models=trained_models,
        all_metrics=all_metrics,
        best_model_name=best_model_name,
        best_model=best_model,
        comparison_df=comparison_df,
    )

    save_test_predictions(
        best_model_name=best_model_name,
        best_model=best_model,
        X_test=X_test,
        y_test=y_test,
    )

    evaluate_external_dataset(
        best_model_name=best_model_name,
        best_model=best_model,
        preprocessor=preprocessor,
    )

    print_section(
        "MODEL TRAINING V3 COMPLETED SUCCESSFULLY"
    )

    print(f"Models directory:\n{MODELS_DIR}")
    print(f"Results directory:\n{RESULTS_DIR}")

    print("\nNext step:")
    print("Run interface_v3.py")


# ==========================================================================================
# 12. SCRIPT ENTRY POINT
# ==========================================================================================

if __name__ == "__main__":
    run_training_pipeline()