# ==========================================================================================
# Student Placement Prediction — Model Training V2
# ==========================================================================================
#
# Models:
#   1. XGBoost with class weighting
#   2. XGBoost with SMOTE
#   3. Random Forest with class weighting
#   4. Gradient Boosting
#
# Evaluation:
#   Accuracy
#   Precision
#   Recall
#   F1-score
#   ROC-AUC
#   PR-AUC
#   Confusion matrix
#   Threshold optimization
#
# Important:
#   - salary_package_lpa is removed because it causes data leakage.
#   - SMOTE is applied only to the training data.
#   - The test set remains untouched and imbalanced.
#
# Run:
#   python model_training_v2.py
#
# ==========================================================================================

from pathlib import Path
import json
import warnings
import time

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ==========================================================================================
# 1. PROJECT PATHS
# ==========================================================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    PROJECT_DIR
    / "dataset"
    / "student_placement_synthetic.csv"
)

MODELS_DIR = PROJECT_DIR / "trained_models_v2"
RESULTS_DIR = PROJECT_DIR / "model_results_v2"
PLOTS_DIR = RESULTS_DIR / "plots"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================================================
# 2. CONFIGURATION
# ==========================================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET_COLUMN = "placement_status"

# This column must not be used for prediction.
LEAKAGE_COLUMNS = [
    "salary_package_lpa",
]

# The class labels are:
# 0 = Not Placed
# 1 = Placed


# ==========================================================================================
# 3. HELPER FUNCTIONS
# ==========================================================================================

def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def safe_filename(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def calculate_metrics(
    model_name: str,
    y_true,
    y_pred,
    y_probability,
    threshold: float = 0.50,
):
    """
    Calculate classification metrics.

    The positive class is:
        1 = Placed
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model": model_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_placed": precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        ),
        "recall_placed": recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        ),
        "f1_placed": f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        ),
        "precision_not_placed": precision_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0
        ),
        "recall_not_placed": recall_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0
        ),
        "f1_not_placed": f1_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_probability
        ),
        "pr_auc": average_precision_score(
            y_true,
            y_probability
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    return metrics


def evaluate_model(
    model_name: str,
    model,
    X_test,
    y_test,
    threshold: float = 0.50,
):
    """
    Evaluate a fitted model using a custom probability threshold.
    """

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        model_name=model_name,
        y_true=y_test,
        y_pred=predictions,
        y_probability=probabilities,
        threshold=threshold,
    )

    return metrics, predictions, probabilities


def find_best_threshold(
    y_true,
    probabilities,
    minimum_recall_not_placed: float = 0.50,
):
    """
    Find a threshold that improves detection of Not Placed students.

    A threshold is selected using:
        - Recall of Not Placed >= minimum target, when possible
        - Highest F1-score for Not Placed
    """

    threshold_rows = []

    thresholds = np.arange(
        0.10,
        0.91,
        0.01
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        not_placed_recall = recall_score(
            y_true,
            predictions,
            pos_label=0,
            zero_division=0
        )

        not_placed_precision = precision_score(
            y_true,
            predictions,
            pos_label=0,
            zero_division=0
        )

        not_placed_f1 = f1_score(
            y_true,
            predictions,
            pos_label=0,
            zero_division=0
        )

        placed_recall = recall_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0
        )

        accuracy = accuracy_score(
            y_true,
            predictions
        )

        threshold_rows.append({
            "threshold": threshold,
            "accuracy": accuracy,
            "recall_not_placed": not_placed_recall,
            "precision_not_placed": not_placed_precision,
            "f1_not_placed": not_placed_f1,
            "recall_placed": placed_recall,
        })

    threshold_df = pd.DataFrame(threshold_rows)

    eligible = threshold_df[
        threshold_df["recall_not_placed"]
        >= minimum_recall_not_placed
    ]

    if not eligible.empty:
        best_row = eligible.sort_values(
            by=[
                "f1_not_placed",
                "recall_not_placed",
                "accuracy",
            ],
            ascending=False
        ).iloc[0]

    else:
        best_row = threshold_df.sort_values(
            by="f1_not_placed",
            ascending=False
        ).iloc[0]

    return (
        float(best_row["threshold"]),
        threshold_df
    )


def save_confusion_matrix(
    model_name: str,
    y_true,
    y_pred,
):
    filename = (
        safe_filename(model_name)
        + "_confusion_matrix.png"
    )

    plot_path = PLOTS_DIR / filename

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Not Placed",
            "Placed"
        ]
    )

    display.plot()

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path


# ==========================================================================================
# 4. LOAD DATA
# ==========================================================================================

print_section("1. LOADING DATASET")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print(f"Dataset path: {DATASET_PATH}")
print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")


# ==========================================================================================
# 5. VALIDATE DATA
# ==========================================================================================

print_section("2. VALIDATING DATA")

required_columns = [
    TARGET_COLUMN,
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
    "extracurriculars",
]

missing_required_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_required_columns:
    raise ValueError(
        "The following required columns are missing:\n"
        + "\n".join(missing_required_columns)
    )

if df[TARGET_COLUMN].isna().any():
    raise ValueError(
        "The target column contains missing values."
    )

print("All required columns are available.")
print("Target values:")
print(df[TARGET_COLUMN].value_counts())


# ==========================================================================================
# 6. REMOVE LEAKAGE COLUMNS
# ==========================================================================================

print_section("3. REMOVING DATA LEAKAGE")

existing_leakage_columns = [
    column
    for column in LEAKAGE_COLUMNS
    if column in df.columns
]

print(
    "Columns removed because they contain information "
    "that would not be available before placement:"
)

for column in existing_leakage_columns:
    print(f"  - {column}")

model_df = df.drop(
    columns=existing_leakage_columns
).copy()


# ==========================================================================================
# 7. FEATURE AND TARGET SEPARATION
# ==========================================================================================

print_section("4. SEPARATING FEATURES AND TARGET")

X = model_df.drop(
    columns=[TARGET_COLUMN]
)

y = model_df[TARGET_COLUMN].astype(int)

print(f"Feature shape: {X.shape}")
print(f"Target shape: {y.shape}")

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentages:")
print(
    y.value_counts(normalize=True) * 100
)


# ==========================================================================================
# 8. TRAIN-TEST SPLIT
# ==========================================================================================

print_section("5. TRAIN-TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Training shape: {X_train.shape}")
print(f"Testing shape: {X_test.shape}")

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ==========================================================================================
# 9. IDENTIFY FEATURE TYPES
# ==========================================================================================

print_section("6. IDENTIFYING FEATURE TYPES")

categorical_features = X.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

print("Categorical features:")
for feature in categorical_features:
    print(f"  - {feature}")

print("\nNumerical features:")
for feature in numeric_features:
    print(f"  - {feature}")


# ==========================================================================================
# 10. PREPROCESSING PIPELINE
# ==========================================================================================

print_section("7. BUILDING PREPROCESSOR")

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
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
            SimpleImputer(
                strategy="most_frequent"
            )
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
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
    ],
    remainder="drop"
)


# ==========================================================================================
# 11. TRANSFORM DATA
# ==========================================================================================

print_section("8. TRANSFORMING DATA")

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

X_train_processed = np.asarray(
    X_train_processed,
    dtype=np.float32
)

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float32
)

print(
    f"Processed training shape: "
    f"{X_train_processed.shape}"
)

print(
    f"Processed testing shape: "
    f"{X_test_processed.shape}"
)


# ==========================================================================================
# 12. CALCULATE CLASS WEIGHT
# ==========================================================================================

print_section("9. CALCULATING CLASS WEIGHTS")

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

scale_pos_weight = (
    negative_count / positive_count
    if positive_count > 0
    else 1.0
)

print(f"Not Placed training samples: {negative_count:,}")
print(f"Placed training samples: {positive_count:,}")
print(f"XGBoost scale_pos_weight: {scale_pos_weight:.4f}")


# ==========================================================================================
# 13. CREATE MODELS
# ==========================================================================================

print_section("10. CREATING MODELS")

models = {
    "XGBoost Class Weight": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "Random Forest Class Weight": RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
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
}

for model_name in models:
    print(f"  - {model_name}")

print("  - XGBoost SMOTE")


# ==========================================================================================
# 14. TRAINING STORAGE
# ==========================================================================================

results = []
trained_models = {}
model_probabilities = {}
model_predictions = {}


# ==========================================================================================
# 15. TRAIN STANDARD MODELS
# ==========================================================================================

print_section("11. TRAINING STANDARD MODELS")

for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    start_time = time.time()

    model.fit(
        X_train_processed,
        y_train
    )

    elapsed_time = time.time() - start_time

    metrics, predictions, probabilities = evaluate_model(
        model_name=model_name,
        model=model,
        X_test=X_test_processed,
        y_test=y_test,
        threshold=0.50,
    )

    metrics["training_time_seconds"] = elapsed_time

    results.append(metrics)

    trained_models[model_name] = model
    model_probabilities[model_name] = probabilities
    model_predictions[model_name] = predictions

    print(f"Training time: {elapsed_time:.2f} seconds")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Placed precision: {metrics['precision_placed']:.4f}")
    print(f"Placed recall: {metrics['recall_placed']:.4f}")
    print(f"Placed F1: {metrics['f1_placed']:.4f}")
    print(
        f"Not Placed recall: "
        f"{metrics['recall_not_placed']:.4f}"
    )
    print(
        f"Not Placed F1: "
        f"{metrics['f1_not_placed']:.4f}"
    )
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Not Placed",
                "Placed"
            ],
            zero_division=0
        )
    )

    plot_path = save_confusion_matrix(
        model_name=model_name,
        y_true=y_test,
        y_pred=predictions,
    )

    print(f"Saved confusion matrix: {plot_path}")


# ==========================================================================================
# 16. TRAIN XGBOOST WITH SMOTE
# ==========================================================================================

print_section("12. TRAINING XGBOOST WITH SMOTE")

smote = SMOTE(
    random_state=RANDOM_STATE,
    sampling_strategy="auto",
    k_neighbors=5,
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train
)

print("Training shape before SMOTE:")
print(X_train_processed.shape)

print("\nTraining shape after SMOTE:")
print(X_train_smote.shape)

print("\nClass distribution after SMOTE:")
print(
    pd.Series(y_train_smote).value_counts()
)

xgb_smote = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=2,
    gamma=0,
    reg_alpha=0.05,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

start_time = time.time()

xgb_smote.fit(
    X_train_smote,
    y_train_smote
)

elapsed_time = time.time() - start_time

smote_model_name = "XGBoost SMOTE"

metrics, predictions, probabilities = evaluate_model(
    model_name=smote_model_name,
    model=xgb_smote,
    X_test=X_test_processed,
    y_test=y_test,
    threshold=0.50,
)

metrics["training_time_seconds"] = elapsed_time

results.append(metrics)

trained_models[smote_model_name] = xgb_smote
model_probabilities[smote_model_name] = probabilities
model_predictions[smote_model_name] = predictions

print(f"Training time: {elapsed_time:.2f} seconds")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Placed precision: {metrics['precision_placed']:.4f}")
print(f"Placed recall: {metrics['recall_placed']:.4f}")
print(f"Placed F1: {metrics['f1_placed']:.4f}")
print(
    f"Not Placed recall: "
    f"{metrics['recall_not_placed']:.4f}"
)
print(
    f"Not Placed F1: "
    f"{metrics['f1_not_placed']:.4f}"
)
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"PR-AUC: {metrics['pr_auc']:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Not Placed",
            "Placed"
        ],
        zero_division=0
    )
)

plot_path = save_confusion_matrix(
    model_name=smote_model_name,
    y_true=y_test,
    y_pred=predictions,
)

print(f"Saved confusion matrix: {plot_path}")


# ==========================================================================================
# 17. SAVE INITIAL MODEL RESULTS
# ==========================================================================================

print_section("13. SAVING INITIAL RESULTS")

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1_not_placed",
    ascending=False
)

print(results_df.to_string(index=False))

results_path = RESULTS_DIR / "model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False
)

print(f"Saved: {results_path}")


# ==========================================================================================
# 18. THRESHOLD OPTIMIZATION
# ==========================================================================================

print_section("14. OPTIMIZING PROBABILITY THRESHOLDS")

threshold_results = []
best_thresholds = {}

for model_name, probabilities in model_probabilities.items():

    print(f"\nOptimizing threshold for: {model_name}")

    best_threshold, threshold_df = find_best_threshold(
        y_true=y_test,
        probabilities=probabilities,
        minimum_recall_not_placed=0.50,
    )

    threshold_df["model"] = model_name

    threshold_results.append(threshold_df)

    best_thresholds[model_name] = best_threshold

    optimized_predictions = (
        probabilities >= best_threshold
    ).astype(int)

    optimized_metrics = calculate_metrics(
        model_name=model_name,
        y_true=y_test,
        y_pred=optimized_predictions,
        y_probability=probabilities,
        threshold=best_threshold,
    )

    optimized_metrics["threshold_type"] = (
        "optimized"
    )

    results.append(optimized_metrics)

    print(f"Best threshold: {best_threshold:.2f}")
    print(
        f"Not Placed recall: "
        f"{optimized_metrics['recall_not_placed']:.4f}"
    )
    print(
        f"Not Placed F1: "
        f"{optimized_metrics['f1_not_placed']:.4f}"
    )
    print(
        f"Placed recall: "
        f"{optimized_metrics['recall_placed']:.4f}"
    )
    print(
        f"Accuracy: "
        f"{optimized_metrics['accuracy']:.4f}"
    )

    save_confusion_matrix(
        model_name=(
            model_name
            + f" Threshold {best_threshold:.2f}"
        ),
        y_true=y_test,
        y_pred=optimized_predictions,
    )

    threshold_plot_path = (
        PLOTS_DIR
        / f"{safe_filename(model_name)}_thresholds.png"
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        threshold_df["threshold"],
        threshold_df["recall_not_placed"],
        label="Not Placed Recall"
    )

    plt.plot(
        threshold_df["threshold"],
        threshold_df["f1_not_placed"],
        label="Not Placed F1"
    )

    plt.plot(
        threshold_df["threshold"],
        threshold_df["accuracy"],
        label="Accuracy"
    )

    plt.axvline(
        best_threshold,
        linestyle="--",
        label=f"Best threshold = {best_threshold:.2f}"
    )

    plt.xlabel("Probability Threshold")
    plt.ylabel("Score")
    plt.title(
        f"{model_name} - Threshold Optimization"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        threshold_plot_path,
        dpi=150
    )

    plt.close()

    print(f"Saved: {threshold_plot_path}")


threshold_results_df = pd.concat(
    threshold_results,
    ignore_index=True
)

threshold_results_path = (
    RESULTS_DIR / "threshold_analysis.csv"
)

threshold_results_df.to_csv(
    threshold_results_path,
    index=False
)

print(f"Saved: {threshold_results_path}")


# ==========================================================================================
# 19. ROC CURVES
# ==========================================================================================

print_section("15. GENERATING ROC CURVES")

plt.figure(figsize=(9, 6))

for model_name, probabilities in model_probabilities.items():

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        probabilities
    )

    auc_score = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"{model_name} AUC={auc_score:.3f}"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

roc_plot_path = PLOTS_DIR / "roc_curve_comparison.png"

plt.savefig(
    roc_plot_path,
    dpi=150
)

plt.close()

print(f"Saved: {roc_plot_path}")


# ==========================================================================================
# 20. PRECISION-RECALL CURVES
# ==========================================================================================

print_section("16. GENERATING PRECISION-RECALL CURVES")

plt.figure(figsize=(9, 6))

for model_name, probabilities in model_probabilities.items():

    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    plt.plot(
        recall,
        precision,
        label=f"{model_name} PR-AUC={pr_auc:.3f}"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

pr_plot_path = (
    PLOTS_DIR / "precision_recall_curve_comparison.png"
)

plt.savefig(
    pr_plot_path,
    dpi=150
)

plt.close()

print(f"Saved: {pr_plot_path}")


# ==========================================================================================
# 21. SELECT BEST MODEL
# ==========================================================================================

print_section("17. SELECTING BEST MODEL")

all_results_df = pd.DataFrame(results)

all_results_df = all_results_df.sort_values(
    by=[
        "f1_not_placed",
        "recall_not_placed",
        "pr_auc",
    ],
    ascending=False
)

all_results_path = (
    RESULTS_DIR / "all_model_results.csv"
)

all_results_df.to_csv(
    all_results_path,
    index=False
)

print(all_results_df.to_string(index=False))

best_result = all_results_df.iloc[0]

best_model_name = best_result["model"]
best_threshold = float(best_result["threshold"])

best_model = trained_models[best_model_name]

print("\nBest model:")
print(f"Model: {best_model_name}")
print(f"Threshold: {best_threshold:.2f}")
print(
    f"Not Placed F1: "
    f"{best_result['f1_not_placed']:.4f}"
)
print(
    f"Not Placed Recall: "
    f"{best_result['recall_not_placed']:.4f}"
)
print(
    f"Placed Recall: "
    f"{best_result['recall_placed']:.4f}"
)
print(
    f"PR-AUC: "
    f"{best_result['pr_auc']:.4f}"
)


# ==========================================================================================
# 22. SAVE BEST MODEL
# ==========================================================================================

print_section("18. SAVING BEST MODEL")

best_model_filename = (
    safe_filename(best_model_name)
    + ".joblib"
)

best_model_path = (
    MODELS_DIR / best_model_filename
)

joblib.dump(
    best_model,
    best_model_path
)

print(f"Saved best model: {best_model_path}")


# ==========================================================================================
# 23. SAVE PREPROCESSOR
# ==========================================================================================

preprocessor_path = (
    MODELS_DIR / "preprocessor.joblib"
)

joblib.dump(
    preprocessor,
    preprocessor_path
)

print(f"Saved preprocessor: {preprocessor_path}")


# ==========================================================================================
# 24. SAVE FEATURE INFORMATION
# ==========================================================================================

feature_information = {
    "target_column": TARGET_COLUMN,
    "leakage_columns_removed": existing_leakage_columns,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "original_feature_count": int(X.shape[1]),
    "processed_feature_count": int(
        X_train_processed.shape[1]
    ),
    "best_model": best_model_name,
    "best_threshold": best_threshold,
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
}

feature_information_path = (
    MODELS_DIR / "feature_information.json"
)

with open(
    feature_information_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        feature_information,
        file,
        indent=4
    )

print(
    f"Saved feature information: "
    f"{feature_information_path}"
)


# ==========================================================================================
# 25. SAVE BEST MODEL METRICS
# ==========================================================================================

best_metrics_path = (
    RESULTS_DIR / "best_model_metrics.json"
)

best_metrics = best_result.to_dict()

with open(
    best_metrics_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        best_metrics,
        file,
        indent=4,
        default=float
    )

print(f"Saved best metrics: {best_metrics_path}")


# ==========================================================================================
# 26. SAVE TEST PREDICTIONS
# ==========================================================================================

print_section("19. SAVING TEST PREDICTIONS")

best_probabilities = (
    model_probabilities[best_model_name]
)

best_predictions = (
    best_probabilities >= best_threshold
).astype(int)

test_predictions_df = X_test.copy()

test_predictions_df["actual_placement_status"] = (
    y_test.values
)

test_predictions_df["predicted_placement_status"] = (
    best_predictions
)

test_predictions_df["placement_probability"] = (
    best_probabilities
)

test_predictions_df["prediction_correct"] = (
    test_predictions_df[
        "actual_placement_status"
    ]
    ==
    test_predictions_df[
        "predicted_placement_status"
    ]
)

predictions_path = (
    RESULTS_DIR / "best_model_test_predictions.csv"
)

test_predictions_df.to_csv(
    predictions_path,
    index=False
)

print(f"Saved: {predictions_path}")


# ==========================================================================================
# 27. FINAL SUMMARY
# ==========================================================================================

print_section("20. TRAINING COMPLETED")

print("Best model:")
print(f"  {best_model_name}")

print(f"\nBest threshold:")
print(f"  {best_threshold:.2f}")

print("\nBest model metrics:")

for key, value in best_metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

print("\nSaved directories:")
print(f"  Models: {MODELS_DIR}")
print(f"  Results: {RESULTS_DIR}")
print(f"  Plots: {PLOTS_DIR}")

print("\nNext step:")
print(
    "Update interface.py to load the V2 preprocessor, "
    "V2 best model, and V2 probability threshold."
)