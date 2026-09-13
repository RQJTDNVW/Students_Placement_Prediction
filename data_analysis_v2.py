# ==========================================================================================
# Student Placement Prediction — Data Analysis V2
# ==========================================================================================
#
# Purpose:
#   1. Inspect the original dataset
#   2. Analyze target-class imbalance
#   3. Detect missing values and duplicates
#   4. Identify possible data leakage
#   5. Analyze numerical and categorical features
#   6. Calculate feature relationships with placement_status
#   7. Generate useful visualizations
#   8. Save analysis reports without modifying the original dataset
#
# Run:
#   python data_analysis_v2.py
#
# ==========================================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ==========================================================================================
# 1. PROJECT PATHS
# ==========================================================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_DIR / "dataset" / "student_placement_synthetic.csv"

OUTPUT_DIR = PROJECT_DIR / "analysis_results_v2"
PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================================================
# 2. DISPLAY SETTINGS
# ==========================================================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 100)

RANDOM_STATE = 42


# ==========================================================================================
# 3. HELPER FUNCTIONS
# ==========================================================================================

def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def save_dataframe(
    dataframe: pd.DataFrame,
    filename: str,
    index: bool = True
):
    output_path = OUTPUT_DIR / filename
    dataframe.to_csv(output_path, index=index)
    print(f"Saved: {output_path}")


def normalize_target_value(value):
    """
    Convert common placement labels into 0 and 1.

    Supported examples:
        Placed, Yes, True, 1  -> 1
        Not Placed, No, False, 0 -> 0
    """

    if pd.isna(value):
        return np.nan

    value_string = str(value).strip().lower()

    positive_values = {
        "1",
        "yes",
        "y",
        "true",
        "placed",
        "selected",
        "successful",
        "success",
    }

    negative_values = {
        "0",
        "no",
        "n",
        "false",
        "not placed",
        "not_placed",
        "notplaced",
        "unplaced",
        "not selected",
        "not_selected",
        "unsuccessful",
        "failure",
        "failed",
    }

    if value_string in positive_values:
        return 1

    if value_string in negative_values:
        return 0

    try:
        numeric_value = float(value_string)

        if numeric_value in [0, 1]:
            return int(numeric_value)

    except ValueError:
        pass

    return value


def calculate_numeric_target_relationship(
    dataframe: pd.DataFrame,
    target_column: str
) -> pd.DataFrame:

    numeric_columns = dataframe.select_dtypes(
        include=np.number
    ).columns.tolist()

    numeric_columns = [
        column
        for column in numeric_columns
        if column != target_column
    ]

    results = []

    for column in numeric_columns:
        valid_data = dataframe[[column, target_column]].dropna()

        if len(valid_data) < 2:
            correlation = np.nan
        else:
            correlation = valid_data[column].corr(
                valid_data[target_column]
            )

        results.append({
            "feature": column,
            "correlation_with_target": correlation,
            "absolute_correlation": (
                abs(correlation)
                if not pd.isna(correlation)
                else np.nan
            ),
            "missing_values": int(dataframe[column].isna().sum()),
            "unique_values": int(dataframe[column].nunique()),
            "mean": dataframe[column].mean(),
            "median": dataframe[column].median(),
            "minimum": dataframe[column].min(),
            "maximum": dataframe[column].max(),
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values(
            by="absolute_correlation",
            ascending=False
        )

    return results_df


def calculate_categorical_target_relationship(
    dataframe: pd.DataFrame,
    target_column: str
) -> pd.DataFrame:

    categorical_columns = dataframe.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in categorical_columns
        if column != target_column
    ]

    results = []

    for column in categorical_columns:
        grouped = (
            dataframe.groupby(column, dropna=False)[target_column]
            .agg(["count", "mean"])
            .reset_index()
        )

        grouped = grouped.rename(
            columns={
                "count": "sample_count",
                "mean": "placement_rate"
            }
        )

        grouped["feature"] = column

        results.append(grouped)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# ==========================================================================================
# 4. LOAD DATASET
# ==========================================================================================

print_section("1. LOADING DATASET")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}\n\n"
        "Check that student_placement_synthetic.csv is inside the dataset folder."
    )

df = pd.read_csv(DATASET_PATH)

print(f"Dataset path: {DATASET_PATH}")
print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")


# ==========================================================================================
# 5. BASIC DATASET INFORMATION
# ==========================================================================================

print_section("2. BASIC DATASET INFORMATION")

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
for index, column in enumerate(df.columns, start=1):
    print(f"{index:02d}. {column}")

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()


# ==========================================================================================
# 6. DUPLICATE ANALYSIS
# ==========================================================================================

print_section("3. DUPLICATE ANALYSIS")

duplicate_count = int(df.duplicated().sum())
duplicate_percentage = (
    duplicate_count / len(df) * 100
    if len(df) > 0
    else 0
)

print(f"Duplicate rows: {duplicate_count:,}")
print(f"Duplicate percentage: {duplicate_percentage:.4f}%")

duplicate_report = pd.DataFrame({
    "metric": [
        "total_rows",
        "duplicate_rows",
        "duplicate_percentage",
    ],
    "value": [
        len(df),
        duplicate_count,
        duplicate_percentage,
    ],
})

save_dataframe(
    duplicate_report,
    "duplicate_report.csv",
    index=False
)


# ==========================================================================================
# 7. MISSING-VALUE ANALYSIS
# ==========================================================================================

print_section("4. MISSING-VALUE ANALYSIS")

missing_report = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_percentage": (
        df.isna().sum().values / len(df) * 100
    ),
    "data_type": df.dtypes.astype(str).values,
})

missing_report = missing_report.sort_values(
    by="missing_count",
    ascending=False
)

print(missing_report.to_string(index=False))

save_dataframe(
    missing_report,
    "missing_values_report.csv",
    index=False
)

missing_columns = missing_report[
    missing_report["missing_count"] > 0
]

if missing_columns.empty:
    print("\nNo missing values found.")
else:
    print("\nColumns containing missing values:")
    print(missing_columns["column"].tolist())


# ==========================================================================================
# 8. TARGET ANALYSIS
# ==========================================================================================

print_section("5. TARGET ANALYSIS")

TARGET_COLUMN = "placement_status"

if TARGET_COLUMN not in df.columns:
    raise KeyError(
        f"Target column '{TARGET_COLUMN}' was not found in the dataset."
    )

print("Original target values:")
print(df[TARGET_COLUMN].value_counts(dropna=False))

target_analysis_df = df.copy()

target_analysis_df["target_numeric"] = (
    target_analysis_df[TARGET_COLUMN]
    .apply(normalize_target_value)
)

print("\nNormalized target values:")
print(
    target_analysis_df["target_numeric"]
    .value_counts(dropna=False)
)

target_counts = (
    target_analysis_df["target_numeric"]
    .value_counts(dropna=False)
    .rename_axis("target")
    .reset_index(name="count")
)

target_counts["percentage"] = (
    target_counts["count"] / len(target_analysis_df) * 100
)

print("\nTarget distribution:")
print(target_counts.to_string(index=False))

save_dataframe(
    target_counts,
    "target_distribution.csv",
    index=False
)

valid_target = target_analysis_df["target_numeric"].dropna()

if len(valid_target) > 0:
    class_counts = valid_target.value_counts()

    majority_count = class_counts.max()
    minority_count = class_counts.min()

    imbalance_ratio = (
        majority_count / minority_count
        if minority_count > 0
        else np.inf
    )

    print(f"\nMajority-class count: {majority_count:,}")
    print(f"Minority-class count: {minority_count:,}")
    print(f"Imbalance ratio: {imbalance_ratio:.4f}:1")


# ==========================================================================================
# 9. TARGET DISTRIBUTION PLOT
# ==========================================================================================

print_section("6. TARGET DISTRIBUTION PLOT")

plt.figure(figsize=(8, 5))

target_counts_plot = (
    target_analysis_df["target_numeric"]
    .value_counts()
    .sort_index()
)

labels = []

for target_value in target_counts_plot.index:
    if target_value == 1:
        labels.append("Placed")
    elif target_value == 0:
        labels.append("Not Placed")
    else:
        labels.append(str(target_value))

plt.bar(
    labels,
    target_counts_plot.values
)

plt.title("Placement Status Distribution")
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

target_plot_path = PLOTS_DIR / "target_distribution.png"
plt.savefig(target_plot_path, dpi=150)
plt.close()

print(f"Saved: {target_plot_path}")


# ==========================================================================================
# 10. NUMERICAL FEATURE ANALYSIS
# ==========================================================================================

print_section("7. NUMERICAL FEATURE ANALYSIS")

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

print(f"Numerical columns: {len(numeric_columns)}")

for column in numeric_columns:
    print(f"  - {column}")

numeric_summary = df[numeric_columns].describe().T

numeric_summary["missing_count"] = (
    df[numeric_columns].isna().sum()
)

numeric_summary["missing_percentage"] = (
    numeric_summary["missing_count"] / len(df) * 100
)

numeric_summary["unique_values"] = (
    df[numeric_columns].nunique()
)

save_dataframe(
    numeric_summary,
    "numeric_summary.csv"
)

print("\nNumerical summary:")
print(numeric_summary)


# ==========================================================================================
# 11. NUMERICAL FEATURE DISTRIBUTIONS
# ==========================================================================================

print_section("8. NUMERICAL FEATURE DISTRIBUTIONS")

for column in numeric_columns:
    if column == TARGET_COLUMN:
        continue

    plt.figure(figsize=(8, 5))

    plt.hist(
        df[column].dropna(),
        bins=30
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    safe_column_name = (
        column.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )

    plot_path = PLOTS_DIR / f"distribution_{safe_column_name}.png"

    plt.savefig(plot_path, dpi=150)
    plt.close()

print("Numerical distribution plots generated.")


# ==========================================================================================
# 12. CATEGORICAL FEATURE ANALYSIS
# ==========================================================================================

print_section("9. CATEGORICAL FEATURE ANALYSIS")

categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print(f"Categorical columns: {len(categorical_columns)}")

for column in categorical_columns:
    print(f"\nColumn: {column}")
    print(f"Unique values: {df[column].nunique(dropna=False)}")
    print(df[column].value_counts(dropna=False).head(20))

categorical_summary_rows = []

for column in categorical_columns:
    categorical_summary_rows.append({
        "column": column,
        "unique_values": df[column].nunique(dropna=False),
        "missing_count": int(df[column].isna().sum()),
        "missing_percentage": (
            df[column].isna().sum() / len(df) * 100
        ),
        "top_value": (
            df[column].mode(dropna=True).iloc[0]
            if not df[column].mode(dropna=True).empty
            else np.nan
        ),
    })

categorical_summary = pd.DataFrame(
    categorical_summary_rows
)

save_dataframe(
    categorical_summary,
    "categorical_summary.csv",
    index=False
)


# ==========================================================================================
# 13. NUMERICAL FEATURE RELATIONSHIP WITH TARGET
# ==========================================================================================

print_section("10. NUMERICAL FEATURE RELATIONSHIP WITH TARGET")

target_relationship_df = target_analysis_df.copy()

target_relationship_df[TARGET_COLUMN] = (
    target_relationship_df["target_numeric"]
)

numeric_target_relationship = (
    calculate_numeric_target_relationship(
        target_relationship_df,
        TARGET_COLUMN
    )
)

print(numeric_target_relationship.to_string(index=False))

save_dataframe(
    numeric_target_relationship,
    "numeric_target_relationship.csv",
    index=False
)


# ==========================================================================================
# 14. CATEGORICAL FEATURE RELATIONSHIP WITH TARGET
# ==========================================================================================

print_section("11. CATEGORICAL FEATURE RELATIONSHIP WITH TARGET")

categorical_target_relationship = (
    calculate_categorical_target_relationship(
        target_relationship_df,
        TARGET_COLUMN
    )
)

if categorical_target_relationship.empty:
    print("No categorical columns available.")
else:
    print(categorical_target_relationship.to_string(index=False))

    save_dataframe(
        categorical_target_relationship,
        "categorical_target_relationship.csv",
        index=False
    )


# ==========================================================================================
# 15. PLACEMENT RATE BY NUMERICAL FEATURES
# ==========================================================================================

print_section("12. PLACEMENT RATE BY NUMERICAL FEATURES")

numeric_features_without_target = [
    column
    for column in numeric_columns
    if column != TARGET_COLUMN
]

placement_rate_rows = []

for column in numeric_features_without_target:

    try:
        bins = pd.qcut(
            target_relationship_df[column],
            q=5,
            duplicates="drop"
        )

        grouped = (
            target_relationship_df
            .groupby(bins, observed=False)["target_numeric"]
            .agg(["count", "mean"])
            .reset_index()
        )

        grouped = grouped.rename(
            columns={
                "count": "sample_count",
                "mean": "placement_rate",
            }
        )

        grouped["feature"] = column

        placement_rate_rows.append(grouped)

    except Exception as error:
        print(
            f"Could not calculate placement rate for {column}: {error}"
        )

if placement_rate_rows:
    placement_rate_df = pd.concat(
        placement_rate_rows,
        ignore_index=True
    )

    save_dataframe(
        placement_rate_df,
        "placement_rate_by_numeric_features.csv",
        index=False
    )


# ==========================================================================================
# 16. POSSIBLE DATA LEAKAGE ANALYSIS
# ==========================================================================================

print_section("13. POSSIBLE DATA LEAKAGE ANALYSIS")

possible_leakage_keywords = [
    "salary",
    "package",
    "placed",
    "placement",
    "offer",
    "company",
    "joining",
    "selection",
    "result",
]

leakage_rows = []

for column in df.columns:

    column_lower = column.lower()

    keyword_matches = [
        keyword
        for keyword in possible_leakage_keywords
        if keyword in column_lower
    ]

    unique_count = df[column].nunique(dropna=False)

    leakage_rows.append({
        "column": column,
        "keyword_matches": ", ".join(keyword_matches),
        "unique_values": unique_count,
        "data_type": str(df[column].dtype),
        "possible_leakage": (
            len(keyword_matches) > 0
            and column != TARGET_COLUMN
        ),
    })

leakage_report = pd.DataFrame(leakage_rows)

print(leakage_report.to_string(index=False))

save_dataframe(
    leakage_report,
    "possible_leakage_report.csv",
    index=False
)

print(
    "\nImportant: This report identifies columns that require review. "
    "It does not automatically remove them."
)


# ==========================================================================================
# 17. CONSTANT AND NEAR-CONSTANT FEATURES
# ==========================================================================================

print_section("14. CONSTANT AND NEAR-CONSTANT FEATURE ANALYSIS")

constant_rows = []

for column in df.columns:

    value_counts = df[column].value_counts(
        normalize=True,
        dropna=False
    )

    top_percentage = (
        value_counts.iloc[0] * 100
        if not value_counts.empty
        else 0
    )

    constant_rows.append({
        "column": column,
        "unique_values": df[column].nunique(dropna=False),
        "most_common_percentage": top_percentage,
        "is_constant": df[column].nunique(dropna=False) <= 1,
        "is_near_constant": top_percentage >= 99,
    })

constant_report = pd.DataFrame(constant_rows)

print(constant_report.to_string(index=False))

save_dataframe(
    constant_report,
    "constant_features_report.csv",
    index=False
)


# ==========================================================================================
# 18. CORRELATION MATRIX
# ==========================================================================================

print_section("15. CORRELATION MATRIX")

numeric_df = target_relationship_df.select_dtypes(
    include=np.number
)

if numeric_df.shape[1] >= 2:

    correlation_matrix = numeric_df.corr()

    save_dataframe(
        correlation_matrix,
        "correlation_matrix.csv"
    )

    plt.figure(figsize=(14, 10))

    image = plt.imshow(
        correlation_matrix,
        aspect="auto"
    )

    plt.colorbar(image)

    plt.xticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        rotation=90
    )

    plt.yticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns
    )

    plt.title("Numerical Feature Correlation Matrix")
    plt.tight_layout()

    correlation_plot_path = PLOTS_DIR / "correlation_matrix.png"

    plt.savefig(
        correlation_plot_path,
        dpi=150
    )

    plt.close()

    print(f"Saved: {correlation_plot_path}")

else:
    print("Not enough numerical columns for a correlation matrix.")


# ==========================================================================================
# 19. OUTLIER ANALYSIS USING IQR
# ==========================================================================================

print_section("16. OUTLIER ANALYSIS")

outlier_rows = []

for column in numeric_features_without_target:

    series = df[column].dropna()

    if series.empty:
        continue

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (series < lower_bound)
        | (series > upper_bound)
    )

    outlier_count = int(outlier_mask.sum())

    outlier_rows.append({
        "column": column,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outlier_count,
        "outlier_percentage": (
            outlier_count / len(series) * 100
        ),
    })

outlier_report = pd.DataFrame(outlier_rows)

if not outlier_report.empty:
    outlier_report = outlier_report.sort_values(
        by="outlier_percentage",
        ascending=False
    )

    print(outlier_report.to_string(index=False))

    save_dataframe(
        outlier_report,
        "outlier_report.csv",
        index=False
    )


# ==========================================================================================
# 20. AUTOMATIC ANALYSIS SUMMARY
# ==========================================================================================

print_section("17. ANALYSIS SUMMARY")

summary_rows = [
    {
        "metric": "total_rows",
        "value": len(df),
    },
    {
        "metric": "total_columns",
        "value": len(df.columns),
    },
    {
        "metric": "duplicate_rows",
        "value": duplicate_count,
    },
    {
        "metric": "columns_with_missing_values",
        "value": int((df.isna().sum() > 0).sum()),
    },
    {
        "metric": "numeric_columns",
        "value": len(numeric_columns),
    },
    {
        "metric": "categorical_columns",
        "value": len(categorical_columns),
    },
    {
        "metric": "target_column",
        "value": TARGET_COLUMN,
    },
]

if len(valid_target) > 0:
    summary_rows.extend([
        {
            "metric": "placed_students",
            "value": int((valid_target == 1).sum()),
        },
        {
            "metric": "not_placed_students",
            "value": int((valid_target == 0).sum()),
        },
        {
            "metric": "placement_rate_percentage",
            "value": float((valid_target == 1).mean() * 100),
        },
    ])

summary_df = pd.DataFrame(summary_rows)

print(summary_df.to_string(index=False))

save_dataframe(
    summary_df,
    "analysis_summary.csv",
    index=False
)


# ==========================================================================================
# 21. FINAL MESSAGE
# ==========================================================================================

print_section("ANALYSIS COMPLETED")

print(f"All reports saved inside:\n{OUTPUT_DIR}")
print(f"All plots saved inside:\n{PLOTS_DIR}")

print("\nGenerated files include:")
print("  - analysis_summary.csv")
print("  - target_distribution.csv")
print("  - missing_values_report.csv")
print("  - duplicate_report.csv")
print("  - numeric_summary.csv")
print("  - categorical_summary.csv")
print("  - numeric_target_relationship.csv")
print("  - categorical_target_relationship.csv")
print("  - possible_leakage_report.csv")
print("  - constant_features_report.csv")
print("  - correlation_matrix.csv")
print("  - outlier_report.csv")
print("  - plots folder")

print("\nNext step:")
print("Review the generated reports before creating model_training_v2.py.")