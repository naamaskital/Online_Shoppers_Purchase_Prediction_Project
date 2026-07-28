"""Compare KNN with PCA and SMOTE on the online shoppers dataset."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/online_shoppers_intention.csv")
RESULTS_DIR = Path("results")
TARGET_COLUMN = "Revenue"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_NEIGHBORS = 5
PCA_VARIANCE_TARGET = 0.90

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]


def load_data():
    """Load the dataset and separate features from the target."""
    dataframe = pd.read_csv(DATA_PATH)

    features = dataframe.drop(columns=[TARGET_COLUMN])
    target = dataframe[TARGET_COLUMN].astype(int)

    return features, target


def create_encoder():
    """Create a dense one-hot encoder across sklearn versions."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def create_preprocessor(features):
    """Scale numeric features and encode categorical features."""
    numeric_features = [
        column
        for column in features.columns
        if column not in CATEGORICAL_FEATURES
    ]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
            (
                "categorical",
                create_encoder(),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def create_experiments(features):
    """Create four KNN configurations for comparison."""
    return {
        "KNN baseline": Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor(features),
                ),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=N_NEIGHBORS,
                    ),
                ),
            ]
        ),
        "PCA + KNN": Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor(features),
                ),
                (
                    "pca",
                    PCA(
                        n_components=PCA_VARIANCE_TARGET,
                        svd_solver="full",
                    ),
                ),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=N_NEIGHBORS,
                    ),
                ),
            ]
        ),
        "SMOTE + KNN": Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor(features),
                ),
                (
                    "smote",
                    SMOTE(
                        random_state=RANDOM_STATE,
                    ),
                ),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=N_NEIGHBORS,
                    ),
                ),
            ]
        ),
        "PCA + SMOTE + KNN": Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor(features),
                ),
                (
                    "pca",
                    PCA(
                        n_components=PCA_VARIANCE_TARGET,
                        svd_solver="full",
                    ),
                ),
                (
                    "smote",
                    SMOTE(
                        random_state=RANDOM_STATE,
                    ),
                ),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=N_NEIGHBORS,
                    ),
                ),
            ]
        ),
    }


def safe_filename(model_name):
    """Convert a model name into a safe filename."""
    return (
        model_name.lower()
        .replace(" + ", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def save_confusion_matrix(
    model_name,
    y_test,
    predictions,
):
    """Save one confusion-matrix figure."""
    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "No Purchase",
            "Purchase",
        ],
    )

    display.plot(values_format="d")

    plt.title(
        f"Confusion Matrix - {model_name}"
    )
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / (
            "confusion_matrix_"
            f"{safe_filename(model_name)}.png"
        ),
        dpi=200,
    )

    plt.close()


def analyze_pca_variance(X_train):
    """Analyze PCA variance using training data only."""
    preprocessor = create_preprocessor(X_train)

    processed_train = preprocessor.fit_transform(
        X_train
    )

    full_pca = PCA(
        svd_solver="full",
    )

    full_pca.fit(processed_train)

    cumulative_variance = np.cumsum(
        full_pca.explained_variance_ratio_
    )

    selected_components = int(
        np.searchsorted(
            cumulative_variance,
            PCA_VARIANCE_TARGET,
        )
        + 1
    )

    variance_df = pd.DataFrame(
        {
            "Component": np.arange(
                1,
                len(cumulative_variance) + 1,
            ),
            "ExplainedVarianceRatio": (
                full_pca.explained_variance_ratio_
            ),
            "CumulativeExplainedVariance": (
                cumulative_variance
            ),
        }
    )

    variance_df.to_csv(
        RESULTS_DIR
        / "pca_knn_explained_variance.csv",
        index=False,
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        variance_df["Component"],
        variance_df[
            "CumulativeExplainedVariance"
        ],
        marker="o",
        markersize=3,
    )

    plt.axhline(
        PCA_VARIANCE_TARGET,
        linestyle="--",
        label=(
            "Variance target = "
            f"{PCA_VARIANCE_TARGET:.0%}"
        ),
    )

    plt.axvline(
        selected_components,
        linestyle="--",
        label=(
            "Selected components = "
            f"{selected_components}"
        ),
    )

    plt.xlabel(
        "Number of Principal Components"
    )
    plt.ylabel(
        "Cumulative Explained Variance"
    )
    plt.title(
        "Cumulative Explained Variance for PCA"
    )
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "pca_knn_cumulative_explained_variance.png",
        dpi=200,
    )

    plt.close()

    summary_df = pd.DataFrame(
        [
            {
                "Original_dataset_features": (
                    X_train.shape[1]
                ),
                "Features_after_preprocessing": (
                    processed_train.shape[1]
                ),
                "Variance_target": (
                    PCA_VARIANCE_TARGET
                ),
                "Selected_PCA_components": (
                    selected_components
                ),
                "Actual_cumulative_variance": (
                    cumulative_variance[
                        selected_components - 1
                    ]
                ),
            }
        ]
    )

    summary_df.to_csv(
        RESULTS_DIR
        / "pca_knn_dimension_summary.csv",
        index=False,
    )

    print(
        "\n========== PCA DIMENSION ANALYSIS =========="
    )
    print(summary_df.to_string(index=False))


def evaluate_experiment(
    model_name,
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    report_file,
):
    """Fit and evaluate one experimental pipeline."""
    print(
        f"\n========== {model_name} =========="
    )

    start_time = time.time()

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    runtime = time.time() - start_time

    result = {
        "Model": model_name,
        "Accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "ROC_AUC": roc_auc_score(
            y_test,
            probabilities,
        ),
        "Runtime_seconds": runtime,
        "PCA_components": np.nan,
    }

    if "pca" in pipeline.named_steps:
        result["PCA_components"] = int(
            pipeline.named_steps[
                "pca"
            ].n_components_
        )

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "No Purchase",
            "Purchase",
        ],
        zero_division=0,
    )

    report_file.write(
        f"{model_name}\n"
    )
    report_file.write(
        "-" * len(model_name)
    )
    report_file.write("\n")
    report_file.write(report)
    report_file.write("\n\n")

    save_confusion_matrix(
        model_name,
        y_test,
        predictions,
    )

    for metric, value in result.items():
        if isinstance(value, float):
            print(
                f"{metric}: {value:.4f}"
            )
        else:
            print(
                f"{metric}: {value}"
            )

    return result


def save_metric_comparison(results_df):
    """Save a plot comparing the evaluation metrics."""
    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
    ]

    plot_df = results_df.set_index(
        "Model"
    )[metric_columns]

    axis = plot_df.plot(
        kind="bar",
        figsize=(11, 6),
    )

    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title(
        "KNN, PCA and SMOTE Comparison"
    )
    axis.tick_params(
        axis="x",
        rotation=15,
    )
    axis.legend(
        loc="lower right",
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "pca_knn_metric_comparison.png",
        dpi=200,
    )

    plt.close()


def main():
    """Run the full PCA, SMOTE and KNN experiment."""
    RESULTS_DIR.mkdir(exist_ok=True)

    X, y = load_data()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    print(
        "Train shape:",
        X_train.shape,
    )
    print(
        "Test shape:",
        X_test.shape,
    )
    print(
        "Train purchase rate:",
        round(y_train.mean(), 4),
    )
    print(
        "Test purchase rate:",
        round(y_test.mean(), 4),
    )

    analyze_pca_variance(X_train)

    report_path = (
        RESULTS_DIR
        / "pca_knn_classification_reports.txt"
    )

    results = []

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report_file:
        report_file.write(
            "PCA, SMOTE and KNN "
            "Classification Reports\n"
        )
        report_file.write(
            "===================================="
            "======\n\n"
        )

        experiments = create_experiments(X)

        for model_name, pipeline in (
            experiments.items()
        ):
            result = evaluate_experiment(
                model_name,
                pipeline,
                X_train,
                X_test,
                y_train,
                y_test,
                report_file,
            )

            results.append(result)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_DIR
        / "pca_knn_comparison.csv",
        index=False,
    )

    save_metric_comparison(results_df)

    print(
        "\n========== FINAL PCA-KNN COMPARISON =========="
    )
    print(
        results_df.to_string(index=False)
    )

    print("\nExperiment completed.")
    print(
        "Results were saved in "
        "the results directory."
    )


if __name__ == "__main__":
    main()
