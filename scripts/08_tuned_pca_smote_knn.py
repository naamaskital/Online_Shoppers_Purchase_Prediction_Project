"""Tune a PCA + SMOTE + KNN pipeline using cross-validation."""

import json
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/online_shoppers_intention.csv")
RESULTS_DIR = Path("results")
TARGET_COLUMN = "Revenue"

RANDOM_STATE = 42
TEST_SIZE = 0.20

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

    X = dataframe.drop(columns=[TARGET_COLUMN])
    y = dataframe[TARGET_COLUMN].astype(int)

    return X, y


def create_encoder():
    """Create a dense OneHotEncoder across sklearn versions."""
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


def create_preprocessor(X):
    """Scale numerical features and encode categorical features."""
    numerical_features = [
        column
        for column in X.columns
        if column not in CATEGORICAL_FEATURES
    ]

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                numerical_features,
            ),
            (
                "categorical",
                create_encoder(),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def create_pipeline(X):
    """Create the PCA, SMOTE and KNN pipeline."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(X),
            ),
            (
                "pca",
                PCA(
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
                KNeighborsClassifier(),
            ),
        ]
    )


def calculate_metrics(
    model_name,
    y_true,
    predictions,
    probabilities,
    runtime_seconds,
    pca_components,
    pca_variance,
    parameters,
):
    """Calculate test-set evaluation metrics."""
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "ROC_AUC": roc_auc_score(
            y_true,
            probabilities,
        ),
        "Runtime_seconds": runtime_seconds,
        "PCA_components": pca_components,
        "PCA_explained_variance": pca_variance,
        "PCA_variance_target": parameters[
            "pca__n_components"
        ],
        "N_neighbors": parameters[
            "classifier__n_neighbors"
        ],
        "Weights": parameters[
            "classifier__weights"
        ],
    }


def evaluate_pipeline(
    model_name,
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    parameters,
):
    """Fit and evaluate one pipeline on the untouched test set."""
    start_time = time.time()

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    runtime_seconds = time.time() - start_time

    pca = pipeline.named_steps["pca"]

    pca_components = int(
        pca.n_components_
    )

    pca_variance = float(
        np.sum(
            pca.explained_variance_ratio_
        )
    )

    metrics = calculate_metrics(
        model_name=model_name,
        y_true=y_test,
        predictions=predictions,
        probabilities=probabilities,
        runtime_seconds=runtime_seconds,
        pca_components=pca_components,
        pca_variance=pca_variance,
        parameters=parameters,
    )

    return metrics, predictions


def save_confusion_matrix(
    model_name,
    y_test,
    predictions,
    filename,
):
    """Save a confusion matrix."""
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

    display.plot(
        values_format="d",
    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / filename,
        dpi=200,
    )

    plt.close()


def save_metric_comparison(results_df):
    """Save a bar chart comparing untuned and tuned models."""
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
        figsize=(10, 6),
    )

    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title(
        "Untuned vs Tuned PCA + SMOTE + KNN"
    )
    axis.tick_params(
        axis="x",
        rotation=0,
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "tuned_pca_smote_knn_metric_comparison.png",
        dpi=200,
    )

    plt.close()


def prepare_cv_results(grid_search):
    """Create a readable cross-validation results table."""
    cv_results = pd.DataFrame(
        grid_search.cv_results_
    )

    selected_columns = [
        "rank_test_F1",
        "mean_test_F1",
        "std_test_F1",
        "mean_test_Precision",
        "mean_test_Recall",
        "mean_test_ROC_AUC",
        "mean_test_Accuracy",
        "mean_train_F1",
        "param_pca__n_components",
        "param_classifier__n_neighbors",
        "param_classifier__weights",
        "mean_fit_time",
    ]

    readable_results = cv_results[
        selected_columns
    ].copy()

    readable_results = readable_results.sort_values(
        by=[
            "rank_test_F1",
            "mean_test_ROC_AUC",
        ],
        ascending=[
            True,
            False,
        ],
    )

    return readable_results


def main():
    """Run hyperparameter tuning and final test evaluation."""
    RESULTS_DIR.mkdir(
        exist_ok=True
    )

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
        round(
            y_train.mean(),
            4,
        ),
    )
    print(
        "Test purchase rate:",
        round(
            y_test.mean(),
            4,
        ),
    )

    print(
        "\n========== GRID SEARCH CONFIGURATION =========="
    )

    parameter_grid = {
        "pca__n_components": [
            0.85,
            0.90,
            0.95,
        ],
        "classifier__n_neighbors": [
            3,
            5,
            7,
            9,
            11,
            15,
        ],
        "classifier__weights": [
            "uniform",
            "distance",
        ],
    }

    total_combinations = (
        len(
            parameter_grid[
                "pca__n_components"
            ]
        )
        * len(
            parameter_grid[
                "classifier__n_neighbors"
            ]
        )
        * len(
            parameter_grid[
                "classifier__weights"
            ]
        )
    )

    print(
        "Parameter combinations:",
        total_combinations,
    )
    print(
        "Cross-validation folds:",
        5,
    )
    print(
        "Total model fits:",
        total_combinations * 5,
    )
    print(
        "Primary selection metric: F1"
    )

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1": "f1",
        "ROC_AUC": "roc_auc",
    }

    pipeline = create_pipeline(X)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring=scoring,
        refit="F1",
        cv=cross_validation,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
        error_score="raise",
    )

    grid_start_time = time.time()

    grid_search.fit(
        X_train,
        y_train,
    )

    grid_runtime = (
        time.time() - grid_start_time
    )

    print(
        "\n========== BEST CROSS-VALIDATION RESULT =========="
    )
    print(
        "Best parameters:",
        grid_search.best_params_,
    )
    print(
        "Best mean CV F1:",
        round(
            grid_search.best_score_,
            4,
        ),
    )
    print(
        "Grid-search runtime:",
        round(
            grid_runtime,
            4,
        ),
        "seconds",
    )

    cv_results = prepare_cv_results(
        grid_search
    )

    cv_results.to_csv(
        RESULTS_DIR
        / "tuned_pca_smote_knn_cv_results.csv",
        index=False,
    )

    cv_results.head(10).to_csv(
        RESULTS_DIR
        / "tuned_pca_smote_knn_top10_cv_results.csv",
        index=False,
    )

    untuned_parameters = {
        "pca__n_components": 0.90,
        "classifier__n_neighbors": 5,
        "classifier__weights": "uniform",
    }

    untuned_pipeline = create_pipeline(X)
    untuned_pipeline.set_params(
        **untuned_parameters
    )

    untuned_metrics, untuned_predictions = (
        evaluate_pipeline(
            model_name=(
                "Untuned PCA + SMOTE + KNN"
            ),
            pipeline=untuned_pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            parameters=untuned_parameters,
        )
    )

    tuned_pipeline = grid_search.best_estimator_

    test_start_time = time.time()

    tuned_predictions = (
        tuned_pipeline.predict(X_test)
    )

    tuned_probabilities = (
        tuned_pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    tuned_test_runtime = (
        time.time() - test_start_time
    )

    tuned_pca = (
        tuned_pipeline.named_steps["pca"]
    )

    tuned_metrics = calculate_metrics(
        model_name=(
            "Tuned PCA + SMOTE + KNN"
        ),
        y_true=y_test,
        predictions=tuned_predictions,
        probabilities=tuned_probabilities,
        runtime_seconds=tuned_test_runtime,
        pca_components=int(
            tuned_pca.n_components_
        ),
        pca_variance=float(
            np.sum(
                tuned_pca.explained_variance_ratio_
            )
        ),
        parameters=grid_search.best_params_,
    )

    results_df = pd.DataFrame(
        [
            untuned_metrics,
            tuned_metrics,
        ]
    )

    results_df.to_csv(
        RESULTS_DIR
        / "tuned_pca_smote_knn_results.csv",
        index=False,
    )

    save_confusion_matrix(
        model_name=(
            "Untuned PCA + SMOTE + KNN"
        ),
        y_test=y_test,
        predictions=untuned_predictions,
        filename=(
            "untuned_pca_smote_knn_"
            "confusion_matrix.png"
        ),
    )

    save_confusion_matrix(
        model_name=(
            "Tuned PCA + SMOTE + KNN"
        ),
        y_test=y_test,
        predictions=tuned_predictions,
        filename=(
            "tuned_pca_smote_knn_"
            "confusion_matrix.png"
        ),
    )

    save_metric_comparison(
        results_df
    )

    best_parameters_output = {
        "best_parameters": (
            grid_search.best_params_
        ),
        "best_cross_validation_f1": float(
            grid_search.best_score_
        ),
        "grid_search_runtime_seconds": (
            grid_runtime
        ),
        "selected_pca_components": int(
            tuned_pca.n_components_
        ),
        "selected_pca_explained_variance": float(
            np.sum(
                tuned_pca.explained_variance_ratio_
            )
        ),
    }

    with (
        RESULTS_DIR
        / "tuned_pca_smote_knn_best_parameters.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            best_parameters_output,
            output_file,
            indent=4,
        )

    print(
        "\n========== TEST-SET COMPARISON =========="
    )
    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\n========== TOP 10 CV CONFIGURATIONS =========="
    )
    print(
        cv_results.head(10).to_string(
            index=False
        )
    )

    print(
        "\nExperiment completed."
    )
    print(
        "Results were saved in "
        "the results directory."
    )


if __name__ == "__main__":
    main()
