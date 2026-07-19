"""Tune the Random Forest model and test different decision thresholds."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "data/online_shoppers_intention.csv"
RESULTS_DIR = "results"
TARGET_COLUMN = "Revenue"
RANDOM_STATE = 42

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

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    return X, y


def create_encoder():
    """Create a one-hot encoder compatible with sklearn versions."""

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
    """Create preprocessing for numerical and categorical features."""

    numeric_features = [
        column
        for column in X.columns
        if column not in CATEGORICAL_FEATURES
    ]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features,
            ),
            (
                "cat",
                create_encoder(),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def split_data(X, y):
    """
    Create separate training, validation, and test sets.

    Final proportions:
    - 60% training
    - 20% validation
    - 20% test
    """

    X_development, X_test, y_development, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X_development,
            y_development,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y_development,
        )
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


def create_baseline_model(X):
    """Create the original balanced Random Forest model."""

    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(X),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def tune_random_forest(X_train, y_train):
    """Search for Random Forest parameters using F1 score."""

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(X_train),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    parameter_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 5],
        "classifier__max_features": ["sqrt", "log2"],
        "classifier__class_weight": [
            "balanced",
            "balanced_subsample",
        ],
    }

    cross_validation = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=cross_validation,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    return search


def evaluate_thresholds(model, X_validation, y_validation):
    """Find the threshold with the highest validation F1 score."""

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    thresholds = np.arange(
        0.20,
        0.81,
        0.01,
    )

    results = []

    for threshold in thresholds:
        predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        results.append(
            {
                "Threshold": threshold,
                "Precision": precision_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                ),
                "F1": f1_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                ),
            }
        )

    threshold_df = pd.DataFrame(results)

    best_row = (
        threshold_df
        .sort_values(
            by="F1",
            ascending=False,
        )
        .iloc[0]
    )

    best_threshold = best_row["Threshold"]

    return threshold_df, best_threshold


def calculate_metrics(
    model_name,
    y_true,
    predictions,
    probabilities,
):
    """Return the evaluation metrics for one model configuration."""

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
    }


def save_threshold_plot(threshold_df, best_threshold):
    """Save the validation threshold comparison plot."""

    plt.figure(figsize=(10, 5))

    plt.plot(
        threshold_df["Threshold"],
        threshold_df["F1"],
        label="F1",
    )
    plt.plot(
        threshold_df["Threshold"],
        threshold_df["Precision"],
        label="Precision",
    )
    plt.plot(
        threshold_df["Threshold"],
        threshold_df["Recall"],
        label="Recall",
    )

    plt.axvline(
        best_threshold,
        linestyle="--",
        label=f"Best threshold = {best_threshold:.2f}",
    )

    plt.title("Threshold Tuning on Validation Set")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "threshold_tuning_plot.png",
        )
    )

    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X, y = load_data()

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_data(X, y)

    print("Training size:", X_train.shape)
    print("Validation size:", X_validation.shape)
    print("Test size:", X_test.shape)

    baseline_model = create_baseline_model(X_train)
    baseline_model.fit(X_train, y_train)

    baseline_predictions = baseline_model.predict(X_test)
    baseline_probabilities = baseline_model.predict_proba(
        X_test
    )[:, 1]

    print("\n========== GRID SEARCH ==========")

    grid_search = tune_random_forest(
        X_train,
        y_train,
    )

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print("\nBest cross-validation F1:")
    print(grid_search.best_score_)

    tuned_model = grid_search.best_estimator_

    threshold_df, best_threshold = evaluate_thresholds(
        tuned_model,
        X_validation,
        y_validation,
    )

    threshold_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "threshold_tuning_validation.csv",
        ),
        index=False,
    )

    print("\nBest validation threshold:")
    print(round(best_threshold, 2))

    test_probabilities = tuned_model.predict_proba(
        X_test
    )[:, 1]

    default_predictions = (
        test_probabilities >= 0.50
    ).astype(int)

    tuned_threshold_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    results = [
        calculate_metrics(
            "Baseline Random Forest",
            y_test,
            baseline_predictions,
            baseline_probabilities,
        ),
        calculate_metrics(
            "Tuned Random Forest - threshold 0.5",
            y_test,
            default_predictions,
            test_probabilities,
        ),
        calculate_metrics(
            (
                "Tuned Random Forest - threshold "
                f"{best_threshold:.2f}"
            ),
            y_test,
            tuned_threshold_predictions,
            test_probabilities,
        ),
    ]

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "improved_random_forest_results.csv",
        ),
        index=False,
    )

    save_threshold_plot(
        threshold_df,
        best_threshold,
    )

    print("\n========== RANDOM FOREST RESULTS ==========")
    print(results_df.to_string(index=False))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            tuned_threshold_predictions,
            target_names=[
                "No Purchase",
                "Purchase",
            ],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            tuned_threshold_predictions,
        )
    )

    print("\nSaved files:")
    print("- results/improved_random_forest_results.csv")
    print("- results/threshold_tuning_validation.csv")
    print("- results/threshold_tuning_plot.png")


if __name__ == "__main__":
    main()
