"""Run the AdaBoost and PCA course extensions."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/online_shoppers_intention.csv")
RESULTS_DIR = Path("results")
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

    numeric_preprocessing = Pipeline(
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

    categorical_preprocessing = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                create_encoder(),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_preprocessing,
                numeric_features,
            ),
            (
                "cat",
                categorical_preprocessing,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def run_adaboost(X, y):
    """Train AdaBoost and save its test-set results."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(X),
            ),
            (
                "classifier",
                AdaBoostClassifier(
                    n_estimators=100,
                    learning_rate=0.5,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    start_time = time.time()

    model.fit(X_train, y_train)

    runtime = time.time() - start_time

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    result = {
        "Model": "AdaBoost",
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
        "ROC AUC": roc_auc_score(
            y_test,
            probabilities,
        ),
        "Runtime_seconds": runtime,
    }

    results_df = pd.DataFrame([result])

    results_df.to_csv(
        RESULTS_DIR / "course_extensions_adaboost_results.csv",
        index=False,
    )

    save_combined_model_comparison(results_df)

    print("========== ADABOOST RESULTS ==========")

    for metric, value in result.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")


def save_combined_model_comparison(adaboost_results):
    """Add AdaBoost to the main model-comparison table."""

    comparison_path = RESULTS_DIR / "model_comparison.csv"

    if not comparison_path.exists():
        return

    base_results = pd.read_csv(comparison_path)

    # The main comparison file uses ROC_AUC rather than ROC AUC.
    adaboost_for_comparison = adaboost_results.rename(
        columns={
            "ROC AUC": "ROC_AUC",
        }
    )

    common_columns = [
        column
        for column in base_results.columns
        if column in adaboost_for_comparison.columns
    ]

    combined_results = pd.concat(
        [
            base_results[common_columns],
            adaboost_for_comparison[common_columns],
        ],
        ignore_index=True,
    )

    combined_results.to_csv(
        RESULTS_DIR / "model_comparison_with_adaboost.csv",
        index=False,
    )


def run_pca(X, y):
    """Create a two-dimensional PCA projection of the sessions."""

    preprocessor = create_preprocessor(X)
    processed_features = preprocessor.fit_transform(X)

    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    )

    projected_features = pca.fit_transform(
        processed_features
    )

    projection_df = pd.DataFrame(
        {
            "PC1": projected_features[:, 0],
            "PC2": projected_features[:, 1],
            "Revenue": y,
        }
    )

    projection_df.to_csv(
        RESULTS_DIR / "pca_projection.csv",
        index=False,
    )

    save_pca_plot(projection_df)

    explained_variance_df = pd.DataFrame(
        {
            "Component": [
                "PC1",
                "PC2",
            ],
            "ExplainedVarianceRatio": (
                pca.explained_variance_ratio_
            ),
        }
    )

    explained_variance_df.to_csv(
        RESULTS_DIR / "pca_explained_variance.csv",
        index=False,
    )

    print("\n========== PCA EXPLAINED VARIANCE ==========")
    print(explained_variance_df.to_string(index=False))


def save_pca_plot(projection_df):
    """Save a scatter plot of the first two principal components."""

    no_purchase = projection_df[
        projection_df["Revenue"] == 0
    ]

    purchase = projection_df[
        projection_df["Revenue"] == 1
    ]

    plt.figure(figsize=(8, 6))

    plt.scatter(
        no_purchase["PC1"],
        no_purchase["PC2"],
        alpha=0.35,
        s=12,
        label="No Purchase",
    )

    plt.scatter(
        purchase["PC1"],
        purchase["PC2"],
        alpha=0.55,
        s=12,
        label="Purchase",
    )

    plt.title("PCA Projection of Online Shopper Sessions")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "pca_projection.png",
        dpi=200,
    )

    plt.close()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    X, y = load_data()

    run_adaboost(X, y)
    run_pca(X, y)

    print("\nCourse extensions completed.")
    print("Results are stored in the results directory.")


if __name__ == "__main__":
    main()
