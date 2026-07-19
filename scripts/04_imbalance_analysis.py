"""Compare balanced and unbalanced classification models."""

import os
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = "data/online_shoppers_intention.csv"
RESULTS_PATH = "results/imbalance_comparison.csv"
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
    """Define preprocessing for numerical and categorical features."""

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


def create_models():
    """Return the models used in the imbalance experiment."""

    return {
        "Dummy Most Frequent": DummyClassifier(
            strategy="most_frequent",
        ),
        "Logistic Regression (unbalanced)": LogisticRegression(
            max_iter=1000,
        ),
        "Logistic Regression (balanced)": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "Decision Tree (unbalanced)": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "Decision Tree (balanced)": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Random Forest (unbalanced)": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "SVM (unbalanced)": LinearSVC(
            max_iter=10000,
            random_state=RANDOM_STATE,
        ),
        "SVM (balanced)": LinearSVC(
            max_iter=10000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_model(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """Train one model and return its evaluation metrics."""

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(X_train),
            ),
            (
                "classifier",
                model,
            ),
        ]
    )

    start_time = time.time()

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

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
        "Runtime_seconds": runtime,
    }

    return result


def main():
    os.makedirs("results", exist_ok=True)

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = []

    for model_name, model in create_models().items():
        print(f"Evaluating: {model_name}")

        result = evaluate_model(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print("\n========== IMBALANCE COMPARISON ==========")
    print(results_df.to_string(index=False))

    print(f"\nSaved file: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
