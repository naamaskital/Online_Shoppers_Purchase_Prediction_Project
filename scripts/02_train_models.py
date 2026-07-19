"""Train and compare classification models."""

import os
import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


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

    print("Numeric features:", numeric_features)
    print("Categorical features:", CATEGORICAL_FEATURES)

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


def create_models():
    """Return the classification models used in the comparison."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "SVM": LinearSVC(
            max_iter=10000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def calculate_roc_auc(pipeline, X_test, y_test):
    """Calculate ROC AUC using probabilities or decision scores."""

    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, scores)

    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X_test)
        return roc_auc_score(y_test, scores)

    return None


def save_classification_report(
    model_name,
    y_test,
    predictions,
):
    """Append one model's classification report to the report file."""

    report = classification_report(
        y_test,
        predictions,
        target_names=["No Purchase", "Purchase"],
        zero_division=0,
    )

    report_path = os.path.join(
        RESULTS_DIR,
        "classification_reports.txt",
    )

    with open(report_path, "a", encoding="utf-8") as file:
        file.write(f"{model_name}\n")
        file.write("-" * len(model_name))
        file.write("\n")
        file.write(report)
        file.write("\n\n")


def save_confusion_matrix(
    model_name,
    y_test,
    predictions,
):
    """Save the confusion matrix of one model."""

    matrix = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No Purchase", "Purchase"],
    )

    display.plot(values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"confusion_matrix_{safe_name}.png",
        )
    )
    plt.close()


def evaluate_model(
    model_name,
    model,
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """Train one model and return its evaluation results."""

    print(f"\n========== Training {model_name} ==========")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    start_time = time.time()

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    runtime = time.time() - start_time

    accuracy = accuracy_score(y_test, predictions)
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
    roc_auc = calculate_roc_auc(
        pipeline,
        X_test,
        y_test,
    )

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))

    if roc_auc is not None:
        print("ROC AUC:", round(roc_auc, 4))

    print("Runtime seconds:", round(runtime, 2))

    save_classification_report(
        model_name,
        y_test,
        predictions,
    )

    save_confusion_matrix(
        model_name,
        y_test,
        predictions,
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "Runtime_seconds": runtime,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X, y = load_data()
    preprocessor = create_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTrain size:", X_train.shape)
    print("Test size:", X_test.shape)

    print("\nTrain target distribution:")
    print(y_train.value_counts(normalize=True) * 100)

    print("\nTest target distribution:")
    print(y_test.value_counts(normalize=True) * 100)

    report_path = os.path.join(
        RESULTS_DIR,
        "classification_reports.txt",
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("Classification Reports\n")
        file.write("======================\n\n")

    results = []

    for model_name, model in create_models().items():
        model_result = evaluate_model(
            model_name,
            model,
            preprocessor,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        results.append(model_result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="F1",
        ascending=False,
    )

    results_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "model_comparison.csv",
        ),
        index=False,
    )

    print("\n========== FINAL MODEL COMPARISON ==========")
    print(results_df)

    print("\nTraining completed.")
    print("Results were saved in the results directory.")


if __name__ == "__main__":
    main()
