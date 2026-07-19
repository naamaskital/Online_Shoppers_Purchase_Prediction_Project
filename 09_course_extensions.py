import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/online_shoppers_intention.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data():
    df = pd.read_csv(DATA_PATH)

    if "Revenue" not in df.columns:
        raise ValueError("Column 'Revenue' was not found in the dataset.")

    X = df.drop(columns=["Revenue"])
    y_raw = df["Revenue"]

    if y_raw.dtype == bool:
        y = y_raw.astype(int)
    else:
        y = y_raw.astype(str).str.lower().map({
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0
        })

        if y.isna().any():
            y = y_raw.astype(int)

    return X, y


def build_preprocessor():
    X, _ = load_data()

    categorical_cols = [
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend",
    ]
    numeric_cols = [
        col
        for col in X.columns
        if col not in categorical_cols
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor


def run_adaboost():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("model", AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.5,
            random_state=42
        ))
    ])

    start = time.time()
    model.fit(X_train, y_train)
    runtime = time.time() - start

    y_pred = model.predict(X_test)

    if hasattr(model.named_steps["model"], "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = np.nan

    result = {
        "Model": "AdaBoost",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc,
        "Runtime_seconds": runtime
    }

    results_df = pd.DataFrame([result])
    results_df.to_csv(RESULTS_DIR / "course_extensions_adaboost_results.csv", index=False)

    model_comparison_path = RESULTS_DIR / "model_comparison.csv"
    if model_comparison_path.exists():
        base_results = pd.read_csv(model_comparison_path)
        common_cols = [col for col in base_results.columns if col in results_df.columns]
        combined = pd.concat(
            [base_results[common_cols], results_df[common_cols]],
            ignore_index=True
        )
        combined.to_csv(RESULTS_DIR / "model_comparison_with_adaboost.csv", index=False)

    print("AdaBoost results:")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def run_pca_visualization():
    X, y = load_data()

    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_processed)

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Revenue": y
    })

    pca_df.to_csv(RESULTS_DIR / "pca_projection.csv", index=False)

    plt.figure(figsize=(8, 6))

    no_purchase = pca_df[pca_df["Revenue"] == 0]
    purchase = pca_df[pca_df["Revenue"] == 1]

    plt.scatter(no_purchase["PC1"], no_purchase["PC2"], alpha=0.35, s=12, label="No Purchase")
    plt.scatter(purchase["PC1"], purchase["PC2"], alpha=0.55, s=12, label="Purchase")

    plt.title("PCA Projection of Online Shopper Sessions")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pca_projection.png", dpi=200)
    plt.close()

    explained = pd.DataFrame({
        "Component": ["PC1", "PC2"],
        "ExplainedVarianceRatio": pca.explained_variance_ratio_
    })
    explained.to_csv(RESULTS_DIR / "pca_explained_variance.csv", index=False)

    print("\nPCA explained variance:")
    print(explained.to_string(index=False))
    print("\nSaved PCA files:")
    print(RESULTS_DIR / "pca_projection.png")
    print(RESULTS_DIR / "pca_projection.csv")
    print(RESULTS_DIR / "pca_explained_variance.csv")


def main():
    run_adaboost()
    run_pca_visualization()


if __name__ == "__main__":
    main()
