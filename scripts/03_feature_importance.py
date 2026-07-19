"""Calculate feature importance for the Random Forest model."""

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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


def create_random_forest_pipeline(X):
    """Create the preprocessing and Random Forest pipeline."""

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


def get_feature_importance(model):
    """Return a sorted table of Random Forest feature importances."""

    feature_names = (
        model.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importance_values = (
        model.named_steps["classifier"]
        .feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance_values,
        }
    )

    importance_df["Feature"] = (
        importance_df["Feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )

    return importance_df.sort_values(
        by="Importance",
        ascending=False,
    )


def save_feature_importance_plot(importance_df):
    """Save a chart of the 15 most important features."""

    top_features = importance_df.head(15)

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"],
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 15 Feature Importances - Random Forest")

    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "random_forest_top_features.png",
        )
    )

    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X, y = load_data()

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = create_random_forest_pipeline(X)
    model.fit(X_train, y_train)

    importance_df = get_feature_importance(model)

    importance_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "random_forest_feature_importance.csv",
        ),
        index=False,
    )

    save_feature_importance_plot(importance_df)

    print("========== TOP 20 FEATURES ==========")
    print(importance_df.head(20).to_string(index=False))

    print("\nFeature importance analysis completed.")
    print(
        "The table and chart are stored "
        "in the results directory."
    )


if __name__ == "__main__":
    main()
