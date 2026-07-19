"""Exploratory data analysis for the online shoppers dataset."""

import os

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = "data/online_shoppers_intention.csv"
RESULTS_DIR = "results"

NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

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
    """Load the dataset from the data directory."""

    return pd.read_csv(DATA_PATH)


def print_dataset_overview(df):
    """Print basic information about the dataset and target variable."""

    print("========== DATASET OVERVIEW ==========")
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\n========== TARGET DISTRIBUTION ==========")
    print(df["Revenue"].value_counts())

    print("\nPercentages:")
    print(df["Revenue"].value_counts(normalize=True) * 100)


def create_target_distribution_plot(df):
    """Save a bar chart of purchase and non-purchase sessions."""

    plt.figure()
    df["Revenue_Label"].value_counts().plot(kind="bar")

    plt.title("Revenue Distribution")
    plt.xlabel("Revenue")
    plt.ylabel("Number of Sessions")

    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "revenue_distribution.png")
    )
    plt.close()


def analyze_numeric_features(df):
    """Compare average numerical feature values by purchase outcome."""

    print("\n========== MEAN VALUES BY REVENUE ==========")

    mean_by_revenue = (
        df.groupby("Revenue")[NUMERIC_FEATURES]
        .mean()
    )

    print(mean_by_revenue)

    mean_by_revenue.to_csv(
        os.path.join(
            RESULTS_DIR,
            "mean_features_by_revenue.csv",
        )
    )

    for feature in NUMERIC_FEATURES:
        plt.figure()

        (
            df.groupby("Revenue_Label")[feature]
            .mean()
            .plot(kind="bar")
        )

        plt.title(f"Average {feature} by Revenue")
        plt.xlabel("Revenue")
        plt.ylabel(f"Average {feature}")

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                f"avg_{feature}_by_revenue.png",
            )
        )
        plt.close()


def analyze_categorical_features(df):
    """Calculate and plot purchase rates for categorical features."""

    print(
        "\n========== CATEGORICAL FEATURES BY REVENUE =========="
    )

    for feature in CATEGORICAL_FEATURES:
        print(f"\n--- {feature} vs Revenue ---")

        purchase_table = (
            pd.crosstab(
                df[feature],
                df["Revenue"],
                normalize="index",
            )
            * 100
        )

        print(purchase_table)

        purchase_table.to_csv(
            os.path.join(
                RESULTS_DIR,
                f"{feature}_vs_revenue.csv",
            )
        )

        plt.figure()
        purchase_table[True].plot(kind="bar")

        plt.title(f"Purchase Rate by {feature}")
        plt.xlabel(feature)
        plt.ylabel("Purchase Rate (%)")

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                f"purchase_rate_by_{feature}.png",
            )
        )
        plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_data()

    df["Revenue_Label"] = df["Revenue"].map(
        {
            False: "No Purchase",
            True: "Purchase",
        }
    )

    print_dataset_overview(df)
    create_target_distribution_plot(df)
    analyze_numeric_features(df)
    analyze_categorical_features(df)

    print("\nEDA completed.")
    print("Tables and figures are stored in the results directory.")


if __name__ == "__main__":
    main()
