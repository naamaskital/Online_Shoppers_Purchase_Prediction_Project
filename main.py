import os
import pandas as pd
import matplotlib.pyplot as plt

# Create results folder
os.makedirs("results", exist_ok=True)

# Load dataset
df = pd.read_csv("data/online_shoppers_intention.csv")

print("========== BASIC INFORMATION ==========")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\n========== TARGET DISTRIBUTION ==========")
target_counts = df["Revenue"].value_counts()
target_percentages = df["Revenue"].value_counts(normalize=True) * 100

print(target_counts)
print("\nPercentages:")
print(target_percentages)

# Plot target distribution
target_counts.plot(kind="bar")
plt.title("Revenue Distribution")
plt.xlabel("Revenue")
plt.ylabel("Number of Sessions")
plt.tight_layout()
plt.savefig("results/revenue_distribution.png")
plt.show()

print("\n========== NUMERIC FEATURES SUMMARY ==========")
print(df.describe())

# Numeric columns
numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Plot histograms for important numeric columns
important_numeric_columns = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay"
]

for column in important_numeric_columns:
    plt.figure()
    df[column].plot(kind="hist", bins=30)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"results/{column}_distribution.png")
    plt.show()

print("\n========== CATEGORICAL FEATURES ==========")
categorical_columns = df.select_dtypes(include=["object", "bool"]).columns.tolist()

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts())

print("\nEDA completed successfully.")