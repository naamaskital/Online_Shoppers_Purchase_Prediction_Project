import os
import pandas as pd
import matplotlib.pyplot as plt

# Create folders
os.makedirs("results", exist_ok=True)

# Load dataset
df = pd.read_csv("data/online_shoppers_intention.csv")

print("========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\n========== TARGET DISTRIBUTION ==========")
print(df["Revenue"].value_counts())
print("\nPercentages:")
print(df["Revenue"].value_counts(normalize=True) * 100)

# Add readable labels
df["Revenue_Label"] = df["Revenue"].map({False: "No Purchase", True: "Purchase"})

# Target distribution plot
plt.figure()
df["Revenue_Label"].value_counts().plot(kind="bar")
plt.title("Revenue Distribution")
plt.xlabel("Revenue")
plt.ylabel("Number of Sessions")
plt.tight_layout()
plt.savefig("results/revenue_distribution.png")
plt.close()

# Important numeric features
important_features = [
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

print("\n========== MEAN VALUES BY REVENUE ==========")
mean_by_revenue = df.groupby("Revenue")[important_features].mean()
print(mean_by_revenue)

mean_by_revenue.to_csv("results/mean_features_by_revenue.csv")

# Bar plots for numeric features
for column in important_features:
    plt.figure()
    df.groupby("Revenue_Label")[column].mean().plot(kind="bar")
    plt.title(f"Average {column} by Revenue")
    plt.xlabel("Revenue")
    plt.ylabel(f"Average {column}")
    plt.tight_layout()
    plt.savefig(f"results/avg_{column}_by_revenue.png")
    plt.close()

# Categorical features
categorical_features = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

print("\n========== CATEGORICAL FEATURES BY REVENUE ==========")

for column in categorical_features:
    print(f"\n--- {column} vs Revenue ---")
    crosstab = pd.crosstab(df[column], df["Revenue"], normalize="index") * 100
    print(crosstab)

    crosstab.to_csv(f"results/{column}_vs_revenue.csv")

    plt.figure()
    crosstab[True].plot(kind="bar")
    plt.title(f"Purchase Rate by {column}")
    plt.xlabel(column)
    plt.ylabel("Purchase Rate (%)")
    plt.tight_layout()
    plt.savefig(f"results/purchase_rate_by_{column}.png")
    plt.close()

print("\nEDA completed successfully.")
print("Files were saved in the results folder.")
