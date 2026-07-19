import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Create results folder
os.makedirs("results", exist_ok=True)

# Load dataset
df = pd.read_csv("data/online_shoppers_intention.csv")

# Split features and target
X = df.drop("Revenue", axis=1)
y = df["Revenue"].astype(int)

# Identify numeric and categorical columns
categorical_features = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

numeric_features = [
    column
    for column in X.columns
    if column not in categorical_features
]

# Encoder compatibility
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", encoder, categorical_features)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Random Forest model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ]
)

# Train model
model.fit(X_train, y_train)

# Get feature names after preprocessing
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

# Get feature importance
importances = model.named_steps["classifier"].feature_importances_

# Create dataframe
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Clean feature names
importance_df["Feature"] = importance_df["Feature"].str.replace("num__", "", regex=False)
importance_df["Feature"] = importance_df["Feature"].str.replace("cat__", "", regex=False)

# Sort by importance
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Save full feature importance table
importance_df.to_csv("results/random_forest_feature_importance.csv", index=False)

print("========== TOP 20 FEATURES ==========")
print(importance_df.head(20).to_string(index=False))

# Plot top 15 features
top_features = importance_df.head(15)

plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Feature Importances - Random Forest")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("results/random_forest_top_features.png")
plt.close()

print("\nFeature importance analysis completed successfully.")
print("Saved files:")
print("- results/random_forest_feature_importance.csv")
print("- results/random_forest_top_features.png")
