import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier


# ---------------------------------------------------------
# 1. Create folders
# ---------------------------------------------------------
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------
# 2. Load / download dataset
# ---------------------------------------------------------
data_path = "data/online_shoppers_intention.csv"

if not os.path.exists(data_path):
    dataset = fetch_ucirepo(id=468)
    X_uci = dataset.data.features
    y_uci = dataset.data.targets
    df = pd.concat([X_uci, y_uci], axis=1)
    df.to_csv(data_path, index=False)
    print("Dataset downloaded successfully.")
else:
    df = pd.read_csv(data_path)
    print("Dataset loaded from local file.")

print("\n========== BASIC INFORMATION ==========")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["Revenue"].value_counts())
print("\nTarget percentages:")
print(df["Revenue"].value_counts(normalize=True) * 100)


# ---------------------------------------------------------
# 3. Exploratory Data Analysis
# ---------------------------------------------------------
target_counts = df["Revenue"].value_counts()

plt.figure()
target_counts.plot(kind="bar")
plt.title("Revenue Distribution")
plt.xlabel("Revenue")
plt.ylabel("Number of Sessions")
plt.tight_layout()
plt.savefig("results/revenue_distribution.png")
plt.close()

important_numeric_features = [
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

mean_by_revenue = df.groupby("Revenue")[important_numeric_features].mean()
mean_by_revenue.to_csv("results/mean_features_by_revenue.csv")

print("\n========== MEAN FEATURES BY REVENUE ==========")
print(mean_by_revenue)


# ---------------------------------------------------------
# 4. Split features and target
# ---------------------------------------------------------
X = df.drop("Revenue", axis=1)
y = df["Revenue"].astype(int)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)


# ---------------------------------------------------------
# 5. Preprocessing
# ---------------------------------------------------------
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", encoder, categorical_features)
    ]
)


# ---------------------------------------------------------
# 6. Train-test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)


# ---------------------------------------------------------
# 7. Train and compare models
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
    "SVM": LinearSVC(max_iter=10000, class_weight="balanced", random_state=42)
}

results = []

with open("results/classification_reports.txt", "w", encoding="utf-8") as f:
    f.write("Classification Reports\n")
    f.write("======================\n\n")

for model_name, model in models.items():
    print(f"\n========== Training {model_name} ==========")
    start_time = time.time()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    runtime = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    roc_auc = None
    try:
        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_score)
        elif hasattr(pipeline, "decision_function"):
            y_score = pipeline.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, y_score)
    except Exception:
        roc_auc = None

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "Runtime_seconds": runtime
    })

    report = classification_report(
        y_test,
        y_pred,
        target_names=["No Purchase", "Purchase"],
        zero_division=0
    )

    with open("results/classification_reports.txt", "a", encoding="utf-8") as f:
        f.write(f"{model_name}\n")
        f.write("-" * len(model_name) + "\n")
        f.write(report)
        f.write("\n\n")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Purchase", "Purchase"]
    )
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(f"results/confusion_matrix_{safe_name}.png")
    plt.close()

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1", ascending=False)
results_df.to_csv("results/model_comparison.csv", index=False)

print("\n========== FINAL MODEL COMPARISON ==========")
print(results_df.to_string(index=False))


# ---------------------------------------------------------
# 8. Class imbalance comparison
# ---------------------------------------------------------
imbalance_models = {
    "Dummy Most Frequent": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression (unbalanced)": LogisticRegression(max_iter=1000),
    "Logistic Regression (balanced)": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree (unbalanced)": DecisionTreeClassifier(random_state=42),
    "Decision Tree (balanced)": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "Random Forest (unbalanced)": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Random Forest (balanced)": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
    "SVM (unbalanced)": LinearSVC(max_iter=10000, random_state=42),
    "SVM (balanced)": LinearSVC(max_iter=10000, class_weight="balanced", random_state=42),
}

imbalance_results = []

for model_name, model in imbalance_models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    imbalance_results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0)
    })

imbalance_df = pd.DataFrame(imbalance_results)
imbalance_df.to_csv("results/imbalance_comparison.csv", index=False)

print("\n========== IMBALANCE COMPARISON ==========")
print(imbalance_df.to_string(index=False))


# ---------------------------------------------------------
# 9. Feature importance using Random Forest
# ---------------------------------------------------------
rf_pipeline = Pipeline(
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

rf_pipeline.fit(X_train, y_train)

feature_names = rf_pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = rf_pipeline.named_steps["classifier"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df["Feature"] = importance_df["Feature"].str.replace("num__", "", regex=False)
importance_df["Feature"] = importance_df["Feature"].str.replace("cat__", "", regex=False)

importance_df = importance_df.sort_values(by="Importance", ascending=False)
importance_df.to_csv("results/random_forest_feature_importance.csv", index=False)

print("\n========== TOP 20 FEATURES ==========")
print(importance_df.head(20).to_string(index=False))

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


# ---------------------------------------------------------
# 10. Extra summary plots
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.bar(results_df["Model"], results_df["F1"])
plt.title("Model Comparison by F1 Score")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("results/model_comparison_f1.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(results_df["Model"], results_df["ROC_AUC"])
plt.title("Model Comparison by ROC AUC")
plt.xlabel("Model")
plt.ylabel("ROC AUC")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("results/model_comparison_roc_auc.png")
plt.close()

plt.figure(figsize=(12, 6))
plt.bar(imbalance_df["Model"], imbalance_df["Recall"])
plt.title("Effect of Class Imbalance on Recall")
plt.xlabel("Model")
plt.ylabel("Recall")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("results/imbalance_comparison_recall.png")
plt.close()


print("\nProject code completed successfully.")
print("Main result files were saved in the results folder.")
