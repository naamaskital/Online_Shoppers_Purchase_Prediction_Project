import os
import time
import pandas as pd
import matplotlib.pyplot as plt

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

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# Encoder compatibility for different sklearn versions
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", encoder, categorical_features)
    ]
)

# Train-test split with stratify because the dataset is imbalanced
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

print("\nTrain target distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True) * 100)

# Models
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

# Clear old report file
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

    # ROC AUC if possible
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

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    if roc_auc is not None:
        print("ROC AUC:", round(roc_auc, 4))
    print("Runtime seconds:", round(runtime, 2))

    # Save classification report
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

    # Confusion matrix
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

# Save results table
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1", ascending=False)

print("\n========== FINAL MODEL COMPARISON ==========")
print(results_df)

results_df.to_csv("results/model_comparison.csv", index=False)

print("\nTraining completed successfully.")
print("Saved files:")
print("- results/model_comparison.csv")
print("- results/classification_reports.txt")
print("- confusion matrix images in results folder")
