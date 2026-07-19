import os
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier

os.makedirs("results", exist_ok=True)

# Load dataset
df = pd.read_csv("data/online_shoppers_intention.csv")

X = df.drop("Revenue", axis=1)
y = df["Revenue"].astype(int)

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Dummy Most Frequent": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression (unbalanced)": LogisticRegression(max_iter=1000),
    "Logistic Regression (balanced)": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree (unbalanced)": DecisionTreeClassifier(random_state=42),
    "Decision Tree (balanced)": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "Random Forest (unbalanced)": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Random Forest (balanced)": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1),
    "SVM (unbalanced)": LinearSVC(max_iter=10000, random_state=42),
    "SVM (balanced)": LinearSVC(max_iter=10000, class_weight="balanced", random_state=42),
}

results = []

for model_name, model in models.items():
    start = time.time()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    runtime = time.time() - start

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "Runtime_seconds": runtime
    })

results_df = pd.DataFrame(results)
results_df.to_csv("results/imbalance_comparison.csv", index=False)

print("========== IMBALANCE COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nSaved file: results/imbalance_comparison.csv")
