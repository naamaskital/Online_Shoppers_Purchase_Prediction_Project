import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

os.makedirs("results", exist_ok=True)
os.makedirs("report", exist_ok=True)

# Load dataset
df = pd.read_csv("data/online_shoppers_intention.csv")

X = df.drop("Revenue", axis=1)
y = df["Revenue"].astype(int)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

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

# Final test set: untouched until final evaluation
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Validation split for threshold tuning
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=42,
    stratify=y_train_full
)

print("Train size:", X_train.shape)
print("Validation size:", X_val.shape)
print("Test size:", X_test.shape)

# Baseline Random Forest
baseline_rf = Pipeline(
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

baseline_rf.fit(X_train, y_train)
baseline_pred = baseline_rf.predict(X_test)
baseline_proba = baseline_rf.predict_proba(X_test)[:, 1]

# Hyperparameter tuning
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))
    ]
)

param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_leaf": [1, 2, 5],
    "classifier__max_features": ["sqrt", "log2"],
    "classifier__class_weight": ["balanced", "balanced_subsample"]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

print("\n========== STARTING GRID SEARCH ==========")
grid_search.fit(X_train, y_train)

print("\nBest parameters:")
print(grid_search.best_params_)
print("Best CV F1:", grid_search.best_score_)

best_model = grid_search.best_estimator_

# Default threshold = 0.5
test_proba_default = best_model.predict_proba(X_test)[:, 1]
test_pred_default = (test_proba_default >= 0.5).astype(int)

# Threshold tuning on validation set
val_proba = best_model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.20, 0.81, 0.01)
threshold_results = []

for threshold in thresholds:
    val_pred = (val_proba >= threshold).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(y_val, val_pred, zero_division=0),
        "Recall": recall_score(y_val, val_pred, zero_division=0),
        "F1": f1_score(y_val, val_pred, zero_division=0)
    })

threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_csv("results/threshold_tuning_validation.csv", index=False)

best_threshold_row = threshold_df.sort_values(by="F1", ascending=False).iloc[0]
best_threshold = best_threshold_row["Threshold"]

print("\nBest threshold based on validation F1:")
print(best_threshold_row.to_string())

# Evaluate tuned model with tuned threshold on test set
test_pred_tuned_threshold = (test_proba_default >= best_threshold).astype(int)

def evaluate_model(name, y_true, y_pred, y_proba):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_proba)
    }

improvement_results = [
    evaluate_model(
        "Baseline Random Forest",
        y_test,
        baseline_pred,
        baseline_proba
    ),
    evaluate_model(
        "Tuned Random Forest - threshold 0.5",
        y_test,
        test_pred_default,
        test_proba_default
    ),
    evaluate_model(
        f"Tuned Random Forest - threshold {best_threshold:.2f}",
        y_test,
        test_pred_tuned_threshold,
        test_proba_default
    )
]

improvement_df = pd.DataFrame(improvement_results)
improvement_df.to_csv("results/improved_random_forest_results.csv", index=False)

print("\n========== IMPROVED RANDOM FOREST RESULTS ==========")
print(improvement_df.to_string(index=False))

print("\nClassification report for final improved model:")
print(classification_report(
    y_test,
    test_pred_tuned_threshold,
    target_names=["No Purchase", "Purchase"],
    zero_division=0
))

print("\nConfusion matrix for final improved model:")
print(confusion_matrix(y_test, test_pred_tuned_threshold))

# Plot threshold tuning
plt.figure(figsize=(10, 5))
plt.plot(threshold_df["Threshold"], threshold_df["F1"], label="F1")
plt.plot(threshold_df["Threshold"], threshold_df["Precision"], label="Precision")
plt.plot(threshold_df["Threshold"], threshold_df["Recall"], label="Recall")
plt.axvline(best_threshold, linestyle="--", label=f"Best threshold = {best_threshold:.2f}")
plt.title("Threshold Tuning on Validation Set")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig("results/threshold_tuning_plot.png")
plt.close()

# Save explanation for report
with open("report/model_improvement_section_hebrew.md", "w", encoding="utf-8") as f:
    f.write("## שיפור המודל המוביל\n\n")
    f.write(
        "לאחר השוואת המודלים הראשונית, Random Forest נמצא כמודל המוביל לפי מדד F1. "
        "כדי לשפר את המודל, בוצע Hyperparameter Tuning באמצעות GridSearchCV עם Stratified K-Fold Cross Validation. "
        "החיפוש בוצע לפי מדד F1, משום שהדאטה אינו מאוזן ויש חשיבות לאיזון בין Precision ו-Recall.\n\n"
    )

    f.write("הפרמטרים הטובים ביותר שנמצאו הם:\n\n")
    for key, value in grid_search.best_params_.items():
        f.write(f"- {key}: {value}\n")

    f.write(f"\nציון ה-F1 הממוצע הטוב ביותר ב-Cross Validation היה {grid_search.best_score_:.4f}.\n\n")

    f.write(
        "בנוסף, בוצע Threshold Tuning על סט Validation. במקום להשתמש תמיד בסף ברירת המחדל 0.5, "
        "נבדקו ספים שונים, ונבחר הסף שנתן את ערך ה-F1 הגבוה ביותר על סט ה-Validation.\n\n"
    )

    f.write(f"הסף הטוב ביותר שנמצא הוא {best_threshold:.2f}.\n\n")

    f.write("תוצאות המודל המשופר:\n\n")
    f.write(improvement_df.to_markdown(index=False))
    f.write("\n\n")

    f.write(
        "שיפור זה הופך את תהליך בניית המודל למקצועי יותר, משום שהוא אינו מסתפק בהרצת מודל בודד "
        "עם פרמטרים קבועים, אלא בודק שילובים שונים של פרמטרים ובוחר סף החלטה שמתאים טוב יותר לבעיה.\n"
    )

print("\nSaved files:")
print("- results/improved_random_forest_results.csv")
print("- results/threshold_tuning_validation.csv")
print("- results/threshold_tuning_plot.png")
print("- report/model_improvement_section_hebrew.md")

print("\nModel improvement completed successfully.")
