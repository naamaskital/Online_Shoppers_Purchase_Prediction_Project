# Results

This directory contains the tables, reports, and figures generated during the project.

## Exploratory Data Analysis

The EDA outputs include:

- Revenue-class distribution.
- Average numerical feature values by purchase outcome.
- Purchase rates across categorical variables.
- Tables and figures describing visitor behavior.

Main files include:

- `mean_features_by_revenue.csv`
- `revenue_distribution.png`
- `avg_*_by_revenue.png`
- `*_vs_revenue.csv`
- `purchase_rate_by_*.png`

## Main Model Comparison

`model_comparison.csv` contains the evaluation results for:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Linear SVM

The reported metrics are:

- Accuracy
- Precision
- Recall
- F1
- ROC AUC
- Runtime

Additional outputs include:

- `classification_reports.txt`
- `confusion_matrix_*.png`

## Class-Imbalance Analysis

`imbalance_comparison.csv` compares:

- A majority-class Dummy Classifier.
- Models trained without class weighting.
- Models trained with balanced class weights.

This experiment demonstrates why Accuracy alone is not sufficient for the dataset.

## Feature Importance

The Random Forest feature-importance analysis is stored in:

- `random_forest_feature_importance.csv`
- `random_forest_top_features.png`

Feature importance describes how the fitted model used each feature. It does not establish a causal relationship.

## Random Forest Improvement Experiment

The parameter-search and threshold-tuning outputs are:

- `improved_random_forest_results.csv`
- `threshold_tuning_validation.csv`
- `threshold_tuning_plot.png`

The experiment compares the baseline Random Forest with a tuned model and different decision thresholds.

## Course Extensions

AdaBoost results are stored in:

- `course_extensions_adaboost_results.csv`
- `model_comparison_with_adaboost.csv`

PCA results are stored in:

- `pca_explained_variance.csv`
- `pca_projection.csv`
- `pca_projection.png`

## Notes

The numerical values presented in the main README and final report are taken directly from the CSV files in this directory.

Runtime values can vary between computers and software environments.
