# Results Directory

This directory contains the tables, figures, and evaluation reports generated during the project experiments.

The CSV files are the main source for the numerical results presented in the project README and final report.

## Main Result Files

### `model_comparison.csv`

Comparison of the five main classification algorithms:

* Logistic Regression
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* Linear SVM

The file includes accuracy, precision, recall, F1, ROC AUC, and runtime.

### `imbalance_comparison.csv`

Comparison between:

* A dummy majority-class classifier.
* Models trained without class balancing.
* Models trained with class balancing.

This experiment demonstrates why accuracy alone is not sufficient for the dataset.

### `random_forest_feature_importance.csv`

Random Forest feature-importance values in descending order.

The values describe the contribution of the input features to the fitted Random Forest model. They should not be interpreted as causal effects.

### `improved_random_forest_results.csv`

Results from the Random Forest parameter-tuning and threshold-adjustment experiment.

The file includes:

* Baseline Random Forest.
* Tuned Random Forest with threshold 0.50.
* Tuned Random Forest with threshold 0.45.

### `course_extensions_adaboost_results.csv`

Evaluation results for the AdaBoost extension.

## Figures and Text Reports

The directory also contains generated visualizations and classification reports, including:

* Target distribution.
* Model-comparison charts.
* Confusion matrices.
* Feature-importance charts.
* Class-imbalance comparisons.
* PCA-related visualizations.
* Per-model classification reports.

## Notes

* Runtime values depend on the computer and software environment.
* The reported project metrics are taken directly from the CSV files in this directory.
* Files should not be manually edited after an experiment. Interpretations and conclusions belong in the main README or final report.
