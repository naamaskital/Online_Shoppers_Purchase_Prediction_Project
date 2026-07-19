# Online Shoppers Purchase Prediction

Machine learning project for predicting whether an online shopping session will end with a purchase.

The project uses the **Online Shoppers Purchasing Intention Dataset** and compares several supervised classification algorithms. The main focus is not only overall accuracy, but also the ability to identify the less common purchase sessions.

## Project Goals

The project examines the following questions:

1. Can a purchase be predicted using information collected during an online shopping session?
2. Which machine learning algorithm performs best on this dataset?
3. How does the class imbalance affect the evaluation results?
4. Which session characteristics contribute most to the prediction?
5. Can parameter tuning improve the best-performing model?

## Dataset

The dataset contains **12,330 online shopping sessions**.

Each row represents one user session. The target column is:

* `Revenue = True`: the session ended with a purchase.
* `Revenue = False`: the session did not end with a purchase.

The dataset is imbalanced: approximately 15.5% of the sessions ended with a purchase.

The input variables describe:

* The number and duration of administrative, informational, and product-related page visits.
* Bounce rate and exit rate.
* Page value.
* Month and visitor type.
* Browser, operating system, region, and traffic source.
* Whether the session occurred during a weekend or near a special day.

## Project Structure

```text
Online_Shoppers_Purchase_Prediction_Project/
├── data/
│   └── online_shoppers_intention.csv
├── scripts/
│   ├── 01_eda.py
│   ├── 02_train_models.py
│   ├── 03_feature_importance.py
│   ├── 04_imbalance_analysis.py
│   ├── 05_random_forest_improvement.py
│   └── 06_course_extensions.py
├── results/
│   ├── model_comparison.csv
│   ├── imbalance_comparison.csv
│   ├── improved_random_forest_results.csv
│   ├── course_extensions_adaboost_results.csv
│   ├── random_forest_feature_importance.csv
│   └── generated figures and classification reports
├── report/
│   └── project documentation
├── requirements.txt
└── README.md
```

## Analysis Process

### 1. Exploratory Data Analysis

The first script examines:

* Dataset dimensions and column types.
* Target-class distribution.
* Numerical variables grouped by purchase outcome.
* Purchase rates across categorical variables.
* Basic relationships between session properties and purchases.

### 2. Data Preparation

The target variable is separated from the input features.

The data is divided into training and test sets using a stratified split, so both sets preserve the original purchase-to-non-purchase ratio.

The preprocessing includes:

* Standardization of numerical variables.
* One-hot encoding of categorical variables.
* Separate preprocessing inside a scikit-learn pipeline to prevent information from the test set from being used during training.

Some categorical variables are stored in the dataset as numerical codes. They are treated as categories rather than continuous numerical measurements.

### 3. Model Comparison

The following algorithms were evaluated:

* Logistic Regression
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* Linear Support Vector Machine

Because the dataset is imbalanced, model quality was evaluated using several metrics:

* Accuracy
* Precision
* Recall
* F1 score
* ROC AUC

## Main Results

| Model               | Accuracy | Precision | Recall |         F1 |    ROC AUC |
| ------------------- | -------: | --------: | -----: | ---------: | ---------: |
| Random Forest       |   0.8901 |    0.6312 | 0.6990 | **0.6634** | **0.9191** |
| Logistic Regression |   0.8410 |    0.4913 | 0.7435 |     0.5917 |     0.8932 |
| SVM                 |   0.8390 |    0.4869 | 0.7304 |     0.5843 |     0.8879 |
| Decision Tree       |   0.8487 |    0.5117 | 0.5157 |     0.5137 |     0.7127 |
| KNN                 |   0.8767 |    0.6822 | 0.3822 |     0.4899 |     0.7633 |

Random Forest achieved the highest F1 score and ROC AUC.

Although KNN achieved relatively high accuracy and precision, its recall was low. It missed many sessions that actually resulted in a purchase.

Logistic Regression and SVM achieved higher recall than Random Forest, but their lower precision produced more false-positive purchase predictions.

## Class-Imbalance Analysis

A classifier that always predicts “no purchase” achieves an accuracy of approximately **84.5%**, but its precision, recall, and F1 score for purchases are all zero.

This demonstrates why accuracy alone is not sufficient for this dataset.

For Random Forest:

| Version                 | Accuracy | Precision | Recall |         F1 |
| ----------------------- | -------: | --------: | -----: | ---------: |
| Without class balancing |   0.8982 |    0.7764 | 0.4817 |     0.5945 |
| With class balancing    |   0.8901 |    0.6312 | 0.6990 | **0.6634** |

Class balancing slightly reduced accuracy and precision, but substantially improved recall and F1.

This tradeoff was considered beneficial because the project aims to identify purchase sessions rather than mainly predict the majority class.

## Feature Importance

The most influential Random Forest features were:

| Feature                 | Importance |
| ----------------------- | ---------: |
| PageValues              |     0.3326 |
| ExitRates               |     0.0911 |
| ProductRelated_Duration |     0.0789 |
| ProductRelated          |     0.0715 |
| BounceRates             |     0.0561 |
| Administrative_Duration |     0.0465 |
| Administrative          |     0.0395 |
| Month_Nov               |     0.0235 |

`PageValues` was the most important feature by a considerable margin.

The results also indicate that engagement with product-related pages, exit behavior, and the timing of the session contribute to the prediction.

Feature importance describes how the trained model used the variables. It does not prove that a variable causes a purchase.

## Random Forest Improvement Attempt

A parameter search and decision-threshold experiment were performed to test whether the Random Forest result could be improved.

| Version                | Accuracy | Precision | Recall |         F1 |    ROC AUC |
| ---------------------- | -------: | --------: | -----: | ---------: | ---------: |
| Baseline Random Forest |   0.8869 |    0.6235 | 0.6806 | **0.6508** | **0.9149** |
| Tuned, threshold 0.50  |   0.8844 |    0.6147 | 0.6806 |     0.6460 |     0.9139 |
| Tuned, threshold 0.45  |   0.8796 |    0.5918 | 0.7173 |     0.6485 |     0.9139 |

The tuning attempt did not improve the baseline F1 or ROC AUC.

Lowering the threshold increased recall, but it also produced more false-positive predictions and reduced precision.

This experiment showed that a more complex configuration does not necessarily outperform a reasonable baseline.

## Course Extensions

AdaBoost was evaluated as an additional ensemble-learning method.

| Model    | Accuracy | Precision | Recall |     F1 | ROC AUC |
| -------- | -------: | --------: | -----: | -----: | ------: |
| AdaBoost |   0.8897 |    0.6618 | 0.5890 | 0.6233 |  0.9148 |

AdaBoost achieved good accuracy and ROC AUC, but its F1 score was lower than that of the balanced Random Forest.

PCA was also explored to examine dimensionality reduction and the amount of variance represented by the leading components.

## Main Challenges

### Imbalanced Target

Most sessions did not end with a purchase. A high accuracy score could therefore be achieved without correctly identifying purchase sessions.

This was addressed by:

* Comparing the models with a dummy baseline.
* Applying class weighting where supported.
* Reporting precision, recall, F1, and ROC AUC in addition to accuracy.

### Different Types of Variables

The dataset combines continuous values, counts, Boolean variables, text categories, and numerical category codes.

The preprocessing was therefore separated into numerical and categorical transformations.

### Choosing an Evaluation Metric

Accuracy favored models that predicted the majority class.

F1 was used as a central comparison metric because it balances precision and recall for the purchase class.

### Improvement Did Not Guarantee Better Results

Parameter tuning and threshold adjustment did not outperform the original Random Forest result.

This was retained in the project because unsuccessful experiments also provide useful information about the model and its tradeoffs.

## Conclusions

Random Forest provided the strongest overall result for predicting purchase sessions.

The main conclusions are:

* Accuracy alone is misleading for an imbalanced dataset.
* Class balancing improved the detection of purchase sessions.
* Random Forest provided the best balance between precision and recall.
* `PageValues` was the strongest predictive feature.
* Parameter tuning did not necessarily improve the baseline model.
* Evaluation should consider the practical cost of missed purchases and false purchase predictions.

## Running the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the scripts from the project root directory in the following order:

```bash
python3 scripts/01_eda.py
python3 scripts/02_train_models.py
python3 scripts/03_feature_importance.py
python3 scripts/04_imbalance_analysis.py
python3 scripts/05_random_forest_improvement.py
python3 scripts/06_course_extensions.py
```

The generated tables, reports, and figures are saved in the `results/` directory.

## Reproducibility Note

The repository contains the result files produced during the project experiments.

Runtime values may vary between computers. Small numerical differences may also occur between library versions, but the reported project results are preserved in the `results/` directory.
