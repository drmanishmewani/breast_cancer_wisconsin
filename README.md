# Breast Cancer Classification - Machine Learning Analysis

A systematic comparison of machine learning algorithms for breast cancer diagnosis using the Wisconsin Breast Cancer Dataset. 


## Overview

This project evaluates 11 different machine learning algorithms across 31 model configurations to identify the most reliable approach for classifying breast tumors as benign or malignant. FNAC is usually employed as a robust screening test even though histopathology remains the gold standard for diagnosis. 
Hencd the focus in this study was on building models where missing a malignant case (false negative) is considered more serious than incorrectly flagging a benign case (false positive).

**Dataset:** Wisconsin Breast Cancer Dataset (569 samples, 30 features)
- Features derived from digitized images of fine needle aspirate (FNA) of breast masses
- 357 benign cases, 212 malignant cases (mild-moderate imbalance)
- hence average_precision was taken as target metric across the study.
- labels are benign=1 and malignant=0. Hence the target labels were changed to to benign=0 and malignant=1 to make it more intuitive.


### Best Overall Models (Default Threshold)
- **Logistic Regression (Ridge/L2)**: 98.25% accuracy, 1 false negative, 1 false positive
- **Elastic Net**: 98.25% accuracy, 1 false negative, 1 false positive  
- **SVM (RBF kernel)**: 98.25% accuracy, 2 false negatives, 0 false positives

### Best for Zero False Negatives (Screening Mode)
- **Random Forest** (optimized threshold): 100% sensitivity, 97.18% specificity, only 2 false positives
- Catches all 43 malignant cases in test set while minimizing unnecessary biopsies

### Model Performance Summary

| Model | Accuracy | False Negatives | False Positives | Notes |
|-------|----------|----------------|-----------------|-------|
| Ridge Regression | 98.25% | 1 | 1 | Best balanced performance |
| Elastic Net | 98.25% | 1 | 1 | Simplified to 27 features |
| SVM (tuned) | 98.25% | 2 | 0 | Perfect specificity |
| Lasso | 97.37% | 1 | 2 | Only 8 features (interpretable) |
| AdaBoost | 97.37% | 2 | 1 | Good ensemble method |
| Random Forest | 96.49% | 3 | 1 | Excellent with threshold tuning |
| Naive Bayes | 96.49% | 3 | 1 | Simplest model |
| XGBoost | 95.61% | 3 | 2 | Extensive tuning required |
| KNN + PCA | 96.49% | 2 | 2 | PCA essential for KNN |
| Decision Tree | 94.74% | 3 | 3 | Baseline comparison |


## Methodology

### 1. Data Preprocessing
- Feature standardization using StandardScaler (required for distance-based and regularized models)
- Class imbalance handling via **class_weight='balanced'** parameter
- Addressed severe multicollinearity (radius/perimeter/area features with correlations ~1.0)

### 2. Models Evaluated
**Linear Models:**
- Logistic Regression (baseline)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Elastic Net (L1 + L2 combination)

**Distance-Based:**
- K-Nearest Neighbors (with and without PCA)
- Support Vector Machine (RBF and linear kernels)

**Probabilistic:**
- Gaussian Naive Bayes

**Tree-Based:**
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost

### 3. Hyperparameter Tuning
- Systematic GridSearchCV for each algorithm
- 5-fold stratified cross-validation (maintains class distribution)
- Optimization metric: PR-AUC (Precision-Recall AUC) - more appropriate for imbalanced data (eg. disease prediction datasets) than ROC-AUC

### 4. Threshold Optimization
- Analyzed classification thresholds from 0 to 1
- Generated trade-off tables showing false negatives vs. false positives at different operating points
- Identified optimal thresholds for:
  - Maximum sensitivity (screening scenarios)
  - Maximum specificity (confirmation scenarios) = not much utility with current dataset because histopathology is considered gold standard for diagnosis
  - Balanced performance

***

## Technical Details

### Multicollinearity Handling
The correlation heatmap revealed perfect correlations (r = 1.0) between:
- mean radius ↔ mean perimeter ↔ mean area
- worst radius ↔ worst perimeter ↔ worst area

**Solutions implemented:**
- Ridge/Elastic Net: L2 regularization shrinks correlated coefficients
- Lasso: Feature selection eliminated redundant features
- PCA: Transformed to 10 uncorrelated principal components for KNN
- Tree methods: Naturally handle correlation via feature splitting


## Clinical Interpretation

### False Negative Analysis
- Best models: 1 false negative out of 43 malignant cases (2.3% miss rate)
- Random Forest at FN=0: Can detect 100% of cancers with only 2.8% false positive rate hence perfect as a screening tool for FNAC requiring human confirmation in the next stage.

### Recommendations by Use Case

**Primary Screening:**
- Use Random Forest with threshold ~0.35
- Ensures zero missed cancers
- Only 2 extra biopsies needed per 114 patients

**Confirmatory Diagnosis:** (Not applicable due to cytology not being the gold standard)
- Use Ridge/Elastic Net at default threshold (0.5)
- 98.25% accuracy with balanced errors
- Simplest to deploy and maintain

**Resource-Constrained Settings:**
- Use standard Logistic Regression


**Explainability Priority:**
- Use Lasso with 8 features
- Clinically interpretable feature set
- 97.37% accuracy still strong


## Limitations and Future Work

**Current Limitations:**
- Test set is relatively small (114 samples)
- Dataset is well-curated and may not reflect real-world noise
- No external validation on independent datasets
- Threshold optimization done post-hoc rather than during training

**Future Directions:**
- Ensemble voting classifier (Random Forest + Ridge + SVM)
- Probability calibration for better confidence estimates
- Cost-sensitive learning with asymmetric loss functions
- External validation on other breast cancer datasets
- Integration with slide review mechanisms


## Summary:
In this study, the breast cancer dataset was taken which provides detailed measurements of the cells aspirated from an FNAC procedure and provides the target (diagnosis) as either benign or malignant. 
This study was undertaken to help make predictions regarding the diagnosis (benign or malignant).

Technique limitations:
On FNAC, it is sometimes not possible to classify the disease as benign or malignant with 100% confidence. But the dataset consisted of these 2 labels only. Therefore an effort was made to tune the threshold to minimize the false negatives.

Use Case:
This method to tune threshold for zero false negatives with minimum false positives suits the screening method.
The model could be incorporated into a screening software which could flag the probable malignant cases. These flagged cases should be further reviewed by a human being. Further tweaking could be done to the model according to the use case. 

## All Results:

| Model Name                        | ROC-AUC  | PR-AUC  | Accuracy | Balanced Accuracy | FN | FP | TN | TP | Recall   | Specificity | Precision | F1-Score  | Threshold  | Best_Params                                                                                           | Best_CV_Score | Param Grid                                                                                                                                    |
|----------------------------------|----------|---------|----------|-------------------|----|----|----|----|----------|-------------|-----------|-----------|------------|----------------------------------------------------------------------------------------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| RANDOM FOREST TUNED (FN=0 Opt)   | 0.998690 | 0.997872| 0.982456 | 0.985915          | 0  | 2  | 69 | 43 | 1.000000 | 0.971831    | 0.955556  | 0.977273  | 0.354608   | NaN                                                                                                | NaN           | NaN                                                                                                                                           |
| RANDOM FOREST TUNED              | 0.998690 | 0.997872| 0.964912 | 0.958074          | 3  | 1  | 70 | 40 | 0.930233 | 0.985915    | 0.975610  | 0.952381  | 0.500000   | {'max_depth': 7, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 300} | 0.985615      | {'n_estimators': [100, 200, 300], 'max_depth': [5, 7, 10], 'min_samples_split': [5, 10, 20], 'min_samples_leaf': [2, 5, 10], 'max_features': ['sqrt', 'log2', None]} |
| RANDOM FOREST                   | 0.998362 | 0.997356| 0.964912 | 0.958074          | 3  | 1  | 70 | 40 | 0.930233 | 0.985915    | 0.975610  | 0.952381  | 0.500000   | NaN                                                                                                | NaN           | NaN                                                                                                                                           |
| LR (FN=0 Optimized)             | 0.997380 | 0.996203| 0.938596 | 0.950704          | 0  | 7  | 64 | 43 | 1.000000 | 0.901408    | 0.860000  | 0.924731  | 0.121033   | NaN                                                                                                | NaN           | NaN                                                                                                                                           |
| LR + L2 (FN=0 Optimized)        | 0.997380 | 0.996203| 0.938596 | 0.950704          | 0  | 7  | 64 | 43 | 1.000000 | 0.901408    | 0.860000  | 0.924731  | 0.121033   | NaN                                                                                                | NaN           | NaN                                                                                                                                           |
| ELASTICNET (FN=0 Optimized)     | 0.997380 | 0.996203| 0.938596 | 0.950704          | 0  | 7  | 64 | 43 | 1.000000 | 0.901408    | 0.860000  | 0.924731  | 0.115244   | NaN                                                                                                | NaN           | NaN                                                                                                                                           |
| LR                             | 0.997380 | 0.996203| 0.982456 | 0.981330          | 1  | 1  | 70 | 42 | 0.976744 | 0.985915    | 0.976744  | 0.976744  | 0.500000   | NaN                                                                                                | NaN           | NaN                                                                                                                                           |
| LR + L2                        | 0.997380 | 0.996203| 0.982456 | 0.981330          | 1  | 1  | 70 | 42 | 0.976744 | 0.985915    | 0.976744  | 0.976744  | 0.500000   | {'C': 1}                                                                                          | 0.993354      | {'C': [0.001, 0.01, 0.1, 1, 10, 100]}                                                                                                        |
| ELASTICNET                    | 0.997380 | 0.996203| 0.982456 | 0.981330          | 1  | 1  | 70 | 42 | 0.976744 | 0.985915    | 0.976744  | 0.976744  | 0.500000   | {'C': 1, 'l1_ratio': 0.3}                                                                       | 0.993824      | {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}                                                               |
| NAIVE BAYES TUNED (FN=0 Opt)  | 0.997380 | 0.995843| 0.964912 | 0.971831          | 0  | 4  | 67 | 43 | 1.000000 | 0.943662    | 0.914894  | 0.955556  | 0.000021   | NaN                                                                                                | NaN           | NaN                                                                                                                                           |
| NAIVE BAYES                   | 0.997380 | 0.995843| 0.964912 | 0.958074          | 3  | 1  | 70 | 40 | 0.930233 | 0.985915    | 0.975610  | 0.952381  | 0.500000   | NaN                                                                                                | NaN           | NaN                                                                                                                                           |
| NAIVE BAYES TUNED             | 0.997380 | 0.995843| 0.964912 | 0.958074          | 3  | 1  | 70 | 40 | 0.930233 | 0.985915    | 0.975610  | 0.952381  | 0.500000   | {'var_smoothing': 1e-10}                                                                        | 0.960401      | {'var_smoothing': [1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05]}                                                                               |
| SVC                            | 0.997052 | 0.995602| 0.964912 | 0.962660          | 2  | 2  | 69 | 41 | 0.953488 | 0.971831    | 0.953488  | 0.953488  | 0.500000   | NaN                                                                                                | NaN           | NaN                                                                                                                                           |
| SVC (FN=0 Optimized)          | 0.996397 | 0.994802| 0.929825 | 0.943662          | 0  | 8  | 63 | 43 | 1.000000 | 0.887324    | 0.843137  | 0.914894  | 0.089500   | NaN                                                                                                | NaN           | NaN                                                                                                                                           |
| SVC TUNED                    | 0.996397 | 0.994802| 0.982456 | 0.976744          | 2  | 0  | 71 | 41 | 0.953488 | 1.000000    | 1.000000  | 0.976190  | 0.500000   | {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}                                                       | 0.995469      | {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1], 'kernel': ['rbf', 'linear']}                                         |
| LR + L1 (FN=0 Optimized)      | 0.996397 | 0.994446| 0.947368 | 0.957746          | 0  | 6  | 65 | 43 | 1.000000 | 0.915493    | 0.877551  | 0.934783  | 0.144153   | NaN                                                                                                | NaN           | NaN                                                                                                                                           |
| LR + L1                      | 0.996397 | 0.994446| 0.973684 | 0.974288          | 1  | 2  | 69 | 42 | 0.976744 | 0.971831    | 0.954545  | 0.965517  | 0.500000   | {'C': 1}                                                                                        | 0.991817      | {'C': [0.001, 0.01, 0.1, 1, 10, 100]}                                                                                                        |
| XGBOOST                      | 0.995414 | 0.993831| 0.964912 | 0.958074          | 3  | 1  | 70 | 40 | 0.930233 | 0.985915    | 0.975610  | 0.952381  | 0.500000   | NaN                                                                                                | NaN           | NaN                                                                                                                                           |
| XGBOOST TUNED (FN=0 Optimized)| 0.995087 | 0.993124| 0.903509 | 0.922535          | 0  | 11 | 60 | 43 | 1.000000 | 0.845070    | 0.796296  | 0.886598  | 0.023911   | NaN                                                                                                | NaN           | NaN                                                                                                                                           |
| XGBOOST TUNED                | 0.995087 | 0.993124| 0.956140 | 0.951032          | 3  | 2  | 69 | 40 | 0.930233 | 0.971831    | 0.952381  | 0.941176  | 0.500000   | {'colsample_bytree': 0.8, 'gamma': 0.1, 'learning_rate': 0.2, 'max_depth': 4, 'min_child_weight': 1, 'n_estimators': 100, 'reg_lambda': 5, 'subsample': 0.8} | 0.991925      | {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1, 0.2], 'max_depth': [3, 4, 5], 'min_child_weight': [1, 3, 5], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0], 'gamma': [0, 0.1], 'reg_lambda': [1, 5]} |
| GRADIENT BOOST (FN=0 Opt)    | 0.995087 | 0.992839| 0.921053 | 0.936620          | 0  | 9  | 62 | 43 | 1.000000 | 0.873239    | 0.826923  | 0.905263  | 0.000003   | NaN                                                                                                | NaN           | NaN                                                                                                                                           |
| GRADIENT BOOST               | 0.995087 | 0.992839| 0.964912 | 0.958074          | 3  | 1  | 70 | 40 | 0.930233 | 0.985915    | 0.975610  | 0.952381  | 0.500000   | {'learning_rate': 0.2, 'max_depth': 3, 'min_samples_split': 5, 'n_estimators': 300, 'subsample': 0.8} | 0.991397      | {'n_estimators': [100, 200, 300], 'learning_rate': [0.05, 0.1, 0.2], 'max_depth': [3, 4, 5], 'min_samples_split': [5, 10, 20], 'subsample': [0.8, 1.0]} |
| ADABOOST (FN=0 Optimized)    | 0.993777 | 0.991458| 0.885965 | 0.908451          | 0  | 13 | 58 | 43 | 1.000000 | 0.816901    | 0.767857  | 0.868687  | 0.355038   | NaN                                                                                                | NaN           | NaN                                                                                                                                           |
| ADABOOST                     | 0.993777 | 0.991458| 0.973684 | 0.969702          | 2  | 1  | 70 | 41 | 0.953488 | 0.985915    | 0.976190  | 0.964706  | 0.500000   | {'estimator__max_depth': 2, 'learning_rate': 1.0, 'n_estimators': 300}                           | 0.994447      | {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.1, 0.5, 1.0], 'estimator__max_depth': [1, 2, 3]}                                |
| KNN PCA (FN=0 Optimized)     | 0.990501 | 0.985863| 0.903509 | 0.922535          | 0  | 11 | 60 | 43 | 1.000000 | 0.845070    | 0.796296  | 0.886598  | 0.099431   | NaN                                                                                                | NaN           | NaN                                                                                                                                           |
| KNN AFTER PCA                | 0.990501 | 0.985863| 0.964912 | 0.962660          | 2  | 2  | 69 | 41 | 0.953488 | 0.971831    | 0.953488  | 0.953488  | 0.500000   | {'metric': 'minkowski', 'n_neighbors': 9, 'p': 1, 'weights': 'distance'}                         | 0.989373      | {'n_neighbors': [3, 5, 7, 9, 11, 15, 21], 'weights': ['uniform', 'distance'], 'metric': ['minkowski'], 'p': [1, 2, 3, 4, 5]}                   |
| KNN TUNED                   | 0.982312 | 0.979070| 0.947368 | 0.943990          | 3  | 3  | 68 | 40 | 0.930233 | 0.957746    | 0.930233  | 0.930233  | 0.500000   | {'metric': 'minkowski', 'n_neighbors': 7, 'p': 2, 'weights': 'distance'}                         | 0.991419      | {'n_neighbors': [3, 5, 7, 9, 11, 15, 21], 'weights': ['uniform', 'distance'], 'metric': ['minkowski'], 'p': [1, 2, 3, 4, 5]}                   |
| KNN BASIC                   | 0.981985 | 0.973182| 0.947368 | 0.943990          | 3  | 3  | 68 | 40 | 0.930233 | 0.957746    | 0.930233  | 0.930233  | 0.500000   | NaN                                                                                                | NaN           | NaN                                                                                                                                           |
| DECISION TREE TUNED (FN=0)    | 0.962987 | 0.939590| 0.377193 | 0.500000          | 0  | 71 | 0  | 43 | 1.000000 | 0.000000    | 0.377193  | 0.547771  | 0.000000   | NaN                                                                                                | NaN           | NaN                                                                                                                                           |
| DECISION TREE TUNED          | 0.962987 | 0.939590| 0.947368 | 0.943990          | 3  | 3  | 68 | 40 | 0.930233 | 0.957746    | 0.930233  | 0.930233  | 0.500000   | {'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 5, 'min_samples_split': 20}             | 0.935746      | {'max_depth': [3, 5, 7, 10, 15, 20, None], 'min_samples_split': [2, 5, 10, 20], 'min_samples_leaf': [1, 2, 5, 10], 'criterion': ['gini', 'entropy']} |
| DECISION TREE                | 0.951032 | 0.912252| 0.956140 | 0.951032          | 3  | 2  | 69 | 40 | 0.930233 | 0.971831    | 0.952381  | 0.941176  | 0.500000   | NaN                                                                                                | NaN           | NaN                                                                                                                                           |
