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
