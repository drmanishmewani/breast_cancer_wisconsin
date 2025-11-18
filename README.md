# breast_cancer_wisconsin
This project was undertaken to help make predictions from breast cancer dataset about diagnoses either benign or malignant.

In this study i took the breast cancer dataset which provides detailed measurements of the cells taken from a FNAC and provides the target (diagnosis) as either benign or malignant. 

FNAC is usually employed as a robust screening test even though histopathology remains the gold standard for diagnosis. 
On FNAC, it is sometimes not possible to classify the disease with 100% confidence as benign or malignant. But the dataset consistent of only 2 labels. Therefore an effort was made to 

First data exploration was done which revealed that:
- it is a mild to moderate imbalanced dataset with 65% benign and rest malignant. Therefore average_precision was taken as target metric across the study.
- labels are benign=1 and malignant=0. Hence I reversed the target labels to benign=0 and malignant=1 to make it more intuitive.

