# Breast Cancer Prediction

## Overview
This project focuses on predicting breast cancer diagnosis (Malignant vs. Benign) using a machine learning model built with the CatBoost Classifier. Our main objective is to identify malignant tumors accurately while placing a strong emphasis on achieving high recall (minimizing false negatives) to avoid missed diagnoses. 

## Why CatBoost?
During the evaluation phase, several models were tested, including XGBoost, LightGBM, and Logistic Regression. CatBoost was ultimately selected as the final model for several key reasons:
- **Superior Out-of-the-box Performance:** CatBoost often provides better performance with default hyperparameters compared to other gradient boosting frameworks, requiring less exhaustive tuning.
- **Built-in Class Imbalance Handling:** With the `auto_class_weights='Balanced'` parameter, CatBoost seamlessly handles the mild class imbalance present in the dataset, effectively boosting the recall for the minority 'Malignant' class.
- **Robustness to Overfitting:** CatBoost uses oblivious decision trees as base predictors, which makes the model more robust to overfitting and generally improves generalization on unseen data.

## Dataset
The dataset utilized for this project is `data.csv`. 
- **Total Records:** 569 instances.
- **Features:** It contains 30 numerical features detailing characteristics of the cell nuclei, such as mean radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.
- **Target Variable:** The `diagnosis` column, containing values 'M' for Malignant and 'B' for Benign.

## Data Splitting (Training and Testing)
To evaluate the model's performance reliably, the data was split using a stratified strategy to preserve the class proportion:
- **Training Data:** 80% of the dataset (455 records) was used to train the CatBoost model.
- **Testing Data:** 20% of the dataset (114 records) was reserved for evaluating the model and creating the confusion matrix.

## Results Obtained Using CatBoost
The CatBoost model was configured with `auto_class_weights='Balanced'` to compensate for the mild class imbalance and heavily optimize recall for malignant tumors.

The performance metrics on the 20% unseen testing data are as follows:
- **Accuracy:** 97.37%
- **Precision:** 100.00%
- **Recall:** 92.86%
- **F1-Score:** 96.30%

The excellent precision (100%) indicates no false positives, and the high recall (92.86%) means the model effectively identifies the vast majority of malignant cases.
