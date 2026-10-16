# Credit Risk Classification

Predicting serious credit delinquency using Logistic Regression, Random Forest and XGBoost, with threshold tuning and feature importance analysis.

## Overview

This project predicts whether a borrower will become seriously delinquent (90+ days past due) within 2 years, using historical financial and behavioural data. The data is highly imbalanced (6.7% defaults), so the focus is on detecting defaulters (recall and AUC), not accuracy.

**Dataset:** Kaggle [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit), 150,000 borrowers, 6.7% default rate. Train/test split: 80/20, stratified.

## Objective

- Build a classification model to predict credit default
- Handle imbalanced data appropriately
- Choose a decision threshold that balances missed defaulters against false alarms

## Approach

### 1. Data Preparation
- Removed invalid entries and handled missing values
- Capped extreme outliers to stabilise distributions
- Cleaned delinquency-related variables iteratively

### 2. Feature Engineering
- Created behavioural features: `Ever_30_59_DPD`, `Ever_60_89_DPD`, `Ever_90plus_DPD`, `Has_RealEstate_Loan`

### 3. Models
- Logistic Regression (baseline)
- Random Forest
- XGBoost

## Results

| Model | AUC | Recall (defaulters) | Precision (defaulters) |
|---|---|---|---|
| Logistic Regression | 0.864 | 75% | 22% |
| Random Forest | 0.843 | 17% | 53% |
| XGBoost | 0.869 | 78% | 22% |

### Key observations
- Accuracy was not used because of class imbalance
- Random Forest had high precision but very low recall, so it missed most defaulters
- Logistic Regression was nearly as good as XGBoost (AUC 0.864 vs 0.869) and is easier to interpret
- XGBoost performed best on AUC and recall

## Threshold Tuning

- Default threshold (0.50): XGBoost recall 78%, precision 22%
- Lower threshold (0.30): recall rises to about 90%, but precision falls to about 15%
- The right threshold depends on the cost of a missed defaulter versus the cost of rejecting a good customer

## Feature Importance

The strongest predictors were:
- Delinquency history (30-59, 60-89 and 90+ days past due)
- Credit utilisation
- Credit exposure

## Files

- `Analysis/Credit_Risk_Classification.ipynb`: full notebook with analysis and modelling
- `Data/`: dataset used (Kaggle *Give Me Some Credit*)
- `requirements.txt`: Python libraries used

## Limitations and Future Improvements

- Thresholds were compared on the test set; tune them on a separate validation set instead
- Add cross-validation for a more robust evaluation
- Hyperparameter tuning for XGBoost
- Model interpretability with SHAP
