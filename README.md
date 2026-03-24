# Credit_Risk_Classification using Machine Learning
End-to-end credit risk classification project using Logistic Regression, Random Forest, and XGBoost. Includes data cleaning, feature engineering, handling imbalanced data, model evaluation using AUC and recall, threshold tuning based on business cost, and feature importance analysis aligned with real-world lending behavior.

This project is part of a structured transition into data-driven financial risk analysis.
## Overview
This project focuses on predicting whether a customer will default within the next 2 years using historical financial and behavioral data. The dataset is highly imbalanced, making it critical to focus on detecting defaulters effectively rather than relying on accuracy alone.
## Objective
- Build a robust classification model to predict credit default  
- Handle imbalanced data appropriately  
- Optimize model performance based on business risk (minimizing missed defaulters)  
## Approach
### 1. Data Preparation
- Removed invalid entries and handled missing values  
- Capped extreme outliers to stabilize distributions  
- Iteratively cleaned delinquency-related variables  
### 2. Feature Engineering
- Created domain-driven features such as:
  - `Ever_30_59_DPD`
  - `Ever_60_89_DPD`
  - `Ever_90plus_DPD`
  - `Has_RealEstate_Loan`  
- Focused on behavioral indicators of credit risk  
### 3. Models Used
- Logistic Regression (baseline)
- Random Forest (bagging approach)
- XGBoost (boosting approach)
## Model Evaluation
Evaluation focused on:
- **Recall** (to capture defaulters)
- **Precision**
- **F1 Score**
- **AUC (Area Under ROC Curve)**
### Key Observations:
- Accuracy was not used due to class imbalance  
- Random Forest showed poor recall for defaulters  
- Logistic Regression performed reasonably but lacked flexibility  
- **XGBoost achieved the best balance of recall and AUC**
## Threshold Tuning
- Explored threshold tuning for Logistic Regression (for understanding impact)  
- Final threshold tuning applied to **XGBoost**  
- Threshold selected to balance:
  - False Negatives (high financial loss)
  - False Positives (customer rejection)
## Feature Importance
Key predictors identified:
- Delinquency history (DPD variables)
- Credit utilization
- Credit exposure
These align with real-world credit risk assessment practices.
## Final Model
- **Selected Model:** XGBoost  
- **Reason:** Best performance in terms of AUC and recall  
- **Strength:** Handles imbalance and captures complex patterns effectively  
## Key Learnings
- Recall is more important than accuracy in risk-based problems  
- AUC should be used for model comparison before threshold tuning  
- Boosting models perform better on imbalanced datasets  
- Feature engineering based on domain knowledge significantly improves performance  
## Files
- `credit-risk-model.ipynb` → Complete notebook with analysis and modeling  
- Dataset → Raw credit risk dataset  
## Future Improvements
- Hyperparameter tuning for XGBoost  
- Cross-validation for more robust evaluation  
- Advanced interpretability techniques (e.g., SHAP)  
