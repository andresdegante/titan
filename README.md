# Titanic Survival Predictor

Interactive web application for predicting the survival probability of Titanic passengers using machine learning.

---

## Overview

This project demonstrates how to apply machine learning to a classic dataset— the **Titanic passenger manifest**—to predict survival outcomes based on demographic and family features. The model is trained using **Logistic Regression**, **Random Forest**, and **Gradient Boosting** algorithms, then deployed as an interactive web app with Streamlit, allowing users to input passenger details and receive instant survival probabilities.

---

## Dataset

- **Source**: Modified version of the Kaggle Titanic training set (`train_formateado.csv`)
- **Number of passengers**: 891
- **Features**:  
  `Survived` (binary target), `Pclass` (class), `Sex`, `Age`, `SibSp` (siblings/spouses), `Parch` (parents/children), `FamilySize`, `FamilyType`, `AgeGroup`
- **Preprocessing**:  
  Categorical variables are label-encoded; numerical features are optionally standardized.

---

## Model Development

- **Algorithms tested**: Logistic Regression, Random Forest, Gradient Boosting
- **Best model**: Logistic Regression (**78.8% accuracy**, **ROC-AUC 0.83**)
- **Dataset split**: 80% train, 20% test (stratified)
- **Evaluation metrics**: Accuracy, ROC-AUC, precision, recall, F1-score, specificity

| Model               | Accuracy | ROC-AUC | Precision | Recall | F1-Score | Specifity |
|---------------------|----------|---------|-----------|--------|----------|-----------|
| LogisticRegression  | 78.8%    | 0.83    | 75%       | 68%    | 71%      | 85%       |
| RandomForest        | 76.0%    | 0.79    | 71%       | 64%    | 67%      | 84%       |
| GradientBoosting    | 73.2%    | 0.78    | 67%       | 59%    | 63%      | 82%       |

---

## Web Deployment

The **best model** and preprocessing objects are exported and served via a **Streamlit web app**, available online. Users can manually enter passenger information (class, sex, age, family members, etc.) and receive a real-time survival probability prediction.

---

## Repository Structure

- **Notebooks/scripts**: For data preprocessing, exploratory analysis, model training, and evaluation.
- **Models**: Exported model and encoders (`titanic_model.pkl`, `label_encoders.pkl`, `scaler.pkl`, `model_metadata.pkl`).
- **Streamlit app**: Source code for the interactive web interface (`app.py`).
- **Dataset**: Processed dataset (`train_formateado.csv`).

---

## Quick Start

1. **Clone the repository**
