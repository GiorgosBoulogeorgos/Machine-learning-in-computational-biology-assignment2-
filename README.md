# Heart Disease Classification: A Reproducible rnCV Pipeline

This repository contains a reproducible machine learning pipeline for binary classification of coronary artery disease (CAD) using the Cleveland subset of the UCI Heart Disease dataset.

## Project Overview
The project implements a **Repeated Nested Cross-Validation (rnCV)** workflow to ensure unbiased performance estimation on a small clinical dataset (n=242). All preprocessing and feature selection steps are encapsulated within the CV folds to prevent data leakage.

### Key Features
* **Object-Oriented Design:** The pipeline is managed via a custom `RepeatedNestedCV` class.
* **Algorithm Comparison:** Evaluation of seven classifiers (LR, GNB, LDA, RF, LightGBM, XGBoost, CatBoost) under both default and Optuna-tuned hyperparameters.
* **Stability Selection:** Data-driven feature selection identifying a stable clinical subset (thal, ca, cp, sex, exang).
* **Interpretability:** Post-hoc interpretation using SHAP values and clinical error analysis.
* **Deployable Artefact:** The final model is serialized as a single scikit-learn Pipeline object.

## Performance Summary
Linear Discriminant Analysis (LDA) was identified as the winning algorithm:
* **Median MCC:** 0.632 (Full features), 0.667 (Reduced stable features)
* **Median AUC:** 0.895 
* **Median PR-AUC:** 0.894 

## Repository Structure
* `src/`: Core logic, including the `RepeatedNestedCV` class.
* `notebooks/`: Exploratory Data Analysis and model development.
* `models/`: Serialized `final_model.pkl`.
* `reports/`: Full academic report.
* `figures/` : All figures generated. 

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/GiorgosBoulogeorgos/Machine-learning-in-computational-biology-assignment2-](https://github.com/GiorgosBoulogeorgos/Machine-learning-in-computational-biology-assignment2-)