# Stroke Prediction: Imbalanced Data Preprocessing & XAI (WIP)

**Author:** Tejas | B.Tech Computer Science (Expected 2028), VIT Vellore

## Project Overview
This repository contains the foundational data pipeline for an ongoing research project analyzing the empirical effects of synthetic oversampling techniques on feature interpretability within medical predictive models. 

**Dataset:** [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)  
This dataset was selected due to its severe class imbalance (~95% majority class, ~5% minority class), making it an ideal real-world environment to test the downstream effects of synthetic oversampling.

## Current Status
* **Phase 1: Exploratory Data Analysis & Preprocessing.** (Current phase)
* **Phase 2: Predictive Modeling & XAI Application.** (Currently in progress)

## Phase 1: Notebook Contents
The `stroke_eda_preprocessing_rough_draft.ipynb` notebook implements the initial Exploratory Data Analysis (EDA) and the rigorous preprocessing pipeline required before introducing synthetic data.

### Key Steps Implemented:
* **Data Cleaning:** Removal of irrelevant identifiers and coercion of noisy data types (e.g., BMI strings).
* **Exploratory Data Analysis (EDA):**
    * Baseline class ratio evaluation to quantify the imbalance severity.
    * Correlation heatmaps to identify primary drivers (e.g., Age) and establish a baseline before SMOTE-induced rank shifts.
    * Outlier detection via boxplots, specifically targeting glucose levels to monitor ADASYN generation boundaries.
    * Bivariate analysis (Age vs. Avg Glucose) to visualize the primary risk clusters and evaluate the placement of synthetic instances.
* **Preprocessing Pipeline:**
    * Construction of a robust `scikit-learn` pipeline ensuring no data leakage during cross-validation.
    * Missing value imputation using `IterativeImputer` for numerical features (like BMI) prior to `StandardScaler` application.
    * Categorical encoding using `OneHotEncoder` with dummy variable trap prevention.
    * Strict stratified train/test splitting to guarantee minority class representation in holdout sets.

## Tech Stack
* **Languages:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Imbalanced-Learn, Matplotlib, Seaborn