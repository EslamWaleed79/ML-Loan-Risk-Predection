# Technical Report: Deep Learning vs Machine Learning

## 1. Project Architecture Overview
This repository implements a modular, production-ready machine learning pipeline. By decoupling data cleaning, preprocessing, and model training into discrete `src/` modules, we eliminate data leakage and ensure fair model comparison.

## 2. Model Justification: ML vs DL
Based on our empirical testing on the Loan Risk Dataset:

**When Classical ML is Better:**
* Tree-based algorithms (XGBoost, Random Forest) inherently handle the feature interactions present in tabular financial data (e.g., Credit Score vs. Loan Amount) without requiring deep architectures. They train faster and offer critical feature importance metrics necessary for regulatory compliance in banking.

**When Deep Learning (ANN) is Better:**
* The Neural Network becomes superior when the dataset scales to hundreds of thousands of rows, or if unstructured data (like raw transaction logs or text notes from loan officers) are incorporated into the feature space.