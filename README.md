🏥 Loan Risk Prediction — Full ML Pipeline
This repository contains an end-to-end machine learning pipeline designed to predict loan approval outcomes. The project transitions from raw data exploration to a comparative analysis of classical machine learning algorithms and deep learning architectures, with a heavy focus on data integrity and reproducibility.

🚀 Key Engineering Highlights
Leakage Prevention: Data is split into training and test sets before any preprocessing or outlier detection to ensure no information from the test set influences the training phase.

Dynamic Preprocessing: Features an automated ColumnTransformer that auto-detects numeric and categorical columns, applying StandardScaler and OneHotEncoder respectively.

Threshold Calibration: Instead of a hardcoded 0.5 classification threshold, the ANN model utilizes a precision-recall curve to find the optimal F1-score threshold on a stratified validation slice.

Model Persistence: The full pipeline exports trained models and the fitted preprocessor as .pkl and .keras files for seamless production deployment.

🛠️ Technical Stack
Logic: Python, NumPy, Pandas

Visualization: Matplotlib, Seaborn

Classical ML: Scikit-learn (Logistic Regression, Random Forest), XGBoost

Deep Learning: TensorFlow/Keras (Sequential ANN with Dropout layers)

Serialization: Joblib

📁 Project Structure
Plaintext
├── notebooks/
│   └── analysis.ipynb      # Full research pipeline (EDA to Evaluation)
├── src/
│   └── streamlit_app.py    # Production-ready web dashboard logic
├── saved_models/           # Serialized models and preprocessor
├── data/                   # Dataset storage
├── requirements.txt        # Reproducible environment configuration
└── .gitignore              # Ensures clean version control (excludes venv/ and IDE files)
📊 Pipeline Overview
1. EDA & Data Cleaning
Duplicate removal and missing value inspection.

Correlation heatmaps and class imbalance visualization to understand feature relationships.

2. Robust Preprocessing
Outlier Detection: An IQR-based mask is applied to the training data to remove anomalies without affecting the test distribution.

Transformation: Categorical variables are handled with OneHotEncoder(drop="first") to prevent the dummy variable trap.

3. Model Training & Comparison
We evaluate four distinct architectures using 5-Fold Stratified Cross-Validation:

Logistic Regression: Baseline with balanced class weights.

Random Forest: Non-linear ensemble approach.

XGBoost: Gradient boosted decision trees optimized with scale_pos_weight.

Deep Learning (ANN): A multi-layer perceptron with Early Stopping to prevent overfitting.

📉 Results
Models are evaluated on the test set using a comprehensive suite of metrics:

F1-Score (Primary metric due to class imbalance)

Precision & Recall

ROC-AUC

Confusion Matrices
