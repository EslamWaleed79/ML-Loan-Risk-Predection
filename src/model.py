import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def train_and_cv_ml_models(X_train, y_train):
    print("\nTraining Classical ML Models...")
    # Handle class imbalance
    neg_cases = (y_train == 0).sum()
    pos_cases = (y_train == 1).sum()
    scale_weight = neg_cases / pos_cases 
    
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(scale_pos_weight=scale_weight, eval_metric='logloss', random_state=42)
    }
    
    trained = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        print(f"{name} 5-Fold CV F1-Score: {np.mean(cv_scores):.4f}")
        
        model.fit(X_train, y_train)
        trained[name] = model
        
    return trained