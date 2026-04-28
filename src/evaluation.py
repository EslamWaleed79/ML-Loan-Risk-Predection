import pandas as pd
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, f1_score

# Local imports
from data_cleaning import load_data
from preprocessing import preprocess_data_final
from model import train_and_cv_ml_models

def evaluate_all_with_ann(X_train, y_train, X_test, y_test, ml_models):
    print("\nTraining Deep Learning ANN...")
    neg_cases = (y_train == 0).sum()
    pos_cases = (y_train == 1).sum()
    class_weights = {0: 1.0, 1: float(neg_cases / pos_cases)}
    
    ann = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    ann.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, 
            callbacks=[early_stop], class_weight=class_weights, verbose=0)
    
    results = []
    
    # ML Evaluation
    for name, model in ml_models.items():
        preds = model.predict(X_test)
        results.append({"Model": name, "Accuracy": accuracy_score(y_test, preds), "F1-Score": f1_score(y_test, preds)})
        
    # ANN Evaluation
    ann_preds = (ann.predict(X_test) > 0.5).astype(int).flatten()
    results.append({"Model": "Deep Learning (ANN)", "Accuracy": accuracy_score(y_test, ann_preds), "F1-Score": f1_score(y_test, ann_preds)})
    
    final_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    print("\n=== FINAL TEST SET COMPARISON ===")
    print(final_df.to_markdown())

if __name__ == "__main__":
    # Point to the data folder from the src folder
    data_path = os.path.join(os.path.dirname(__file__), '../data/loan_risk_prediction_dataset.csv')
    
    df_clean = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data_final(df_clean)
    trained_ml = train_and_cv_ml_models(X_train, y_train)
    evaluate_all_with_ann(X_train, y_train, X_test, y_test, trained_ml)