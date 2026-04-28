from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from data_cleaning import remove_outliers_train

def preprocess_data_final(df):
    print("Preprocessing data and preventing leakage...")
    
    # 1. Split FIRST to prevent data leakage
    X = df.drop('LoanApproved', axis=1)
    y = df['LoanApproved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Clean only the training set
    num_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'YearsExperience']
    cat_cols = X_train.select_dtypes(include=['object']).columns
    X_train_clean, y_train_clean = remove_outliers_train(X_train, y_train, num_cols)
    
    # 3. Scale & Encode safely
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    preprocessor.set_output(transform="pandas")
    
    X_train_processed = preprocessor.fit_transform(X_train_clean)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train_clean, y_test