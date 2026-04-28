import pandas as pd

def load_data(filepath):
    """Loads and performs initial drop of duplicates and nulls."""
    df = pd.read_csv(filepath)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    return df

def remove_outliers_train(X_train, y_train, num_cols):
    """Removes outliers using IQR strictly on the training set."""
    train_data = pd.concat([X_train, y_train], axis=1)
    
    for col in num_cols:
        Q1, Q3 = train_data[col].quantile(0.25), train_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        train_data = train_data[(train_data[col] >= lower_bound) & (train_data[col] <= upper_bound)]
        
    return train_data.drop('LoanApproved', axis=1), train_data['LoanApproved']