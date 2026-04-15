import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train_model():
    print("Loading dataset...")
    data_path = 'data/credit_risk_dataset.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    try:
        chunks = pd.read_csv(data_path, chunksize=5000)
        df = pd.concat(chunks)
    except Exception as e:
        print(f"Pandas error: {e}")
        return

    # basic preprocessing
    print("Preprocessing data...")
    
    # Fill missing values
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

    # Drop outliers as per common practice with this dataset (age > 100, emp_length > 60)
    df = df[df['person_age'] < 100]
    df = df[df['person_emp_length'] < 60]

    # Categorical encoding
    cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Features and Target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Save artifacts
    print("Saving artifacts to backend/...")
    if not os.path.exists('backend'):
        os.makedirs('backend')
        
    joblib.dump(model, 'backend/model.joblib')
    joblib.dump(encoders, 'backend/encoders.joblib')
    joblib.dump(X.columns.tolist(), 'backend/features.joblib')
    
    print("Done!")

if __name__ == "__main__":
    train_model()
