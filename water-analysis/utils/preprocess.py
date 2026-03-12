import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib, os

FEATURES = [
    'ph', 'Hardness', 'Solids', 'Chloramines',
    'Sulfate', 'Conductivity', 'Organic_carbon',
    'Trihalomethanes', 'Turbidity'
]
TARGET = 'Potability'

SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'scaler.pkl')

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in FEATURES:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df

def get_features_target(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET] if TARGET in df.columns else None
    return X, y

def fit_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    return scaler

def load_scaler():
    return joblib.load(SCALER_PATH)

def scale_features(X, scaler=None):
    if scaler is None:
        scaler = load_scaler()
    return scaler.transform(X)

def potability_label(prob: float):
    if prob >= 0.70:
        return "Potable", "Safe to drink", "#2ecc71"
    elif prob >= 0.50:
        return "Likely Potable", "Probably safe – verify with lab", "#27ae60"
    elif prob >= 0.35:
        return "Uncertain", "Treatment recommended", "#f39c12"
    else:
        return "Not Potable", "Unsafe for drinking", "#e74c3c"
