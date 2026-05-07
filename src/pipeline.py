"""
Shared pipeline logic — used by both notebooks.
Edit this file to change the model; both notebooks reflect the change automatically.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_sample_weight

MODELS_DIR    = Path(__file__).parent.parent / 'models'
DATA_PATH     = Path(__file__).parent.parent / 'data' / 'Maternal Health Risk Data Set.csv'
FEATURE_COLS  = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
TARGET_COL    = 'RiskLevel'

def load_data():
    """Load and return features and target from CSV."""
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURE_COLS]
    y  = df[TARGET_COL]
    return X, y

def train_pipeline(X, y, random_state=42):
    """Train the full pipeline: scaler, label encoder, and MLP."""
    le     = LabelEncoder()
    y_enc  = le.fit_transform(y)
    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, stratify=y_enc, random_state=random_state
    )
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    sw = compute_sample_weight('balanced', y_train)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
    )
    mlp.fit(X_train_s, y_train, sample_weight=sw)

    return mlp, scaler, le, X_test_s, y_test

def save_artifacts(mlp, scaler, le):
    """Save trained artifacts to models/ directory."""
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(mlp,    MODELS_DIR / 'mlp_model.joblib')
    joblib.dump(scaler, MODELS_DIR / 'scaler.joblib')
    joblib.dump(le,     MODELS_DIR / 'label_encoder.joblib')
    print("Saved: mlp_model.joblib, scaler.joblib, label_encoder.joblib")

def load_artifacts():
    """Load trained artifacts from models/ directory."""
    mlp    = joblib.load(MODELS_DIR / 'mlp_model.joblib')
    scaler = joblib.load(MODELS_DIR / 'scaler.joblib')
    le     = joblib.load(MODELS_DIR / 'label_encoder.joblib')
    return mlp, scaler, le
