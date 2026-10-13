"""
Model Training Script for Agro World
Trains:
1. Crop Recommendation Model (RandomForest on Crop_recommendation.csv, 22 classes, 7 features)
2. Fertilizer Recommendation Model (RandomForest on Fertilizer Prediction.csv, 7 classes, 8 features)

Saves models and scalers to: models/ML_models/
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models", "ML_models")

# Consistent categorical mappings used across the application
SOIL_TYPES = ['Black', 'Clayey', 'Loamy', 'Red', 'Sandy']
CROP_TYPES_FERT = [
    'Barley', 'Cotton', 'Ground Nuts', 'Maize', 'Millets',
    'Oil seeds', 'Paddy', 'Pulses', 'Sugarcane', 'Tobacco', 'Wheat'
]
FERTILIZER_CLASSES = ['10-26-26', '14-35-14', '17-17-17', '20-20', '28-28', 'DAP', 'Urea']

CROP_CLASSES = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut',
    'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans',
    'lentil', 'maize', 'mango', 'mothbeans', 'mungbean',
    'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate',
    'rice', 'watermelon'
]


def train_crop_recommendation(save_model=True):
    print("=" * 60)
    print("Training Crop Recommendation Model...")
    print("=" * 60)
    
    csv_path = os.path.join(DATASET_DIR, "Crop_recommendation.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Crop dataset not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape} (2200 samples, 22 crops, 100 per crop)")
    
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[feature_cols]
    
    crop_to_idx = {crop: idx for idx, crop in enumerate(CROP_CLASSES)}
    y = df['label'].map(crop_to_idx)
    
    if y.isna().any():
        unknowns = df['label'][y.isna()].unique()
        raise ValueError(f"Found unknown crops in dataset: {unknowns}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy:  {test_acc * 100:.2f}%")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_preds, target_names=CROP_CLASSES, digits=4))
    
    if save_model:
        os.makedirs(MODELS_DIR, exist_ok=True)
        scaler_out = os.path.join(MODELS_DIR, "crop_scaler.pkl")
        model_out = os.path.join(MODELS_DIR, "crop_model.pkl")
        
        with open(scaler_out, 'wb') as f:
            pickle.dump(scaler, f)
        with open(model_out, 'wb') as f:
            pickle.dump(model, f)
            
        print(f"Saved crop scaler to: {scaler_out}")
        print(f"Saved crop model to:  {model_out}")
        
    return model, scaler


def train_fertilizer_recommendation(save_model=True):
    print("\n" + "=" * 60)
    print("Training Fertilizer Recommendation Model...")
    print("=" * 60)
    
    csv_path = os.path.join(DATASET_DIR, "Fertilizer Prediction.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fertilizer dataset not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape} (99 samples across 7 fertilizer types)")
    print(
        "NOTE: With only 99 samples (~14 samples/class), high training accuracy is expected,\n"
        "but cross-validation must be used to realistically estimate generalization."
    )
    
    num_cols = ['Temparature', 'Humidity ', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    soil_to_idx = {s: i for i, s in enumerate(SOIL_TYPES)}
    crop_to_idx = {c: i for i, c in enumerate(CROP_TYPES_FERT)}
    fert_to_idx = {f: i for i, f in enumerate(FERTILIZER_CLASSES)}
    
    X_num = df[num_cols].values
    soil_enc = df['Soil Type'].map(soil_to_idx).values.reshape(-1, 1)
    crop_enc = df['Crop Type'].map(crop_to_idx).values.reshape(-1, 1)
    y = df['Fertilizer Name'].map(fert_to_idx).values
    
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)
    
    # 8 features total: 6 scaled numerical [Temp, Humidity, Moisture, N, K, P] + 2 categorical [Soil, Crop]
    X = np.concatenate([X_num_scaled, soil_enc, crop_enc], axis=1)
    
    # 5-fold cross validation for honest evaluation on small dataset
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(cv_clf, X, y, cv=skf)
    print(f"5-Fold Stratified CV Accuracy: {cv_scores.mean() * 100:.2f}% (std: {cv_scores.std() * 100:.2f}%)")
    print(f"Per-fold accuracy: {[round(float(s) * 100, 2) for s in cv_scores]}")
    
    # Holdout evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"Holdout (80/20) Test Accuracy: {test_acc * 100:.2f}%")
    print("\nClassification Report (Holdout Test Set):")
    print(classification_report(y_test, test_preds, target_names=FERTILIZER_CLASSES, digits=4, zero_division=0))
    
    if save_model:
        os.makedirs(MODELS_DIR, exist_ok=True)
        scaler_out = os.path.join(MODELS_DIR, "fertilizer_scaler.pkl")
        model_out = os.path.join(MODELS_DIR, "fertilizer_model.pkl")
        
        with open(scaler_out, 'wb') as f:
            pickle.dump(scaler, f)
        with open(model_out, 'wb') as f:
            pickle.dump(model, f)
            
        print(f"Saved fertilizer scaler to: {scaler_out}")
        print(f"Saved fertilizer model to:  {model_out}")
        
    return model, scaler


def main():
    parser = argparse.ArgumentParser(description="Train ML models for Agro World")
    parser.add_argument("--crop", action="store_true", help="Train crop recommendation model only")
    parser.add_argument("--fertilizer", action="store_true", help="Train fertilizer recommendation model only")
    parser.add_argument("--no-save", action="store_true", help="Do not save trained model artifacts")
    args = parser.parse_args()
    
    train_all = not (args.crop or args.fertilizer)
    save = not args.no_save
    
    if train_all or args.crop:
        train_crop_recommendation(save_model=save)
        
    if train_all or args.fertilizer:
        train_fertilizer_recommendation(save_model=save)
        
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
