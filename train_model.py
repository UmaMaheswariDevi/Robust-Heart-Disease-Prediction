# train_model.py - Model Training, Cross-Validation, & Evaluation Script
import os
import sys
import numpy as np
import pandas as pd
from model import CustomStandardScaler, CustomLogisticRegression, train_and_save_model

def run_training_pipeline():
    print("=" * 65)
    print("      CARDIOGUARD AI - HEART DISEASE ML TRAINING PIPELINE      ")
    print("=" * 65)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "heart.csv")
    model_path = os.path.join(base_dir, "heart_disease_model.pkl")
    scaler_path = os.path.join(base_dir, "scaler.pkl")

    print(f"\n[1/4] Loading Dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"      Rows: {len(df)} | Features: {df.shape[1] - 1} | Target: 'target'")
    print(f"      Positive cases (Disease): {sum(df['target'] == 1)} ({sum(df['target'] == 1)/len(df)*100:.1f}%)")
    print(f"      Negative cases (Normal):  {sum(df['target'] == 0)} ({sum(df['target'] == 0)/len(df)*100:.1f}%)")

    print("\n[2/4] Training Model with 80/20 Stratified Split & L2 Regularization...")
    model, scaler, metrics = train_and_save_model(csv_path, model_path, scaler_path)

    print("\n[3/4] Evaluation Metrics on Held-out Test Set (61 Patients):")
    print(f"      - Accuracy:     {metrics['accuracy'] * 100:.2f}%")
    print(f"      - Precision:    {metrics['precision'] * 100:.2f}%")
    print(f"      - Recall:       {metrics['recall'] * 100:.2f}%")
    print(f"      - F1 Score:     {metrics['f1'] * 100:.2f}%")
    print("\n      Confusion Matrix:")
    print(f"      {'':18} Predicted Healthy    Predicted Disease")
    print(f"      Actual Healthy:    {metrics['tn']:^17} {metrics['fp']:^17}")
    print(f"      Actual Disease:    {metrics['fn']:^17} {metrics['tp']:^17}")

    print("\n[4/4] Feature Importance / Clinical Risk Factor Impact:")
    feature_labels = {
        'age': 'Age',
        'sex': 'Sex (Male/Female)',
        'cp': 'Chest Pain Type (0-3)',
        'trestbps': 'Resting Blood Pressure',
        'chol': 'Serum Cholesterol',
        'fbs': 'Fasting Blood Sugar > 120',
        'restecg': 'Resting ECG',
        'thalach': 'Maximum Heart Rate',
        'exang': 'Exercise Induced Angina',
        'oldpeak': 'ST Depression (Oldpeak)',
        'slope': 'Slope of ST Segment',
        'ca': 'Major Vessels Colored',
        'thal': 'Thalassemia Condition'
    }

    feature_names = [col for col in df.columns if col != 'target']
    impacts = sorted(zip(feature_names, model.coef_), key=lambda x: abs(x[1]), reverse=True)
    print(f"      {'Feature':<28} {'Weight (Impact)':<15} {'Effect on Risk'}")
    print("      " + "-" * 58)
    for feat, weight in impacts:
        label = feature_labels.get(feat, feat)
        direction = "Increases Risk (+)" if weight > 0 else "Protective/Lowers Risk (-)"
        print(f"      {label:<28} {weight:+.4f}{'':<8} {direction}")

    print("\n" + "=" * 65)
    print(f"SUCCESS: Model saved to:  {model_path}")
    print(f"         Scaler saved to: {scaler_path}")
    print("=" * 65)

if __name__ == "__main__":
    run_training_pipeline()
