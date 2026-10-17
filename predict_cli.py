# predict_cli.py - Command Line Interface for Heart Disease Prediction
import os
import sys
import pickle
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model import CustomStandardScaler, CustomLogisticRegression, train_and_save_model

def load_system():
    model_path = os.path.join(SCRIPT_DIR, "heart_disease_model.pkl")
    scaler_path = os.path.join(SCRIPT_DIR, "scaler.pkl")
    csv_path = os.path.join(SCRIPT_DIR, "heart.csv")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print("Training model on dataset...")
        model, scaler, _ = train_and_save_model(csv_path, model_path, scaler_path)
    else:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    return model, scaler

def predict_single(features_dict):
    model, scaler = load_system()
    feature_order = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    raw_vals = [float(features_dict[k]) for k in feature_order]
    scaled = scaler.transform([raw_vals])
    proba = model.predict_proba(scaled)[0]
    percent = round(proba[1] * 100, 2)
    prediction = int(proba[1] >= 0.5)

    print("\n" + "=" * 55)
    print("         CARDIOGUARD AI - CLINICAL PREDICTION          ")
    print("=" * 55)
    print(f"Risk Probability:       {percent}%")
    if prediction == 1:
        print("Assessment Result:      [ALERT] POTENTIAL HEART DISEASE DETECTED")
        print("Risk Tier:              HIGH RISK (>= 50%)")
    else:
        print("Assessment Result:      [CLEAR] UNLIKELY TO HAVE HEART DISEASE")
        print("Risk Tier:              LOW RISK (< 50%)")
    print("=" * 55 + "\n")
    return prediction, percent

def interactive_mode():
    print("=" * 55)
    print("   CARDIOGUARD AI: INTERACTIVE CLINICAL ASSESSMENT     ")
    print("=" * 55)
    print("Enter patient parameters (press Enter for default):\n")

    fields = [
        ("age", "Age (years)", "54"),
        ("sex", "Sex (1=Male, 0=Female)", "1"),
        ("cp", "Chest Pain Type (0=typical, 1=atypical, 2=non-anginal, 3=asymptomatic)", "1"),
        ("trestbps", "Resting Blood Pressure (mm Hg)", "130"),
        ("chol", "Serum Cholesterol (mg/dL)", "240"),
        ("fbs", "Fasting Blood Sugar > 120 mg/dL (1=yes, 0=no)", "0"),
        ("restecg", "Resting ECG (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)", "0"),
        ("thalach", "Maximum Heart Rate Achieved (bpm)", "150"),
        ("exang", "Exercise Induced Angina (1=yes, 0=no)", "0"),
        ("oldpeak", "ST Depression Induced by Exercise (oldpeak)", "1.0"),
        ("slope", "Slope of Peak ST (0=upsloping, 1=flat, 2=downsloping)", "1"),
        ("ca", "Number of Major Vessels Colored (0-3)", "0"),
        ("thal", "Thalassemia (1=normal, 2=fixed, 3=reversible defect)", "2")
    ]

    patient = {}
    for key, prompt, default in fields:
        val = input(f"{prompt} [{default}]: ").strip()
        patient[key] = val if val else default

    predict_single(patient)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict heart disease risk from clinical metrics.")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt")
    parser.add_argument("--age", type=float, default=54)
    parser.add_argument("--sex", type=float, default=1)
    parser.add_argument("--cp", type=float, default=1)
    parser.add_argument("--trestbps", type=float, default=130)
    parser.add_argument("--chol", type=float, default=240)
    parser.add_argument("--fbs", type=float, default=0)
    parser.add_argument("--restecg", type=float, default=0)
    parser.add_argument("--thalach", type=float, default=150)
    parser.add_argument("--exang", type=float, default=0)
    parser.add_argument("--oldpeak", type=float, default=1.0)
    parser.add_argument("--slope", type=float, default=1)
    parser.add_argument("--ca", type=float, default=0)
    parser.add_argument("--thal", type=float, default=2)

    args = parser.parse_args()
    if args.interactive:
        interactive_mode()
    else:
        predict_single(vars(args))
