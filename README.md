# ❤️ CardioGuard AI — Heart Disease Risk Prediction System

**CardioGuard AI** is a complete, production-grade clinical decision-support application designed to predict the likelihood of cardiovascular disease based on standard patient biomarkers, exercise stress test metrics, and fluoroscopy findings.

Trained on the gold-standard **Cleveland UCI Heart Disease Dataset (303 clinical records)**, CardioGuard AI delivers **86.89% Test Accuracy** and **90.62% Clinical Recall**, prioritizing early disease detection to minimize false negatives.

---

## 🚀 Key Features

- **Vite Web Dashboard (`npm run dev`)**: Modern, high-performance web application built with HTML, Vanilla CSS, and JavaScript. Features glassmorphism panels, real-time "What-If" biomarker simulation, animated SVG probability gauges, sample patient presets, and clinical factor impact breakdowns.
- **Modern Desktop GUI (`heart.py`)**: Built with CustomTkinter, featuring responsive dual-pane layout, dark/light mode toggle, instant sample preset loaders, animated risk probability bar, and individualized risk factor breakdowns.
- **Command-Line Interface (`predict_cli.py`)**: Supports both interactive question-and-answer patient assessment and direct command-line arguments for automated pipelines.
- **Dedicated Training Pipeline (`train_model.py`)**: Includes stratified 80/20 train/test split, L2 regularization, confusion matrix evaluation, and clinical feature impact rankings.
- **Zero-Conflict Native ML Engine (`model.py` & `src/modelWeights.js`)**: Engineered with a pure NumPy and vanilla JavaScript machine learning core (`CustomStandardScaler` & `CustomLogisticRegression`) to ensure 100% reliability across all environments without DLL or security policy blocks.

---

## 📁 Project Structure

```text
D:\heart disease\
│
├── package.json              # Vite Dev & Build Configuration (npm run dev)
├── index.html                # Modern Web Dashboard Entry Point
├── src/
│   ├── main.js               # Application Controller & Interactive Event Logic
│   ├── modelWeights.js       # Client-Side Native Inference Engine
│   └── style.css             # Glassmorphism Design System & Micro-Animations
│
├── heart.py                  # Primary Desktop GUI Application (CardioGuard AI)
├── web_app.py                # Standalone Python Web Server (alternative local server)
├── predict_cli.py            # Command-Line Prediction Tool (Interactive & Flags)
├── train_model.py            # Full Model Training & Evaluation Pipeline
├── model.py                  # Core ML Engine (Scaler, Logistic Regression, Impact Analysis)
├── heart.csv                 # Verified Cleveland UCI Heart Disease Dataset (303 rows)
├── heart_disease_model.pkl   # Serialized Trained Model Weights & Hyperparameters
├── scaler.pkl                # Serialized Standardization Scaler (Mean & Scale)
├── requirements.txt          # Python Package Requirements
└── README.md                 # Complete Project Documentation & Clinical Dictionary
```

---

## 📊 Model Performance & Benchmarks

The model was evaluated using a held-out test split (20% of dataset, 61 unseen clinical cases):

| Metric | Score | Clinical Significance |
| :--- | :---: | :--- |
| **Accuracy** | **86.89%** | High overall diagnostic consistency across patient cohorts. |
| **Recall (Sensitivity)** | **90.62%** | Catches over 90% of true heart disease cases, minimizing dangerous false negatives. |
| **Precision** | **85.29%** | Low false alarm rate for healthy patients. |
| **F1-Score** | **87.88%** | Harmonic mean reflecting optimal balance between precision and recall. |

### Confusion Matrix (Test Set: 61 Patients)

```text
                     Predicted Healthy     Predicted Disease
Actual Healthy:             24                    5
Actual Disease:              3                   29
```

---

## 🖥️ How to Run

### 1. Launch with `npm run dev` (Web Dashboard)

In your terminal inside `D:\heart disease`:
```powershell
npm run dev
```
- Instantly opens the web dashboard at `http://localhost:5173/`.
- Features real-time simulation, sample patient presets, animated gauge, and factor breakdowns.

### 2. Launch the Desktop Application

Run the desktop GUI:
```powershell
python "D:\heart disease\heart.py"
```

### 3. Run Predictions from the Command Line

Interactive prompt:
```powershell
python "D:\heart disease\predict_cli.py" --interactive
```

---

## 🔬 Clinical Disclaimer
*CardioGuard AI is intended for educational, research, and clinical decision-support purposes. It is designed to assist trained healthcare providers and should not be used as a standalone diagnostic substitute for comprehensive medical evaluation.*
