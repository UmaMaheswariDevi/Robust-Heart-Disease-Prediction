# heart.py - CardioGuard AI: Heart Disease Risk Assessment System
import os
import sys
import pickle
import tkinter as tk
from tkinter import messagebox

# Attempt modern CustomTkinter, fall back to standard Tkinter/ttk if ever unavailable
try:
    import customtkinter as ctk
    HAS_CTK = True
except ImportError:
    HAS_CTK = False
    import tkinter.ttk as ttk

# Ensure local imports work when running from any working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model import CustomStandardScaler, CustomLogisticRegression, train_and_save_model

MODEL_FILE = os.path.join(SCRIPT_DIR, "heart_disease_model.pkl")
SCALER_FILE = os.path.join(SCRIPT_DIR, "scaler.pkl")
DATASET_FILE = os.path.join(SCRIPT_DIR, "heart.csv")


def load_or_train_model():
    """Loads existing trained model or trains a fresh one from heart.csv."""
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        try:
            with open(SCALER_FILE, "rb") as f:
                scaler = pickle.load(f)
            with open(MODEL_FILE, "rb") as f:
                model = pickle.load(f)
            return model, scaler, "Loaded existing model & scaler."
        except Exception as e:
            print(f"Error loading saved model: {e}. Retraining...")

    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Missing {DATASET_FILE}. Please ensure heart.csv is in the project folder.")

    model, scaler, metrics = train_and_save_model(DATASET_FILE, MODEL_FILE, SCALER_FILE)
    return model, scaler, f"Model trained successfully! (Test Accuracy: {metrics['accuracy']*100:.1f}%)"


# =========================================================================
# CustomTkinter Modern GUI Application
# =========================================================================
class CardioGuardModernApp(ctk.CTk if HAS_CTK else tk.Tk):
    def __init__(self, model, scaler, status_msg=""):
        super().__init__()
        self.model = model
        self.scaler = scaler

        self.title("CardioGuard AI — Heart Disease Risk Prediction System")
        self.geometry("1100x780")
        self.minsize(980, 700)

        if HAS_CTK:
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")

        self.create_widgets(status_msg)
        self.load_preset_healthy()

    def create_widgets(self, status_msg):
        # Top Header Bar
        header = ctk.CTkFrame(self, height=65, corner_radius=0, fg_color=("#1e293b", "#0f172a"))
        header.pack(fill="x", side="top")

        title_lbl = ctk.CTkLabel(
            header,
            text="❤️ CardioGuard AI  |  Clinical Risk Assessment System",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#38bdf8"
        )
        title_lbl.pack(side="left", padx=20, pady=15)

        # Theme toggle
        self.theme_switch = ctk.CTkSwitch(
            header,
            text="Dark Mode",
            command=self.toggle_theme,
            font=ctk.CTkFont(size=12)
        )
        self.theme_switch.select()
        self.theme_switch.pack(side="right", padx=20)

        # Main Body: Split into Left Input Form and Right Results Dashboard
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=15, pady=15)
        body.grid_columnconfigure(0, weight=6)
        body.grid_columnconfigure(1, weight=5)
        body.grid_rowconfigure(0, weight=1)

        # ---------------- Left Panel: Inputs (Scrollable) ----------------
        left_container = ctk.CTkFrame(body, corner_radius=12, fg_color=("#f8fafc", "#1e293b"))
        left_container.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        form_title = ctk.CTkLabel(
            left_container,
            text="Patient Clinical Parameters",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#0f172a", "#f1f5f9")
        )
        form_title.pack(anchor="w", padx=20, pady=(15, 5))

        self.scroll_form = ctk.CTkScrollableFrame(left_container, fg_color="transparent")
        self.scroll_form.pack(fill="both", expand=True, padx=10, pady=5)

        # Build Form Controls
        self.entries = {}

        # 1. Age
        self.create_input_row("Age (Years):", "age", default="54", hint="18 - 95")

        # 2. Sex
        self.create_segmented_row("Biological Sex:", "sex", ["Female", "Male"], default="Male")

        # 3. Chest Pain Type
        self.create_dropdown_row(
            "Chest Pain Type:",
            "cp",
            ["0: Typical Angina", "1: Atypical Angina", "2: Non-Anginal Pain", "3: Asymptomatic"],
            default="1: Atypical Angina"
        )

        # 4. Resting Blood Pressure
        self.create_input_row("Resting Blood Pressure (mm Hg):", "trestbps", default="130", hint="Normal: 90 - 140")

        # 5. Serum Cholesterol
        self.create_input_row("Serum Cholesterol (mg/dL):", "chol", default="240", hint="Normal: 125 - 200")

        # 6. Fasting Blood Sugar > 120
        self.create_segmented_row("Fasting Blood Sugar > 120 mg/dL:", "fbs", ["No (<= 120)", "Yes (> 120)"], default="No (<= 120)")

        # 7. Resting ECG
        self.create_dropdown_row(
            "Resting ECG:",
            "restecg",
            ["0: Normal", "1: ST-T Wave Abnormality", "2: LV Hypertrophy"],
            default="0: Normal"
        )

        # 8. Max Heart Rate
        self.create_input_row("Max Heart Rate Achieved (bpm):", "thalach", default="150", hint="70 - 210 bpm")

        # 9. Exercise Induced Angina
        self.create_segmented_row("Exercise Induced Angina:", "exang", ["No", "Yes"], default="No")

        # 10. ST Depression (oldpeak)
        self.create_input_row("ST Depression (oldpeak):", "oldpeak", default="1.0", hint="0.0 - 6.2 mm")

        # 11. Slope of ST Segment
        self.create_dropdown_row(
            "Slope of Peak Exercise ST:",
            "slope",
            ["0: Upsloping", "1: Flat", "2: Downsloping"],
            default="1: Flat"
        )

        # 12. Major Vessels Colored (ca)
        self.create_dropdown_row(
            "Major Vessels Colored by Fluoroscopy:",
            "ca",
            ["0 vessels", "1 vessel", "2 vessels", "3 vessels"],
            default="0 vessels"
        )

        # 13. Thalassemia
        self.create_dropdown_row(
            "Thalassemia Status:",
            "thal",
            ["1: Normal", "2: Fixed Defect", "3: Reversible Defect"],
            default="2: Fixed Defect"
        )

        # Bottom Buttons inside Left Panel
        btn_box = ctk.CTkFrame(left_container, fg_color="transparent")
        btn_box.pack(fill="x", padx=15, pady=10)

        predict_btn = ctk.CTkButton(
            btn_box,
            text="🩺 Run Risk Prediction",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#0284c7",
            hover_color="#0369a1",
            height=42,
            command=self.on_predict
        )
        predict_btn.pack(fill="x", pady=4)

        presets_box = ctk.CTkFrame(btn_box, fg_color="transparent")
        presets_box.pack(fill="x", pady=4)
        presets_box.columnconfigure((0, 1), weight=1)

        low_risk_btn = ctk.CTkButton(
            presets_box,
            text="👤 Sample: Low Risk",
            fg_color="#059669",
            hover_color="#047857",
            height=32,
            command=self.load_preset_healthy
        )
        low_risk_btn.grid(row=0, column=0, padx=2, sticky="ew")

        high_risk_btn = ctk.CTkButton(
            presets_box,
            text="⚠️ Sample: High Risk",
            fg_color="#dc2626",
            hover_color="#b91c1c",
            height=32,
            command=self.load_preset_high_risk
        )
        high_risk_btn.grid(row=0, column=1, padx=2, sticky="ew")

        # ---------------- Right Panel: Diagnostic Results ----------------
        right_container = ctk.CTkFrame(body, corner_radius=12, fg_color=("#f8fafc", "#1e293b"))
        right_container.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        report_title = ctk.CTkLabel(
            right_container,
            text="Diagnostic Assessment Report",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#0f172a", "#f1f5f9")
        )
        report_title.pack(anchor="w", padx=20, pady=(15, 10))

        # Outcome Card
        self.outcome_card = ctk.CTkFrame(right_container, corner_radius=10, fg_color=("#e2e8f0", "#334155"))
        self.outcome_card.pack(fill="x", padx=20, pady=5)

        self.outcome_icon = ctk.CTkLabel(self.outcome_card, text="🩺", font=ctk.CTkFont(size=36))
        self.outcome_icon.pack(pady=(12, 0))

        self.outcome_badge = ctk.CTkLabel(
            self.outcome_card,
            text="READY FOR EVALUATION",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#94a3b8"
        )
        self.outcome_badge.pack(pady=4)

        self.probability_lbl = ctk.CTkLabel(
            self.outcome_card,
            text="Click 'Run Risk Prediction' to calculate risk",
            font=ctk.CTkFont(size=13),
            text_color=("#475569", "#94a3b8")
        )
        self.probability_lbl.pack(pady=(0, 10))

        self.progress_bar = ctk.CTkProgressBar(self.outcome_card, height=14, corner_radius=7)
        self.progress_bar.set(0.0)
        self.progress_bar.pack(fill="x", padx=25, pady=(0, 15))

        # Contributing Clinical Factors Card
        factors_card = ctk.CTkFrame(right_container, corner_radius=10, fg_color=("#f1f5f9", "#0f172a"))
        factors_card.pack(fill="both", expand=True, padx=20, pady=10)

        ctk.CTkLabel(
            factors_card,
            text="Key Clinical Risk Drivers:",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#38bdf8"
        ).pack(anchor="w", padx=15, pady=(10, 5))

        self.factors_text = ctk.CTkTextbox(
            factors_card,
            fg_color="transparent",
            font=ctk.CTkFont(size=12),
            wrap="word",
            activate_scrollbars=True
        )
        self.factors_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.factors_text.insert(
            "1.0",
            "Individual patient risk contributors will appear here after calculation.\n\n"
            "This model evaluates cardiovascular strain, ST-segment deviations, chest pain presentation, "
            "and fluoroscopy findings."
        )
        self.factors_text.configure(state="disabled")

        # Recommendation Card
        recom_card = ctk.CTkFrame(right_container, corner_radius=10, fg_color=("#f1f5f9", "#0f172a"))
        recom_card.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(
            recom_card,
            text="Clinical Recommendations:",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#38bdf8"
        ).pack(anchor="w", padx=15, pady=(10, 5))

        self.recom_lbl = ctk.CTkLabel(
            recom_card,
            text="Awaiting patient parameters for recommendation.",
            font=ctk.CTkFont(size=12),
            wraplength=380,
            justify="left",
            text_color=("#475569", "#cbd5e1")
        )
        self.recom_lbl.pack(anchor="w", padx=15, pady=(0, 12))

        # Model Info Footer
        footer_lbl = ctk.CTkLabel(
            right_container,
            text=f"Engine: L2 Regularized Logistic Classifier | Accuracy: 86.89% | Cleveland UCI Dataset",
            font=ctk.CTkFont(size=10),
            text_color="#64748b"
        )
        footer_lbl.pack(side="bottom", pady=8)

    # ---------------- Helper UI Builders ----------------
    def create_input_row(self, label, key, default="", hint=""):
        row = ctk.CTkFrame(self.scroll_form, fg_color="transparent")
        row.pack(fill="x", pady=4)
        lbl = ctk.CTkLabel(row, text=label, font=ctk.CTkFont(size=12, weight="bold"), anchor="w")
        lbl.pack(side="left")
        if hint:
            hint_lbl = ctk.CTkLabel(row, text=f"({hint})", font=ctk.CTkFont(size=10), text_color="#64748b")
            hint_lbl.pack(side="left", padx=5)

        entry = ctk.CTkEntry(row, width=120, height=28, font=ctk.CTkFont(size=12))
        entry.insert(0, default)
        entry.pack(side="right")
        self.entries[key] = entry

    def create_segmented_row(self, label, key, values, default=""):
        row = ctk.CTkFrame(self.scroll_form, fg_color="transparent")
        row.pack(fill="x", pady=4)
        lbl = ctk.CTkLabel(row, text=label, font=ctk.CTkFont(size=12, weight="bold"), anchor="w")
        lbl.pack(side="left")

        seg = ctk.CTkSegmentedButton(row, values=values, height=28, font=ctk.CTkFont(size=11))
        seg.set(default if default else values[0])
        seg.pack(side="right")
        self.entries[key] = seg

    def create_dropdown_row(self, label, key, values, default=""):
        row = ctk.CTkFrame(self.scroll_form, fg_color="transparent")
        row.pack(fill="x", pady=4)
        lbl = ctk.CTkLabel(row, text=label, font=ctk.CTkFont(size=12, weight="bold"), anchor="w")
        lbl.pack(side="left")

        opt = ctk.CTkOptionMenu(row, values=values, width=170, height=28, font=ctk.CTkFont(size=11))
        opt.set(default if default else values[0])
        opt.pack(side="right")
        self.entries[key] = opt

    def toggle_theme(self):
        if self.theme_switch.get() == 1:
            ctk.set_appearance_mode("dark")
        else:
            ctk.set_appearance_mode("light")

    # ---------------- Presets ----------------
    def load_preset_healthy(self):
        """Loads a realistic low-risk case (patient with clear angiography, normal ECG)."""
        self.entries["age"].delete(0, "end"); self.entries["age"].insert(0, "41")
        self.entries["sex"].set("Female")
        self.entries["cp"].set("1: Atypical Angina")
        self.entries["trestbps"].delete(0, "end"); self.entries["trestbps"].insert(0, "120")
        self.entries["chol"].delete(0, "end"); self.entries["chol"].insert(0, "190")
        self.entries["fbs"].set("No (<= 120)")
        self.entries["restecg"].set("0: Normal")
        self.entries["thalach"].delete(0, "end"); self.entries["thalach"].insert(0, "172")
        self.entries["exang"].set("No")
        self.entries["oldpeak"].delete(0, "end"); self.entries["oldpeak"].insert(0, "0.2")
        self.entries["slope"].set("2: Downsloping")
        self.entries["ca"].set("0 vessels")
        self.entries["thal"].set("1: Normal")

    def load_preset_high_risk(self):
        """Loads a realistic high-risk case (ST depression, angina, vessel blockage)."""
        self.entries["age"].delete(0, "end"); self.entries["age"].insert(0, "64")
        self.entries["sex"].set("Male")
        self.entries["cp"].set("3: Asymptomatic")
        self.entries["trestbps"].delete(0, "end"); self.entries["trestbps"].insert(0, "155")
        self.entries["chol"].delete(0, "end"); self.entries["chol"].insert(0, "295")
        self.entries["fbs"].set("Yes (> 120)")
        self.entries["restecg"].set("1: ST-T Wave Abnormality")
        self.entries["thalach"].delete(0, "end"); self.entries["thalach"].insert(0, "118")
        self.entries["exang"].set("Yes")
        self.entries["oldpeak"].delete(0, "end"); self.entries["oldpeak"].insert(0, "2.6")
        self.entries["slope"].set("1: Flat")
        self.entries["ca"].set("2 vessels")
        self.entries["thal"].set("3: Reversible Defect")

    # ---------------- Prediction & Analytics ----------------
    def extract_features(self):
        try:
            age = float(self.entries["age"].get())
            sex = 1.0 if self.entries["sex"].get() == "Male" else 0.0

            cp_raw = self.entries["cp"].get()
            cp = float(cp_raw.split(":")[0].strip())

            trestbps = float(self.entries["trestbps"].get())
            chol = float(self.entries["chol"].get())

            fbs = 1.0 if "Yes" in self.entries["fbs"].get() else 0.0

            restecg_raw = self.entries["restecg"].get()
            restecg = float(restecg_raw.split(":")[0].strip())

            thalach = float(self.entries["thalach"].get())
            exang = 1.0 if self.entries["exang"].get() == "Yes" else 0.0
            oldpeak = float(self.entries["oldpeak"].get())

            slope_raw = self.entries["slope"].get()
            slope = float(slope_raw.split(":")[0].strip())

            ca_raw = self.entries["ca"].get()
            ca = float(ca_raw.split()[0].strip())

            thal_raw = self.entries["thal"].get()
            thal = float(thal_raw.split(":")[0].strip())

            # Basic sanity ranges
            if not (10 <= age <= 120):
                raise ValueError("Age should typically be between 10 and 120.")
            if not (50 <= trestbps <= 260):
                raise ValueError("Resting BP should be between 50 and 260 mm Hg.")
            if not (50 <= chol <= 700):
                raise ValueError("Cholesterol should be between 50 and 700 mg/dL.")
            if not (40 <= thalach <= 250):
                raise ValueError("Max Heart Rate should be between 40 and 250 bpm.")
            if not (0.0 <= oldpeak <= 10.0):
                raise ValueError("ST Depression should be between 0.0 and 10.0.")

            return [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        except ValueError as ve:
            messagebox.showerror("Invalid Input", str(ve))
            return None
        except Exception as e:
            messagebox.showerror("Error", f"Please check all input values:\n{e}")
            return None

    def on_predict(self):
        features = self.extract_features()
        if features is None:
            return

        # Scale features
        features_scaled = self.scaler.transform([features])[0]

        # Predict probability
        proba = self.model.predict_proba([features_scaled])[0]
        prob_disease = proba[1]
        percent = round(prob_disease * 100, 1)

        # Update Progress Bar & Outcome Card
        self.progress_bar.set(prob_disease)

        if prob_disease >= 0.5:
            # High Risk / Alert
            self.outcome_card.configure(fg_color="#7f1d1d")
            self.outcome_icon.configure(text="⚠️")
            self.outcome_badge.configure(
                text="ELEVATED RISK: HEART DISEASE LIKELY",
                text_color="#fca5a5"
            )
            self.probability_lbl.configure(
                text=f"Calculated Probability: {percent}% (High Alert Threshold >= 50%)",
                text_color="#fecaca"
            )
            self.progress_bar.configure(progress_color="#ef4444")
        else:
            # Low Risk / Healthy
            self.outcome_card.configure(fg_color="#064e3b")
            self.outcome_icon.configure(text="✅")
            self.outcome_badge.configure(
                text="LOW RISK: UNLIKELY TO HAVE HEART DISEASE",
                text_color="#6ee7b7"
            )
            self.probability_lbl.configure(
                text=f"Calculated Probability: {percent}% (Within Safe Threshold < 50%)",
                text_color="#a7f3d0"
            )
            self.progress_bar.configure(progress_color="#10b981")

        # Feature impacts analysis
        impacts = self.model.get_feature_impacts(features_scaled)
        feature_labels = [
            "Age", "Sex", "Chest Pain Type", "Resting Blood Pressure",
            "Cholesterol", "Fasting Blood Sugar", "Resting ECG",
            "Max Heart Rate Achieved", "Exercise Angina", "ST Depression",
            "Slope of ST", "Major Vessels (Fluoroscopy)", "Thalassemia"
        ]

        # Top 4 factors pushing towards or against risk
        factor_pairs = list(zip(feature_labels, impacts, features))
        factor_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        lines = []
        lines.append(f"Patient Score: {percent}% probability | Key Influences:\n")
        for name, impact, val in factor_pairs[:5]:
            icon = "🔴" if impact > 0 else "🟢"
            direction = "Increased Risk" if impact > 0 else "Decreased Risk"
            lines.append(f"  {icon} {name}: value={val}  ({direction})")

        self.factors_text.configure(state="normal")
        self.factors_text.delete("1.0", "end")
        self.factors_text.insert("1.0", "\n".join(lines))
        self.factors_text.configure(state="disabled")

        # Clinical Recommendations
        recoms = []
        if prob_disease >= 0.5:
            recoms.append("• Prompt consultation with a cardiologist is recommended.")
            if features[8] == 1.0 or features[9] > 1.5:
                recoms.append("• Significant exercise ischemia or ST depression noted; consider stress echocardiography.")
            if features[3] >= 140:
                recoms.append("• Blood pressure is hypertensive; assess lifestyle and antihypertensive regimen.")
            if features[4] >= 240:
                recoms.append("• Elevated cholesterol; consider lipid panel and statin therapy evaluation.")
        else:
            recoms.append("• Cardiovascular biomarkers appear within stable baseline ranges.")
            recoms.append("• Continue routine aerobic activity, balanced diet, and periodic screening.")
            if features[3] >= 135:
                recoms.append("• Monitor resting blood pressure periodically.")

        self.recom_lbl.configure(text="\n".join(recoms))


# =========================================================================
# Application Entry Point
# =========================================================================
def main():
    try:
        model, scaler, status_msg = load_or_train_model()
    except Exception as e:
        messagebox.showerror("Initialization Error", f"Failed to initialize machine learning model:\n{e}")
        return

    app = CardioGuardModernApp(model, scaler, status_msg)
    app.mainloop()


if __name__ == "__main__":
    main()
