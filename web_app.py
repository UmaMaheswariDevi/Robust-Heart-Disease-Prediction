# web_app.py - High-Performance Python Web Server for CardioGuard AI
import os
import sys
import json
import pickle
import mimetypes
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model import CustomStandardScaler, CustomLogisticRegression, train_and_save_model

MODEL_FILE = os.path.join(SCRIPT_DIR, "heart_disease_model.pkl")
SCALER_FILE = os.path.join(SCRIPT_DIR, "scaler.pkl")
DATASET_FILE = os.path.join(SCRIPT_DIR, "heart.csv")

def get_model_and_scaler():
    if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE)):
        model, scaler, _ = train_and_save_model(DATASET_FILE, MODEL_FILE, SCALER_FILE)
    else:
        with open(SCALER_FILE, "rb") as f:
            scaler = pickle.load(f)
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
    return model, scaler


class CardioGuardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        clean_path = self.path.split("?")[0].lstrip("/")
        if clean_path == "" or clean_path == "index.html":
            file_path = os.path.join(SCRIPT_DIR, "index.html")
        else:
            file_path = os.path.join(SCRIPT_DIR, clean_path.replace("/", os.sep))

        # Security check: ensure path is inside SCRIPT_DIR
        try:
            real_file_path = os.path.realpath(file_path)
            if not real_file_path.startswith(os.path.realpath(SCRIPT_DIR)):
                self.send_response(403)
                self.end_headers()
                return
        except Exception:
            self.send_response(404)
            self.end_headers()
            return

        if os.path.exists(file_path) and os.path.isfile(file_path):
            content_type, _ = mimetypes.guess_type(file_path)
            if file_path.endswith(".js"):
                content_type = "application/javascript"
            elif file_path.endswith(".css"):
                content_type = "text/css"
            elif not content_type:
                content_type = "application/octet-stream"

            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", f"{content_type}; charset=utf-8" if "text" in content_type or "javascript" in content_type else content_type)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"Server error: {e}".encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    def do_POST(self):
        if self.path == "/api/predict":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)

                model, scaler = get_model_and_scaler()

                keys = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
                features = [float(data.get(k, 0)) for k in keys]

                scaled = scaler.transform([features])[0]
                proba = model.predict_proba([scaled])[0]
                prob_percent = round(float(proba[1]) * 100, 1)
                prediction = int(proba[1] >= 0.5)

                # Impact breakdown
                impacts = model.get_feature_impacts(scaled)
                labels = [
                    "Age", "Biological Sex", "Chest Pain Type", "Resting Blood Pressure",
                    "Serum Cholesterol", "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate",
                    "Exercise Angina", "ST Depression", "ST Slope", "Major Vessels (ca)", "Thalassemia"
                ]
                factor_list = []
                for name, imp, val in zip(labels, impacts, features):
                    factor_list.append({"name": name, "impact": round(float(imp), 4), "val": val})
                factor_list.sort(key=lambda x: abs(x["impact"]), reverse=True)

                # Recommendations
                recoms = []
                if prediction == 1:
                    recoms.append("Immediate consultation with a cardiologist is recommended.")
                    if features[8] == 1.0 or features[9] > 1.5:
                        recoms.append("Significant exercise ischemia or ST depression observed; consider stress echocardiography.")
                    if features[3] >= 140:
                        recoms.append("Elevated resting blood pressure; evaluate antihypertensive medication.")
                    if features[4] >= 240:
                        recoms.append("High serum cholesterol detected; recommend comprehensive lipid profile and dietary intervention.")
                else:
                    recoms.append("Cardiovascular metrics currently within standard baseline parameters.")
                    recoms.append("Maintain regular cardiovascular exercise and schedule annual routine wellness checkups.")

                response_data = {
                    "prediction": prediction,
                    "probability": prob_percent,
                    "influences": factor_list[:5],
                    "recommendations": recoms
                }

                resp_bytes = json.dumps(response_data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)
            except Exception as e:
                err_bytes = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(err_bytes)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress default noisy console logging


def run_web_server(port=5000):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, CardioGuardHandler)
    url = f"http://localhost:{port}"
    print("=" * 65)
    print("      CARDIOGUARD AI — CLINICAL TELEMETRY WEB SERVER          ")
    print(f"      Dashboard URL: {url}")
    print("      Press Ctrl+C to terminate the server")
    print("=" * 65)
    webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    run_web_server()
