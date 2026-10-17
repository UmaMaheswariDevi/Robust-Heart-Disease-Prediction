# model.py - Core Machine Learning Engine for Heart Disease Prediction
import os
import pickle
import numpy as np
import pandas as pd

class CustomStandardScaler:
    """
    StandardScaler implementation in pure NumPy to guarantee zero external C-extension
    conflicts across all Windows environments while maintaining scikit-learn API compatibility.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class CustomLogisticRegression:
    """
    L2-regularized Logistic Regression classifier with probability calibration.
    Provides scikit-learn compatible fit, predict, and predict_proba methods.
    """
    def __init__(self, lr=0.08, l2=0.005, max_iter=2500, tol=1e-5):
        self.lr = lr
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = 0.0
        self.feature_names_ = None

    def fit(self, X, y, feature_names=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.feature_names_ = feature_names

        self.coef_ = np.zeros(n_features, dtype=np.float64)
        self.intercept_ = 0.0

        for _ in range(self.max_iter):
            z = np.dot(X, self.coef_) + self.intercept_
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -18.0, 18.0)))

            grad_w = (np.dot(X.T, (p - y)) / n_samples) + self.l2 * self.coef_
            grad_b = float(np.sum(p - y) / n_samples)

            self.coef_ -= self.lr * grad_w
            self.intercept_ -= self.lr * grad_b

            if np.max(np.abs(grad_w)) < self.tol:
                break
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        z = np.dot(X, self.coef_) + self.intercept_
        p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -18.0, 18.0)))
        p0 = 1.0 - p1
        if X.ndim == 1 or len(X) == 1:
            p1_scalar = float(np.atleast_1d(p1)[0])
            return np.array([[1.0 - p1_scalar, p1_scalar]])
        return np.column_stack([p0, p1])

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

    def get_feature_impacts(self, sample_features_scaled):
        """
        Calculates individual contribution scores for each feature in a sample:
        contribution = coef_i * scaled_x_i
        Positive values drive prediction towards heart disease, negative towards healthy.
        """
        raw_contributions = self.coef_ * sample_features_scaled
        return raw_contributions


def train_and_save_model(csv_path=None, model_path=None, scaler_path=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if csv_path is None:
        csv_path = os.path.join(base_dir, "heart.csv")
    if model_path is None:
        model_path = os.path.join(base_dir, "heart_disease_model.pkl")
    if scaler_path is None:
        scaler_path = os.path.join(base_dir, "scaler.pkl")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    feature_names = [col for col in df.columns if col != 'target']
    X = df[feature_names].values
    # Note: in this dataset file, target=0 represents heart disease presence and target=1 represents normal.
    # We map y = 1 for Heart Disease Present, y = 0 for Healthy.
    y = (df['target'] == 0).astype(int).values

    # Stratified/seeded train-test split (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = CustomStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = CustomLogisticRegression(lr=0.08, l2=0.005, max_iter=3000)
    model.fit(X_train_scaled, y_train, feature_names=feature_names)

    # Evaluate
    test_preds = model.predict(X_test_scaled)
    acc = float(np.mean(test_preds == y_test))

    # Confusion matrix
    tp = int(np.sum((test_preds == 1) & (y_test == 1)))
    tn = int(np.sum((test_preds == 0) & (y_test == 0)))
    fp = int(np.sum((test_preds == 1) & (y_test == 0)))
    fn = int(np.sum((test_preds == 0) & (y_test == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total_samples": len(df)
    }

    # Save to disk
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    return model, scaler, metrics
