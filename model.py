import os
import json
import time
import joblib
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from skimage.feature import hog, local_binary_pattern

# ================= CONFIG =================
DATASET_PATH = "dataset"
IMG_SIZE = 96
MODEL_DIR = "saved_models"

LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
METRICS_JSON_PATH = os.path.join(MODEL_DIR, "metrics.json")
GRAPH_PATH = os.path.join(MODEL_DIR, "graph.png")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
BEST_MODEL_INFO_PATH = os.path.join(MODEL_DIR, "best_model_info.json")

LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
LBP_BINS = LBP_POINTS + 2

os.makedirs(MODEL_DIR, exist_ok=True)

# ================= PREPROCESS =================
def preprocess_fingerprint(img):
    if img is None:
        return None

    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = gray.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray

# ================= GABOR FEATURES =================
def gabor_features(gray):
    feats = []
    gray_f = gray.astype(np.float32) / 255.0

    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        kernel = cv2.getGaborKernel(
            (9, 9), 4.0, theta, 8.0, 0.5, 0, ktype=cv2.CV_32F
        )
        filtered = cv2.filter2D(gray_f, cv2.CV_32F, kernel)
        feats.append(filtered.mean())
        feats.append(filtered.std())

    return np.array(feats, dtype=np.float32)

# ================= FEATURE EXTRACTION =================
def extract_features(img):
    gray = preprocess_fingerprint(img)
    if gray is None:
        return None

    # HOG
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True
    ).astype(np.float32)

    # LBP
    lbp = local_binary_pattern(
        gray.astype(np.uint8),
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform"
    )
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_BINS + 1),
        range=(0, LBP_BINS)
    )
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    # Edges
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    edge_density = np.array([edges.mean() / 255.0], dtype=np.float32)

    # Adaptive threshold ratio
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    threshold_ratio = np.array([th.mean() / 255.0], dtype=np.float32)

    # Gabor texture
    gabor = gabor_features(gray)

    features = np.hstack([
        hog_features,
        lbp_hist,
        edge_density,
        threshold_ratio,
        gabor
    ]).astype(np.float32)

    return features

# ================= LOAD DATASET =================
def load_dataset():
    X = []
    y = []
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset folder '{DATASET_PATH}' not found.")

    for label in sorted(os.listdir(DATASET_PATH)):
        folder = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(folder):
            continue

        for file_name in os.listdir(folder):
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in valid_exts:
                continue

            file_path = os.path.join(folder, file_name)

            try:
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                features = extract_features(img)

                if features is None:
                    continue

                X.append(features)
                y.append(label)

            except Exception:
                continue

    if len(X) == 0:
        raise ValueError("No valid images found in dataset.")

    return np.array(X, dtype=np.float32), np.array(y)

# ================= GRAPH =================
def save_graph(metrics_dict):
    model_names = list(metrics_dict.keys())
    acc_values = [metrics_dict[name]["Accuracy"] for name in model_names]

    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(model_names, acc_values)

    plt.title("Simulation Results: Model Accuracy Comparison")
    plt.xlabel("Models")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 100)

    for bar, val in zip(bars, acc_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.5,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(GRAPH_PATH, dpi=100)
    plt.close()

# ================= TRAIN =================
def train():
    print("Loading dataset...")
    X, y = load_dataset()
    print(f"Samples: {len(X)}")
    print(f"Feature length: {X.shape[1]}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ]),

        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),

        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=800, solver="lbfgs"))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=70,
            learning_rate=0.07,
            max_depth=5,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            eval_metric="mlogloss",
            n_estimators=180,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.95,
            colsample_bytree=0.95,
            tree_method="hist",
            n_jobs=-1
        )
    }

    metrics = {}
    trained_models = {}
    training_times = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start = time.time()

        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds) * 100
            precision = precision_score(y_test, preds, average="weighted", zero_division=0) * 100
            recall = recall_score(y_test, preds, average="weighted", zero_division=0) * 100
            f1 = f1_score(y_test, preds, average="weighted", zero_division=0) * 100
            elapsed = time.time() - start

            metrics[name] = {
                "Accuracy": round(acc, 2),
                "Precision": round(precision, 2),
                "Recall": round(recall, 2),
                "F1-Score": round(f1, 2)
            }
            trained_models[name] = model
            training_times[name] = round(elapsed, 2)

            print(
                f"{name}: "
                f"Accuracy={acc:.2f}% | "
                f"Precision={precision:.2f}% | "
                f"Recall={recall:.2f}% | "
                f"F1={f1:.2f}% | "
                f"time={elapsed:.2f}s"
            )

        except Exception as e:
            print(f"{name} failed: {e}")

    if len(metrics) == 0:
        raise RuntimeError("All models failed. No model was trained.")

    best_accuracy = max(m["Accuracy"] for m in metrics.values())
    top_models = [name for name, m in metrics.items() if m["Accuracy"] == best_accuracy]
    best_name = min(top_models, key=lambda name: training_times[name])
    best_model = trained_models[best_name]

    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    with open(BEST_MODEL_INFO_PATH, "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "accuracy": metrics[best_name]["Accuracy"],
                "precision": metrics[best_name]["Precision"],
                "recall": metrics[best_name]["Recall"],
                "f1_score": metrics[best_name]["F1-Score"],
                "img_size": IMG_SIZE,
                "feature_type": "CLAHE + HOG + LBP + edge density + adaptive threshold + Gabor",
                "training_times_seconds": training_times,
                "selection_rule": "highest accuracy, then fastest among ties"
            },
            f,
            indent=4
        )

    save_graph(metrics)

    print("\nTraining complete.")
    print(f"Best Model: {best_name}")
    print(f"Best Accuracy: {metrics[best_name]['Accuracy']:.2f}%")
    print(f"Saved files inside: {MODEL_DIR}")

if __name__ == "__main__":
    train()