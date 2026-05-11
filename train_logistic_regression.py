import json
import pickle
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

RANDOM_SEED = 42
FEATURES = [
    "Toan", "Van", "Anh", "Ly", "Hoa", "Sinh", "Su", "Dia", "GDCD",
    "KHTN", "KHXH", "R", "I", "A", "S", "E", "C",
]
TARGET_COLUMN = "Major"
DATA_PATH = Path("data") / "synthetic_students.csv"
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports") / "reportRegression"

# Outputs
CHECKPOINT_PATH = MODEL_DIR / "lr_synthetic_best.pkl"
REPORT_JSON_PATH = REPORT_DIR / "classification_report.json"
REPORT_TXT_PATH = REPORT_DIR / "classification_report.txt"
METRICS_JSON_PATH = REPORT_DIR / "test_metrics.json"
PLOT_CM_PATH = REPORT_DIR / "confusion_matrix.png"
PLOT_TEST_METRICS_PATH = REPORT_DIR / "test_metrics.png"

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

def plot_confusion_matrix(cm, class_names, output_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Logistic Regression Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_test_metrics(test_metrics, output_path):
    metric_names = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
    metric_values = [test_metrics[name] for name in metric_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metric_names, metric_values, color=["#2563eb", "#0f766e", "#7c3aed", "#ca8a04", "#dc2626"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Logistic Regression Test Metrics")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.4f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COLUMN].values)
    class_names = label_encoder.classes_

    # Tách dữ liệu giống hệt với train_model1.py để so sánh công bằng
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Bắt đầu huấn luyện mô hình Logistic Regression...")
    model = LogisticRegression(random_state=RANDOM_SEED, max_iter=2000, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(model, f)

    y_pred = model.predict(X_test_scaled)
    test_metrics = compute_metrics(y_test, y_pred)
    
    report_dict = classification_report(y_test, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    plot_confusion_matrix(cm, class_names, PLOT_CM_PATH)
    plot_test_metrics(test_metrics, PLOT_TEST_METRICS_PATH)

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)
    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"test_metrics": test_metrics}, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("TÓM TẮT KẾT QUẢ TEST - LOGISTIC REGRESSION")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print("\n✅ Đã lưu kết quả tại: ", REPORT_DIR.resolve())

if __name__ == "__main__":
    main()
