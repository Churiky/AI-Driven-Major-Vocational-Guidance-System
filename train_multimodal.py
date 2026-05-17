import copy
import json
import pickle
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
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
from torch.utils.data import DataLoader, Dataset


RANDOM_SEED = 42
FEATURES = [
    "Toan",
    "Van",
    "Anh",
    "Ly",
    "Hoa",
    "Sinh",
    "Su",
    "Dia",
    "GDCD",
    "KHTN",
    "KHXH",
    "R",
    "I",
    "A",
    "S",
    "E",
    "C",
]
TARGET_COLUMN = "Major"
DATA_PATH = Path("data") / "synthetic_students.csv"
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports") / "train_multimodal"
CHECKPOINT_PATH = MODEL_DIR / "multimodal_transformer.pth"
SCALER_PATH = MODEL_DIR / "multimodal_transformer_scaler.pkl"
CLASSES_PATH = MODEL_DIR / "multimodal_transformer_classes.pkl"
HISTORY_PATH = REPORT_DIR / "training_history.csv"
REPORT_JSON_PATH = REPORT_DIR / "classification_report.json"
REPORT_TXT_PATH = REPORT_DIR / "classification_report.txt"
METRICS_JSON_PATH = REPORT_DIR / "test_metrics.json"
PLOT_CURVES_PATH = REPORT_DIR / "training_curves.png"
PLOT_CM_PATH = REPORT_DIR / "confusion_matrix.png"
PLOT_TEST_METRICS_PATH = REPORT_DIR / "test_metrics.png"
TRAIN_BATCH_SIZE = 256
MAX_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 10
MODEL_D_MODEL = 64
MODEL_NHEAD = 4
MODEL_NUM_LAYERS = 3
MODEL_FF_MULTIPLIER = 3
MODEL_DROPOUT = 0.15


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_directories() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False",
)


class CareerDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


from src.transformer_model import MultimodalCareerTransformer

def compute_class_weights(y_train: np.ndarray) -> torch.Tensor:
    class_counts = np.bincount(y_train)
    weights = len(y_train) / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor,
        device: torch.device,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        label_smoothing: float = 0.02,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=label_smoothing,
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
        self.history = {
            "epoch": [],
            "lr": [],
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_macro_f1": [],
            "val_macro_f1": [],
            "train_weighted_f1": [],
            "val_weighted_f1": [],
            "train_macro_precision": [],
            "val_macro_precision": [],
            "train_macro_recall": [],
            "val_macro_recall": [],
        }

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict:
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            else:
                with torch.inference_mode():
                    logits = self.model(X_batch)
                    loss = self.criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
            all_targets.extend(y_batch.detach().cpu().numpy())

        metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
        metrics["loss"] = total_loss / len(loader.dataset)
        return metrics

    def train(
        self,
        epochs: int,
        patience: int,
        checkpoint_path: Path,
    ) -> dict:
        best_state = copy.deepcopy(self.model.state_dict())
        best_epoch = 0
        best_metrics = None
        best_score = float("-inf")
        wait = 0

        print(f"Bat dau huan luyen Multimodal Transformer voi toi da {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(self.train_loader, training=True)
            val_metrics = self._run_epoch(self.val_loader, training=False)
            self.scheduler.step(val_metrics["macro_f1"])

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["epoch"].append(epoch)
            self.history["lr"].append(current_lr)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_accuracy"].append(train_metrics["accuracy"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["train_macro_f1"].append(train_metrics["macro_f1"])
            self.history["val_macro_f1"].append(val_metrics["macro_f1"])
            self.history["train_weighted_f1"].append(train_metrics["weighted_f1"])
            self.history["val_weighted_f1"].append(val_metrics["weighted_f1"])
            self.history["train_macro_precision"].append(
                train_metrics["macro_precision"]
            )
            self.history["val_macro_precision"].append(val_metrics["macro_precision"])
            self.history["train_macro_recall"].append(train_metrics["macro_recall"])
            self.history["val_macro_recall"].append(val_metrics["macro_recall"])

            improved = (
                val_metrics["macro_f1"] > best_score + 1e-4
                or (
                    best_metrics is not None
                    and abs(val_metrics["macro_f1"] - best_score) <= 1e-4
                    and val_metrics["loss"] < best_metrics["loss"]
                )
            )

            if improved:
                best_score = val_metrics["macro_f1"]
                best_epoch = epoch
                best_metrics = val_metrics
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_state,
                        "val_metrics": best_metrics,
                        "config": {
                            "num_features": len(FEATURES),
                            "random_seed": RANDOM_SEED,
                        },
                    },
                    checkpoint_path,
                )
            else:
                wait += 1

            if epoch == 1 or epoch % 5 == 0:
                print(
                    f"Epoch {epoch:03d} | "
                    f"train_loss={train_metrics['loss']:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"val_acc={val_metrics['accuracy']:.4f} | "
                    f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
                    f"lr={current_lr:.6f}"
                )

            if wait >= patience:
                print(
                    f"Early stopping tai epoch {epoch}. "
                    f"Best epoch la {best_epoch} voi val_macro_f1={best_score:.4f}."
                )
                break

        self.model.load_state_dict(best_state)
        return {
            "best_epoch": best_epoch,
            "best_val_metrics": best_metrics,
        }


def plot_training_curves(history_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plots = [
        ("Loss", "train_loss", "val_loss"),
        ("Accuracy", "train_accuracy", "val_accuracy"),
        ("Macro F1", "train_macro_f1", "val_macro_f1"),
        ("Weighted F1", "train_weighted_f1", "val_weighted_f1"),
        ("Macro Precision", "train_macro_precision", "val_macro_precision"),
        ("Macro Recall", "train_macro_recall", "val_macro_recall"),
    ]

    for ax, (title, train_col, val_col) in zip(axes.flatten(), plots):
        ax.plot(history_df["epoch"], history_df[train_col], label=f"Train {title}")
        ax.plot(history_df["epoch"], history_df[val_col], label=f"Val {title}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray, class_names: np.ndarray, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title("Transformer Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_test_metrics(test_metrics: dict, output_path: Path) -> None:
    metric_names = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
    ]
    metric_values = [test_metrics[name] for name in metric_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metric_names, metric_values, color=["#2563eb", "#0f766e", "#7c3aed", "#ca8a04", "#dc2626"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Final Test Metrics")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, metric_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = TRAIN_BATCH_SIZE,
):
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        CareerDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        CareerDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        CareerDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader, test_loader


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, np.array(all_targets), np.array(all_preds)


def main() -> None:
    set_seed(RANDOM_SEED)
    ensure_directories()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay dataset tai {DATA_PATH.resolve()}. "
            "Script se dung de tranh train tren du lieu khong hop le."
        )

    df = pd.read_csv(DATA_PATH)
    missing_columns = [col for col in FEATURES + [TARGET_COLUMN] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset dang thieu cac cot: {missing_columns}")

    X = df[FEATURES].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COLUMN].values)
    class_names = label_encoder.classes_

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    class_weights = compute_class_weights(y_train)
    train_loader, val_loader, test_loader = build_dataloaders(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        X_test_scaled,
        y_test,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Su dung thiet bi: {device}")
    print(
        f"Train/Val/Test sizes: {len(X_train_scaled)}/{len(X_val_scaled)}/{len(X_test_scaled)}"
    )

    model = MultimodalCareerTransformer(
        num_acad_features=11,
        num_psych_features=6,
        num_classes=len(class_names),
        d_model=MODEL_D_MODEL,
        nhead=MODEL_NHEAD,
        num_layers=MODEL_NUM_LAYERS,
        ff_multiplier=MODEL_FF_MULTIPLIER,
        dropout=MODEL_DROPOUT,
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=device,
        lr=5e-4,
        label_smoothing=0.01,
    )
    training_summary = trainer.train(
        epochs=MAX_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        checkpoint_path=CHECKPOINT_PATH,
    )

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(CLASSES_PATH, "wb") as f:
        pickle.dump(class_names, f)

    history_df = pd.DataFrame(trainer.history)
    history_df.to_csv(HISTORY_PATH, index=False)

    test_metrics, y_true, y_pred = evaluate_model(
        trainer.model,
        test_loader,
        trainer.criterion,
        device,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)
    plot_training_curves(history_df, PLOT_CURVES_PATH)
    plot_confusion_matrix(cm, class_names, PLOT_CM_PATH)
    plot_test_metrics(test_metrics, PLOT_TEST_METRICS_PATH)

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)
    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": training_summary["best_epoch"],
                "best_val_metrics": training_summary["best_val_metrics"],
                "test_metrics": test_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("TOM TAT KET QUA TEST")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nBEST VALIDATION")
    print(
        f"best_epoch: {training_summary['best_epoch']} | "
        f"val_macro_f1: {training_summary['best_val_metrics']['macro_f1']:.4f} | "
        f"val_accuracy: {training_summary['best_val_metrics']['accuracy']:.4f}"
    )

    print("\nBAO CAO PHAN LOAI CHI TIET")
    print(report_text)

    print("\nARTEFACT DA LUU")
    print(f"- Checkpoint: {CHECKPOINT_PATH.resolve()}")
    print(f"- Scaler: {SCALER_PATH.resolve()}")
    print(f"- Classes: {CLASSES_PATH.resolve()}")
    print(f"- History CSV: {HISTORY_PATH.resolve()}")
    print(f"- Training curves: {PLOT_CURVES_PATH.resolve()}")
    print(f"- Confusion matrix: {PLOT_CM_PATH.resolve()}")
    print(f"- Test metrics chart: {PLOT_TEST_METRICS_PATH.resolve()}")


if __name__ == "__main__":
    main()
