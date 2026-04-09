"""
DWDM Project — evaluate_model.py
Full evaluation of saved models on the TEST set
=====================================================================
Generates:
    · Accuracy / Precision / Recall / F1 / AUC-ROC
    · Confusion matrix plot
    · ROC curve plot
    · Per-image prediction CSV

Usage:
    python evaluate_model.py
    python evaluate_model.py --model rf
    python evaluate_model.py --model ensemble
"""

import os
import sys
import argparse
import csv
import json
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
SPLITS_DIR  = r"C:\Users\shash\Downloads\DATASET\DATASET\cleaned_data\splits"
MODELS_DIR  = r"C:\Users\shash\Downloads\DATASET\DATASET\models"
IMG_SIZE    = 224
CLASSES     = ["ai", "real"]
# ─────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models, datasets
    from torchvision.models import ResNet50_Weights
    from torch.utils.data import DataLoader
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        roc_auc_score, roc_curve, average_precision_score
    )
    from sklearn.preprocessing import StandardScaler
    import pickle
    SK_OK = True
except ImportError:
    SK_OK = False
    print("[ERROR] scikit-learn required: pip install scikit-learn")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL = True
except ImportError:
    MPL = False

from PIL import Image


# ── Feature extraction (same as training) ─────────────────────────
def _entropy(arr, bins=64):
    hist, _ = np.histogram(arr, bins=bins, range=(0, 255), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist + 1e-12)))

def _laplacian_var(gray):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = gray.shape
    lap = np.zeros((h-2, w-2), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            lap += kernel[i, j] * gray[i:h-2+i, j:w-2+j]
    return float(lap.var())

def extract_features(img_path):
    with Image.open(img_path) as img:
        img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    gray = 0.299*r + 0.587*g + 0.114*b
    return np.array([
        r.mean(), r.std(), g.mean(), g.std(), b.mean(), b.std(),
        gray.mean(), gray.std(),
        r.mean() / (g.mean() + 1e-6),
        b.mean() / (g.mean() + 1e-6),
        np.percentile(gray, 25), np.percentile(gray, 75),
        np.percentile(gray, 75) - np.percentile(gray, 25),
        _entropy(gray.flatten()), _laplacian_var(gray),
    ], dtype=np.float32)


# ── Model architecture ─────────────────────────────────────────────
def _build_resnet50():
    model = models.resnet50(weights=None)
    in_f  = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(in_f, 256),
        nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2),
    )
    return model


# ══════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    if not MPL: return
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           title=title, ylabel="True", xlabel="Predicted")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    fontsize=14, color="white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [✓] Saved: {save_path}")


def plot_roc_curve(y_true, y_score, save_path, title="ROC Curve"):
    if not MPL: return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#2E75B6", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0,1], [0,1], "k--", lw=1)
    ax.set(xlim=[0,1], ylim=[0,1.02],
           xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(loc="lower right")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2E75B6")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [✓] Saved: {save_path}")


def plot_confidence_histogram(confidences_correct, confidences_wrong, save_path):
    """Show confidence distribution for correct vs wrong predictions."""
    if not MPL: return
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(50, 100, 30)
    ax.hist(confidences_correct, bins=bins, alpha=0.7, color="#2E75B6", label="Correct predictions")
    ax.hist(confidences_wrong,   bins=bins, alpha=0.7, color="#E45C5C", label="Wrong predictions")
    ax.set(title="Confidence Distribution", xlabel="Confidence (%)", ylabel="Count")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [✓] Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════
# EVALUATE RESNET50
# ══════════════════════════════════════════════════════════════════
def evaluate_resnet50():
    if not TORCH_OK:
        print("[SKIP] PyTorch not available")
        return

    ckpt_path = os.path.join(MODELS_DIR, "resnet50_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Not found: {ckpt_path}")
        return

    device = (torch.device("cuda") if torch.cuda.is_available()
               else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
               else torch.device("cpu"))

    ckpt  = torch.load(ckpt_path, map_location=device)
    model = _build_resnet50().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    class_to_idx = ckpt.get("class_to_idx", {"ai": 0, "real": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    test_dir = os.path.join(SPLITS_DIR, "test")
    test_ds  = datasets.ImageFolder(test_dir, transform=tf)
    loader   = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    print(f"\n  Evaluating ResNet50 on {len(test_ds):,} test images ...")

    all_labels, all_preds, all_probs = [], [], []
    per_image_rows = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            out    = model(images)
            probs  = torch.softmax(out, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1])   # prob of "real"

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_score = np.array(all_probs)

    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_score)
    ap   = average_precision_score(y_true, y_score)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  ══ ResNet50 TEST EVALUATION ══")
    print(f"  Accuracy         : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  AUC-ROC          : {auc:.4f}")
    print(f"  Avg Precision    : {ap:.4f}")
    print(f"  True Positives   : {tp}")
    print(f"  True Negatives   : {tn}")
    print(f"  False Positives  : {fp}  (Real predicted as AI)")
    print(f"  False Negatives  : {fn}  (AI predicted as Real)")
    print()
    class_names = [idx_to_class.get(i, str(i)) for i in range(len(class_to_idx))]
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Save plots
    plot_confusion_matrix(cm, class_names,
                           os.path.join(MODELS_DIR, "eval_resnet50_cm.png"),
                           title="ResNet50 — Test Confusion Matrix")
    plot_roc_curve(y_true, y_score,
                   os.path.join(MODELS_DIR, "eval_resnet50_roc.png"),
                   title="ResNet50 — ROC Curve (Test)")

    # Save report JSON
    report = {
        "model":    "ResNet50",
        "accuracy": round(acc, 4),
        "auc_roc":  round(auc, 4),
        "avg_precision": round(ap, 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }
    with open(os.path.join(MODELS_DIR, "eval_resnet50.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"  [✓] Report → {os.path.join(MODELS_DIR, 'eval_resnet50.json')}")
    return report


# ══════════════════════════════════════════════════════════════════
# EVALUATE RANDOM FOREST
# ══════════════════════════════════════════════════════════════════
def evaluate_rf():
    rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    sc_path = os.path.join(MODELS_DIR, "rf_scaler.pkl")
    if not os.path.exists(rf_path):
        print(f"[ERROR] Not found: {rf_path}")
        return

    with open(rf_path, "rb") as f: rf = pickle.load(f)
    with open(sc_path, "rb") as f: scaler = pickle.load(f)

    test_dir = os.path.join(SPLITS_DIR, "test")
    X_list, y_list, paths = [], [], []

    print("\n  Loading test features for Random Forest ...")
    for label_idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(test_dir, cls)
        files   = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
        print(f"    {cls}: {len(files):,}")
        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            try:
                X_list.append(extract_features(fpath))
                y_list.append(label_idx)
                paths.append(fpath)
            except Exception:
                pass

    X = scaler.transform(np.array(X_list))
    y_true  = np.array(y_list)
    y_pred  = rf.predict(X)
    y_score = rf.predict_proba(X)[:, 1]

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    ap  = average_precision_score(y_true, y_score)
    cm  = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  ══ Random Forest TEST EVALUATION ══")
    print(f"  Accuracy         : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  AUC-ROC          : {auc:.4f}")
    print(f"  Avg Precision    : {ap:.4f}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print()
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    plot_confusion_matrix(cm, CLASSES,
                           os.path.join(MODELS_DIR, "eval_rf_cm.png"),
                           title="Random Forest — Test Confusion Matrix")
    plot_roc_curve(y_true, y_score,
                   os.path.join(MODELS_DIR, "eval_rf_roc.png"),
                   title="Random Forest — ROC Curve (Test)")

    # Confidence histogram
    confidences = rf.predict_proba(X).max(axis=1) * 100
    correct_mask = y_pred == y_true
    plot_confidence_histogram(
        confidences[correct_mask], confidences[~correct_mask],
        os.path.join(MODELS_DIR, "eval_rf_confidence.png")
    )

    report = {
        "model":    "RandomForest",
        "accuracy": round(acc, 4),
        "auc_roc":  round(auc, 4),
        "avg_precision": round(ap, 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }
    with open(os.path.join(MODELS_DIR, "eval_rf.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"  [✓] Report → {os.path.join(MODELS_DIR, 'eval_rf.json')}")
    return report


# ══════════════════════════════════════════════════════════════════
# COMPARE MODELS SIDE-BY-SIDE
# ══════════════════════════════════════════════════════════════════
def compare_models():
    """Load both eval JSONs and print a side-by-side comparison table."""
    rn_path = os.path.join(MODELS_DIR, "eval_resnet50.json")
    rf_path = os.path.join(MODELS_DIR, "eval_rf.json")

    reports = {}
    for name, path in [("ResNet50", rn_path), ("RandomForest", rf_path)]:
        if os.path.exists(path):
            with open(path) as f:
                reports[name] = json.load(f)

    if not reports:
        return

    print("\n" + "═"*60)
    print("  MODEL COMPARISON — TEST SET")
    print("═"*60)
    print(f"  {'Metric':<20} ", end="")
    for name in reports:
        print(f"{name:>15}", end="")
    print()
    print("  " + "─"*55)

    for metric in ["accuracy", "auc_roc", "avg_precision"]:
        print(f"  {metric:<20} ", end="")
        for name, r in reports.items():
            val = r.get(metric, "-")
            print(f"{val:>15.4f}" if isinstance(val, float) else f"{val:>15}", end="")
        print()

    print("═"*60)

    if MPL and len(reports) == 2:
        names   = list(reports.keys())
        metrics = ["accuracy", "auc_roc", "avg_precision"]
        vals    = [[reports[n][m] for m in metrics] for n in names]

        x = np.arange(len(metrics))
        fig, ax = plt.subplots(figsize=(8, 4))
        w = 0.35
        colors = ["#2E75B6", "#E45C5C"]
        for i, (name, v) in enumerate(zip(names, vals)):
            bars = ax.bar(x + i*w - w/2, v, width=w, label=name, color=colors[i], alpha=0.85)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{bar.get_height():.3f}", ha="center", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(["Accuracy", "AUC-ROC", "Avg Precision"])
        ax.set_ylim(0, 1.15)
        ax.set_title("Model Comparison — Test Set", fontsize=12, fontweight="bold")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        save_path = os.path.join(MODELS_DIR, "model_comparison.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  [✓] Comparison chart → {save_path}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on test set")
    parser.add_argument("--model", default="all",
                        choices=["all", "resnet50", "rf"],
                        help="Which model to evaluate (default: all)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  DWDM — Model Evaluation on Test Set")
    print("="*60)
    print(f"  Test data  : {os.path.join(SPLITS_DIR, 'test')}")
    print(f"  Models dir : {MODELS_DIR}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.model in ("all", "resnet50"):
        evaluate_resnet50()

    if args.model in ("all", "rf"):
        evaluate_rf()

    compare_models()

    print("\n  Evaluation complete. Check", MODELS_DIR, "for plots and JSON reports.\n")


if __name__ == "__main__":
    main()