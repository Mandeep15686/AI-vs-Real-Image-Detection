"""
DWDM Project - Model Training
AI-Generated vs Real Image Detection
=====================================================================
Dataset path: C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\cleaned_data\\splits

Folder structure expected:
    splits/
        train/
            ai/       ← AI-generated images
            real/     ← Real images
        val/
            ai/
            real/
        test/
            ai/
            real/

Models trained:
    1. ResNet50  (Transfer Learning - Deep CNN)
    2. Random Forest on handcrafted features (traditional ML baseline)

Outputs saved to:  models/
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION  ← only change these
# ─────────────────────────────────────────────────────────────────
BASE_PATH   = r"C:\Users\shash\Downloads\DATASET\DATASET\cleaned_data\splits"
MODELS_DIR  = r"C:\Users\shash\Downloads\DATASET\DATASET\models"
IMG_SIZE    = 224          # input size (pixels)
BATCH_SIZE  = 32
EPOCHS      = 20           # increase for better accuracy (try 30-50)
LR          = 1e-4         # learning rate
NUM_WORKERS = 0            # set to 4+ on Linux; keep 0 on Windows
DEVICE_PREF = "auto"       # "auto" | "cpu" | "cuda" | "mps"
# ─────────────────────────────────────────────────────────────────

TRAIN_DIR = os.path.join(BASE_PATH, "train")
VAL_DIR   = os.path.join(BASE_PATH, "val")
TEST_DIR  = os.path.join(BASE_PATH, "test")
CLASSES   = ["ai", "real"]   # ai=0, real=1

os.makedirs(MODELS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  DWDM — AI vs Real Image Classifier — Training")
print("="*65)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models
    from torchvision.models import ResNet50_Weights
    TORCH_AVAILABLE = True
    print("  [✓] PyTorch available")
except ImportError:
    TORCH_AVAILABLE = False
    print("  [!] PyTorch not found — deep learning training skipped")
    print("      Install: pip install torch torchvision")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, classification_report,
                                  confusion_matrix, roc_auc_score)
    from sklearn.preprocessing import StandardScaler
    import pickle
    SK_AVAILABLE = True
    print("  [✓] scikit-learn available")
except ImportError:
    SK_AVAILABLE = False
    print("  [!] scikit-learn not found")
    print("      Install: pip install scikit-learn")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL = True
except ImportError:
    MPL = False

from PIL import Image


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — DEVICE SETUP
# ══════════════════════════════════════════════════════════════════
def get_device():
    if DEVICE_PREF == "cpu":
        return torch.device("cpu")
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            print(f"  [✓] GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
            print("  [✓] Apple MPS GPU")
        else:
            dev = torch.device("cpu")
            print("  [i] CPU only (no GPU detected)")
        return dev
    return None


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — DATA TRANSFORMS
# ══════════════════════════════════════════════════════════════════
def get_transforms():
    """ImageNet normalisation + augmentation for training."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — RESNET50 MODEL DEFINITION
# ══════════════════════════════════════════════════════════════════
def build_resnet50(num_classes=2, freeze_backbone=True):
    """
    Load pretrained ResNet50, replace final FC layer for binary classification.
    freeze_backbone=True : only train the classifier head (faster, less data needed)
    freeze_backbone=False: fine-tune all layers (slower, better with large dataset)
    """
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze last two residual blocks for fine-tuning
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Replace final classifier
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    return model


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx+1}/{len(loader)}  "
                  f"Loss: {loss.item():.4f}  Acc: {correct/total:.4f}")

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = total = 0
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs) if SK_AVAILABLE else 0.0
    return running_loss / total, correct / total, auc


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — PLOT TRAINING CURVES
# ══════════════════════════════════════════════════════════════════
def plot_training_curves(history: dict, save_dir: str):
    if not MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss", color="#E45C5C")
    axes[0].plot(history["val_loss"],   label="Val Loss",   color="#4C9BE8")
    axes[0].set_title("Loss Curve", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Acc", color="#E45C5C")
    axes[1].plot(history["val_acc"],   label="Val Acc",   color="#4C9BE8")
    axes[1].set_title("Accuracy Curve", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [✓] Training curves saved → {path}")


def plot_confusion_matrix(cm, classes, save_dir, title="Confusion Matrix"):
    if not MPL:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes,
           title=title, ylabel="True label", xlabel="Predicted label")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    fname = title.lower().replace(" ", "_") + ".png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# SECTION 7 — HANDCRAFTED FEATURES (for Random Forest)
# ══════════════════════════════════════════════════════════════════
def extract_features(img_path: str) -> np.ndarray:
    """
    Extract statistical features from a single image for traditional ML.
    Returns a 1D feature vector.
    """
    with Image.open(img_path) as img:
        img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32)

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    gray = 0.299*r + 0.587*g + 0.114*b

    features = [
        # Channel means & stds
        r.mean(), r.std(), g.mean(), g.std(), b.mean(), b.std(),
        # Brightness
        gray.mean(), gray.std(),
        # Channel ratios
        r.mean() / (g.mean() + 1e-6),
        b.mean() / (g.mean() + 1e-6),
        # Percentiles (texture proxy)
        np.percentile(gray, 25), np.percentile(gray, 75),
        np.percentile(gray, 75) - np.percentile(gray, 25),  # IQR
        # Histogram entropy (complexity)
        _entropy(gray.flatten()),
        # High-frequency energy (Laplacian variance — sharpness)
        _laplacian_var(gray),
    ]

    return np.array(features, dtype=np.float32)


def _entropy(arr: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(arr, bins=bins, range=(0, 255), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist + 1e-12)))


def _laplacian_var(gray: np.ndarray) -> float:
    """Variance of Laplacian — proxy for image sharpness."""
    # Manual 3×3 Laplacian convolution
    kernel = np.array([[0, 1, 0],
                        [1,-4, 1],
                        [0, 1, 0]], dtype=np.float32)
    # Simple convolution (valid region)
    h, w = gray.shape
    lap = np.zeros((h-2, w-2), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            lap += kernel[i, j] * gray[i:h-2+i, j:w-2+j]
    return float(lap.var())


def load_features_from_folder(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load features for all images in train/val/test split."""
    X_list, y_list = [], []
    split_dir = os.path.join(BASE_PATH, split)

    for label_idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"  [WARNING] Missing: {cls_dir}")
            continue
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"    [{split}] {cls}: {len(files):,} images")

        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            try:
                feat = extract_features(fpath)
                X_list.append(feat)
                y_list.append(label_idx)
            except Exception:
                pass

    return np.array(X_list), np.array(y_list)


# ══════════════════════════════════════════════════════════════════
# SECTION 8 — RANDOM FOREST TRAINING
# ══════════════════════════════════════════════════════════════════
def train_random_forest():
    print("\n" + "─"*65)
    print("  TRAINING: Random Forest (Handcrafted Features)")
    print("─"*65)

    print("\n  Extracting features — Train ...")
    X_train, y_train = load_features_from_folder("train")
    print("  Extracting features — Val ...")
    X_val, y_val = load_features_from_folder("val")
    print("  Extracting features — Test ...")
    X_test, y_test = load_features_from_folder("test")

    if len(X_train) == 0:
        print("  [ERROR] No training data found.")
        return

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    print("\n  Training Random Forest ...")
    t0 = time.time()
    rf.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Evaluate
    for split_name, X, y in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
        preds = rf.predict(X)
        probs = rf.predict_proba(X)[:, 1]
        acc   = accuracy_score(y, preds)
        auc   = roc_auc_score(y, probs)
        print(f"\n  [{split_name}] Accuracy: {acc:.4f}  AUC: {auc:.4f}")
        print(classification_report(y, preds, target_names=CLASSES))

        cm = confusion_matrix(y, preds)
        plot_confusion_matrix(cm, CLASSES, MODELS_DIR,
                               title=f"RF Confusion Matrix {split_name}")

    # Feature importance
    feat_names = ["r_mean","r_std","g_mean","g_std","b_mean","b_std",
                  "gray_mean","gray_std","r_g_ratio","b_g_ratio",
                  "pct25","pct75","iqr","entropy","laplacian_var"]
    importance = list(zip(feat_names, rf.feature_importances_))
    importance.sort(key=lambda x: x[1], reverse=True)
    print("\n  Feature Importances (top 10):")
    for fname, imp in importance[:10]:
        print(f"    {fname:<20} {imp:.4f}")

    # Save model
    rf_path     = os.path.join(MODELS_DIR, "random_forest.pkl")
    scaler_path = os.path.join(MODELS_DIR, "rf_scaler.pkl")
    with open(rf_path,     "wb") as f: pickle.dump(rf, f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)
    print(f"\n  [✓] Random Forest saved → {rf_path}")
    print(f"  [✓] Scaler saved        → {scaler_path}")

    return rf, scaler


# ══════════════════════════════════════════════════════════════════
# SECTION 9 — RESNET50 TRAINING
# ══════════════════════════════════════════════════════════════════
def train_resnet50(device):
    print("\n" + "─"*65)
    print("  TRAINING: ResNet50 (Transfer Learning)")
    print("─"*65)

    train_tf, val_tf = get_transforms()

    # Datasets
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tf)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=val_tf)

    print(f"\n  Train: {len(train_ds):,} images | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    print(f"  Class → index map: {train_ds.class_to_idx}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = build_resnet50(num_classes=2, freeze_backbone=True).to(device)

    # Separate LRs: backbone gets 10× smaller LR
    backbone_params = [p for name, p in model.named_parameters()
                       if "fc" not in name and p.requires_grad]
    head_params     = model.fc.parameters()
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": LR * 0.1},
        {"params": head_params,     "lr": LR},
    ], weight_decay=1e-4)

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Label smoothing cross-entropy (reduces overconfidence)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[], "val_auc":[]}
    best_val_auc = 0.0
    best_model_path = os.path.join(MODELS_DIR, "resnet50_best.pth")

    print(f"\n  Training for {EPOCHS} epochs on {device} ...")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Phase 1 (epoch 5): unfreeze full backbone for fine-tuning
        if epoch == 6:
            print("\n  [Phase 2] Unfreezing full backbone for fine-tuning ...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=LR * 0.05, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=EPOCHS - epoch, eta_min=1e-7)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  Epoch [{epoch:02d}/{EPOCHS}] "
              f"TrLoss={train_loss:.4f} TrAcc={train_acc:.4f} | "
              f"VLoss={val_loss:.4f}  VAcc={val_acc:.4f}  VAUC={val_auc:.4f}  "
              f"({elapsed:.0f}s)")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_auc":    val_auc,
                "val_acc":    val_acc,
                "class_to_idx": train_ds.class_to_idx,
                "img_size":   IMG_SIZE,
            }, best_model_path)
            print(f"  ✓ Best model saved (AUC={val_auc:.4f})")

    # ── Test Evaluation ────────────────────────────────────────────
    print("\n  Loading best model for test evaluation ...")
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion, device)
    print(f"\n  ── TEST RESULTS ──────────────────────────────────")
    print(f"     Accuracy : {test_acc:.4f}")
    print(f"     AUC-ROC  : {test_auc:.4f}")
    print(f"     Loss     : {test_loss:.4f}")

    # Detailed classification report
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    class_names  = [idx_to_class[i] for i in range(len(CLASSES))]
    print("\n" + classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, MODELS_DIR, title="ResNet50 Confusion Matrix Test")

    # Save training history
    plot_training_curves(history, MODELS_DIR)
    with open(os.path.join(MODELS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save final (last epoch) model too
    final_path = os.path.join(MODELS_DIR, "resnet50_final.pth")
    torch.save(model.state_dict(), final_path)

    print(f"\n  [✓] Best model  → {best_model_path}")
    print(f"  [✓] Final model → {final_path}")

    return model, train_ds.class_to_idx


# ══════════════════════════════════════════════════════════════════
# SECTION 10 — MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    # Verify paths
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            d = os.path.join(BASE_PATH, split, cls)
            if not os.path.exists(d):
                print(f"[ERROR] Missing folder: {d}")
                print("  Make sure step6_train_val_test_split.py has been run.")
                sys.exit(1)
    print(f"\n  Dataset path: {BASE_PATH}")
    print(f"  Models will be saved to: {MODELS_DIR}")

    # ── Random Forest ──────────────────────────────────────────────
    if SK_AVAILABLE:
        train_random_forest()
    else:
        print("\n  [SKIP] Random Forest — scikit-learn not installed")

    # ── ResNet50 ───────────────────────────────────────────────────
    if TORCH_AVAILABLE:
        device = get_device()
        train_resnet50(device)
    else:
        print("\n  [SKIP] ResNet50 — PyTorch not installed")

    print("\n" + "="*65)
    print("  Training complete! Models saved to:", MODELS_DIR)
    print("  Next: run predict.py to classify new images")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()