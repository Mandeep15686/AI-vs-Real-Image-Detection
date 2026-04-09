"""
DWDM Project — predict.py
AI-Generated vs Real Image Detection — Inference on New Images
=====================================================================
Usage:
    # Single image
    python predict.py --image path/to/image.jpg

    # Entire folder
    python predict.py --folder path/to/images/

    # Specify model explicitly
    python predict.py --image photo.jpg --model resnet50   (default)
    python predict.py --image photo.jpg --model rf

    # Save results to CSV
    python predict.py --folder my_images/ --output results.csv

Requirements:
    pip install torch torchvision Pillow scikit-learn numpy
"""

import os
import sys
import argparse
import json
import csv
import time
import numpy as np
from pathlib import Path
from PIL import Image

# ─────────────────────────────────────────────────────────────────
# PATHS  ← update if you moved the models folder
# ─────────────────────────────────────────────────────────────────
MODELS_DIR = r"C:\Users\shash\Downloads\DATASET\DATASET\models"
IMG_SIZE   = 224
CLASSES    = ["ai", "real"]
# ─────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ══════════════════════════════════════════════════════════════════
# 1.  IMPORTS (graceful degradation)
# ══════════════════════════════════════════════════════════════════
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    from torchvision.models import ResNet50_Weights
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import pickle
    from sklearn.preprocessing import StandardScaler
    SK_OK = True
except ImportError:
    SK_OK = False


# ══════════════════════════════════════════════════════════════════
# 2.  HELPERS — feature extraction for Random Forest
# ══════════════════════════════════════════════════════════════════
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


def extract_features(img_path: str) -> np.ndarray:
    """Handcrafted feature vector — same as used during RF training."""
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
        _entropy(gray.flatten()),
        _laplacian_var(gray),
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════
# 3.  MODEL LOADERS
# ══════════════════════════════════════════════════════════════════
def _get_device():
    if not TORCH_OK:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_resnet50_arch(num_classes=2):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model


def load_resnet50(models_dir: str):
    """Load the best saved ResNet50 checkpoint."""
    if not TORCH_OK:
        raise RuntimeError("PyTorch not installed. Run: pip install torch torchvision")

    ckpt_path = os.path.join(models_dir, "resnet50_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model not found: {ckpt_path}\nRun train_model.py first.")

    device = _get_device()
    ckpt   = torch.load(ckpt_path, map_location=device)

    # Rebuild architecture and load weights
    num_classes = len(ckpt.get("class_to_idx", {"ai":0,"real":1}))
    model = _build_resnet50_arch(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Recreate class index mapping
    class_to_idx = ckpt.get("class_to_idx", {"ai":0,"real":1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print(f"  [✓] ResNet50 loaded  (Val AUC={ckpt.get('val_auc',0):.4f}, "
          f"device={device})")
    return model, idx_to_class, device


def load_random_forest(models_dir: str):
    """Load saved Random Forest and scaler."""
    if not SK_OK:
        raise RuntimeError("scikit-learn not installed. Run: pip install scikit-learn")

    rf_path  = os.path.join(models_dir, "random_forest.pkl")
    sc_path  = os.path.join(models_dir, "rf_scaler.pkl")

    for p in [rf_path, sc_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}\nRun train_model.py first.")

    with open(rf_path, "rb") as f: rf = pickle.load(f)
    with open(sc_path, "rb") as f: scaler = pickle.load(f)
    print("  [✓] Random Forest loaded")
    return rf, scaler


# ══════════════════════════════════════════════════════════════════
# 4.  SINGLE IMAGE PREDICTION
# ══════════════════════════════════════════════════════════════════
def _get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])


def predict_resnet50(image_path: str, model, idx_to_class: dict, device) -> dict:
    """Run ResNet50 inference on a single image."""
    tf = _get_val_transform()
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
    tensor = tf(img_rgb).unsqueeze(0).to(device)   # [1, 3, H, W]

    with torch.no_grad():
        logits = model(tensor)                       # [1, num_classes]
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx   = int(probs.argmax())
    pred_label = idx_to_class.get(pred_idx, str(pred_idx))
    confidence = float(probs[pred_idx])

    return {
        "model":       "ResNet50",
        "prediction":  pred_label.upper(),
        "confidence":  round(confidence * 100, 2),
        "prob_ai":     round(float(probs[0]) * 100, 2),
        "prob_real":   round(float(probs[1]) * 100, 2),
    }


def predict_rf(image_path: str, rf, scaler) -> dict:
    """Run Random Forest inference on a single image."""
    feat   = extract_features(image_path).reshape(1, -1)
    feat_s = scaler.transform(feat)
    pred   = rf.predict(feat_s)[0]
    probs  = rf.predict_proba(feat_s)[0]   # [prob_ai, prob_real]

    label      = CLASSES[pred]
    confidence = float(probs[pred])

    return {
        "model":       "RandomForest",
        "prediction":  label.upper(),
        "confidence":  round(confidence * 100, 2),
        "prob_ai":     round(float(probs[0]) * 100, 2),
        "prob_real":   round(float(probs[1]) * 100, 2),
    }


def predict_ensemble(image_path: str,
                     rn_model=None, rn_idx_to_class=None, rn_device=None,
                     rf_model=None, rf_scaler=None) -> dict:
    """
    Soft-voting ensemble: average softmax probabilities from both models.
    Uses whichever models are loaded.
    """
    prob_ai_list   = []
    prob_real_list = []

    if rn_model is not None:
        r = predict_resnet50(image_path, rn_model, rn_idx_to_class, rn_device)
        prob_ai_list.append(r["prob_ai"])
        prob_real_list.append(r["prob_real"])

    if rf_model is not None:
        r = predict_rf(image_path, rf_model, rf_scaler)
        prob_ai_list.append(r["prob_ai"])
        prob_real_list.append(r["prob_real"])

    avg_ai   = np.mean(prob_ai_list)
    avg_real = np.mean(prob_real_list)
    pred     = "AI" if avg_ai > avg_real else "REAL"
    conf     = max(avg_ai, avg_real)

    return {
        "model":       "Ensemble",
        "prediction":  pred,
        "confidence":  round(float(conf), 2),
        "prob_ai":     round(float(avg_ai), 2),
        "prob_real":   round(float(avg_real), 2),
    }


# ══════════════════════════════════════════════════════════════════
# 5.  DISPLAY RESULT
# ══════════════════════════════════════════════════════════════════
VERDICT_WIDTH = 60

def display_result(image_path: str, result: dict, show_bar: bool = True):
    fname = Path(image_path).name
    pred  = result["prediction"]
    conf  = result["confidence"]
    color = "\033[91m" if pred == "AI" else "\033[92m"   # red / green
    reset = "\033[0m"

    print(f"\n{'─'*VERDICT_WIDTH}")
    print(f"  Image      : {fname}")
    print(f"  Model      : {result['model']}")
    print(f"  Verdict    : {color}{pred}{reset}")
    print(f"  Confidence : {conf:.1f}%")
    print(f"  AI score   : {result['prob_ai']:.1f}%  |  Real score: {result['prob_real']:.1f}%")

    if show_bar:
        bar_len = 40
        ai_fill   = int(bar_len * result["prob_ai"]   / 100)
        real_fill = int(bar_len * result["prob_real"] / 100)
        ai_bar    = "\033[91m" + "█" * ai_fill   + reset + "░" * (bar_len - ai_fill)
        real_bar  = "\033[92m" + "█" * real_fill + reset + "░" * (bar_len - real_fill)
        print(f"  AI  [{ai_bar}] {result['prob_ai']:.1f}%")
        print(f"  Real[{real_bar}] {result['prob_real']:.1f}%")

    print(f"{'─'*VERDICT_WIDTH}")


# ══════════════════════════════════════════════════════════════════
# 6.  BATCH FOLDER PREDICTION
# ══════════════════════════════════════════════════════════════════
def predict_folder(folder_path: str, predict_fn, output_csv: str = None):
    """Run inference on all images in a folder."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"[ERROR] Folder not found: {folder_path}")
        return

    files = [f for f in folder.rglob("*")
             if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]

    if not files:
        print(f"[WARNING] No images found in {folder_path}")
        return

    print(f"\n  Found {len(files):,} images in {folder_path}")
    print(f"  Running inference ...")

    results = []
    ai_count = real_count = error_count = 0
    t0 = time.time()

    for i, fpath in enumerate(files, 1):
        try:
            result = predict_fn(str(fpath))
            result["image_path"] = str(fpath)
            result["filename"]   = fpath.name
            results.append(result)

            if result["prediction"] == "AI":
                ai_count += 1
            else:
                real_count += 1

            display_result(str(fpath), result, show_bar=False)

        except Exception as e:
            print(f"  [ERROR] {fpath.name}: {e}")
            error_count += 1

        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Progress: {i}/{len(files)} ({elapsed:.0f}s elapsed)")

    # Summary
    total_ok = ai_count + real_count
    print(f"\n{'═'*60}")
    print(f"  BATCH PREDICTION SUMMARY")
    print(f"{'═'*60}")
    print(f"  Total processed : {total_ok}")
    print(f"  AI-Generated    : {ai_count}  ({ai_count/total_ok*100:.1f}% of processed)" if total_ok else "")
    print(f"  Real            : {real_count}  ({real_count/total_ok*100:.1f}% of processed)" if total_ok else "")
    print(f"  Errors          : {error_count}")
    print(f"  Time elapsed    : {time.time()-t0:.1f}s")
    print(f"{'═'*60}")

    # Save CSV
    if output_csv and results:
        fields = ["filename", "model", "prediction", "confidence",
                  "prob_ai", "prob_real", "image_path"]
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  [✓] Results saved → {output_csv}")

    return results


# ══════════════════════════════════════════════════════════════════
# 7.  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="DWDM: Predict whether image(s) are AI-generated or Real",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --image photo.jpg
  python predict.py --image photo.jpg --model rf
  python predict.py --image photo.jpg --model ensemble
  python predict.py --folder my_images/ --output results.csv
        """
    )
    parser.add_argument("--image",  type=str, help="Path to a single image file")
    parser.add_argument("--folder", type=str, help="Path to folder of images")
    parser.add_argument("--model",  type=str, default="resnet50",
                        choices=["resnet50", "rf", "ensemble"],
                        help="Model to use for prediction (default: resnet50)")
    parser.add_argument("--output", type=str, default=None,
                        help="(Batch mode) Save predictions to this CSV file")
    parser.add_argument("--models_dir", type=str, default=MODELS_DIR,
                        help=f"Directory containing saved models (default: {MODELS_DIR})")

    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.print_help()
        sys.exit(0)

    print("\n" + "="*60)
    print("  DWDM — AI vs Real Image Predictor")
    print("="*60)
    print(f"  Model selected: {args.model}")
    print(f"  Models dir    : {args.models_dir}")

    # ── Load requested model(s) ────────────────────────────────────
    rn_model = rn_idx_to_class = rn_device = None
    rf_model = rf_scaler = None

    if args.model in ("resnet50", "ensemble"):
        try:
            rn_model, rn_idx_to_class, rn_device = load_resnet50(args.models_dir)
        except Exception as e:
            print(f"  [!] ResNet50 load failed: {e}")

    if args.model in ("rf", "ensemble"):
        try:
            rf_model, rf_scaler = load_random_forest(args.models_dir)
        except Exception as e:
            print(f"  [!] Random Forest load failed: {e}")

    if rn_model is None and rf_model is None:
        print("\n[ERROR] No models could be loaded. Run train_model.py first.")
        sys.exit(1)

    # ── Build prediction function ─────────────────────────────────
    def predict_fn(image_path: str) -> dict:
        if args.model == "resnet50":
            return predict_resnet50(image_path, rn_model, rn_idx_to_class, rn_device)
        elif args.model == "rf":
            return predict_rf(image_path, rf_model, rf_scaler)
        else:
            return predict_ensemble(image_path,
                                     rn_model, rn_idx_to_class, rn_device,
                                     rf_model, rf_scaler)

    # ── Run prediction ────────────────────────────────────────────
    if args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] Image not found: {args.image}")
            sys.exit(1)
        result = predict_fn(args.image)
        display_result(args.image, result, show_bar=True)

    elif args.folder:
        predict_folder(args.folder, predict_fn, output_csv=args.output)


if __name__ == "__main__":
    main()