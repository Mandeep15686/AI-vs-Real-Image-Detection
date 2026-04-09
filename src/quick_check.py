"""
DWDM Project — quick_check.py
Instant drag-and-drop style image checker (no CLI arguments needed)
=====================================================================
Just run this script and it will ask you for an image path interactively.
Perfect for demos and quick testing.

Usage:
    python quick_check.py
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

# ─────────────────────────────────────────────────────────────────
MODELS_DIR = r"C:\Users\shash\Downloads\DATASET\DATASET\models"
IMG_SIZE   = 224
CLASSES    = ["ai", "real"]
# ─────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

try:
    import torch, torch.nn as nn
    from torchvision import transforms, models
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import pickle
    from sklearn.preprocessing import StandardScaler
    SK_OK = True
except ImportError:
    SK_OK = False


# ── Feature helpers ────────────────────────────────────────────────
def _entropy(arr, bins=64):
    hist, _ = np.histogram(arr, bins=bins, range=(0,255), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist + 1e-12)))

def _laplacian_var(gray):
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    h, w = gray.shape
    lap = np.zeros((h-2, w-2), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            lap += kernel[i,j] * gray[i:h-2+i, j:w-2+j]
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
        r.mean()/(g.mean()+1e-6), b.mean()/(g.mean()+1e-6),
        np.percentile(gray,25), np.percentile(gray,75),
        np.percentile(gray,75)-np.percentile(gray,25),
        _entropy(gray.flatten()), _laplacian_var(gray),
    ], dtype=np.float32)


# ── Model loaders ──────────────────────────────────────────────────
def load_resnet50():
    if not TORCH_OK:
        return None, None, None
    path = os.path.join(MODELS_DIR, "resnet50_best.pth")
    if not os.path.exists(path):
        return None, None, None

    device = (torch.device("cuda") if torch.cuda.is_available()
               else torch.device("mps") if hasattr(torch.backends,"mps") and torch.backends.mps.is_available()
               else torch.device("cpu"))

    ckpt = torch.load(path, map_location=device)
    base = models.resnet50(weights=None)
    in_f = base.fc.in_features
    base.fc = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(in_f, 256), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(256, 2)
    )
    base.load_state_dict(ckpt["model_state"])
    base.eval().to(device)
    idx_to_class = {v:k for k,v in ckpt.get("class_to_idx",{"ai":0,"real":1}).items()}
    return base, idx_to_class, device

def load_rf():
    if not SK_OK:
        return None, None
    rp = os.path.join(MODELS_DIR, "random_forest.pkl")
    sp = os.path.join(MODELS_DIR, "rf_scaler.pkl")
    if not os.path.exists(rp):
        return None, None
    with open(rp,"rb") as f: rf = pickle.load(f)
    with open(sp,"rb") as f: sc = pickle.load(f)
    return rf, sc


# ── Predictions ────────────────────────────────────────────────────
def predict_rn(path, model, idx_to_class, device):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    with Image.open(path) as img:
        t = tf(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1).squeeze().cpu().numpy()
    idx   = int(probs.argmax())
    return idx_to_class.get(idx, str(idx)).upper(), float(probs[idx])*100, float(probs[0])*100, float(probs[1])*100

def predict_rf(path, rf, scaler):
    feat  = extract_features(path).reshape(1,-1)
    feats = scaler.transform(feat)
    pred  = rf.predict(feats)[0]
    probs = rf.predict_proba(feats)[0]
    return CLASSES[pred].upper(), float(probs[pred])*100, float(probs[0])*100, float(probs[1])*100


# ── Pretty printing ────────────────────────────────────────────────
RESET  = "\033[0m"
RED    = "\033[91m"
GREEN  = "\033[92m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
YELLOW = "\033[93m"

def bar(pct, width=35, fill_color=RED):
    filled = int(width * pct / 100)
    return fill_color + "█"*filled + RESET + "░"*(width-filled)

def print_header():
    print("\n" + "═"*62)
    print(f"  {BOLD}DWDM — AI vs Real Image Detector{RESET}")
    print("═"*62)

def print_verdict(label, confidence, prob_ai, prob_real, model_name):
    color = RED if label == "AI" else GREEN
    print(f"\n  {BOLD}┌{'─'*50}┐{RESET}")
    print(f"  {BOLD}│  Model      : {model_name:<36}│{RESET}")
    print(f"  {BOLD}│  Verdict    : {color}{BOLD}{label:<36}{RESET}{BOLD}│{RESET}")
    print(f"  {BOLD}│  Confidence : {confidence:.1f}%{' '*(35-len(f'{confidence:.1f}%'))}│{RESET}")
    print(f"  {BOLD}└{'─'*50}┘{RESET}")
    print()
    ai_color   = RED   if label == "AI"   else "\033[37m"
    real_color = GREEN if label == "REAL" else "\033[37m"
    print(f"  AI   [{bar(prob_ai,   fill_color=ai_color  )}] {prob_ai:.1f}%")
    print(f"  Real [{bar(prob_real, fill_color=real_color)}] {prob_real:.1f}%")

def print_ensemble(rn_result, rf_result):
    """Print combined ensemble verdict."""
    labels = [rn_result[0] if rn_result else None,
              rf_result[0]  if rf_result  else None]
    labels = [l for l in labels if l]
    if not labels:
        return

    prob_ai_list   = [r[2] for r in [rn_result, rf_result] if r]
    prob_real_list = [r[3] for r in [rn_result, rf_result] if r]
    avg_ai   = np.mean(prob_ai_list)
    avg_real = np.mean(prob_real_list)
    verdict  = "AI" if avg_ai > avg_real else "REAL"
    conf     = max(avg_ai, avg_real)

    print(f"\n  {'─'*60}")
    print(f"  {BOLD}{YELLOW}  ENSEMBLE VERDICT (Soft Voting){RESET}")
    print_verdict(verdict, conf, avg_ai, avg_real, "Ensemble")


# ══════════════════════════════════════════════════════════════════
# MAIN INTERACTIVE LOOP
# ══════════════════════════════════════════════════════════════════
def main():
    print_header()

    # Load models once
    print("  Loading models ...")
    rn_model, rn_idx_to_class, rn_device = load_resnet50()
    rf_model, rf_scaler = load_rf()

    if rn_model:
        print(f"  {GREEN}[✓]{RESET} ResNet50 loaded  (device: {rn_device})")
    else:
        print(f"  {YELLOW}[!]{RESET} ResNet50 not available (run train_model.py first or install PyTorch)")

    if rf_model:
        print(f"  {GREEN}[✓]{RESET} Random Forest loaded")
    else:
        print(f"  {YELLOW}[!]{RESET} Random Forest not available (run train_model.py first)")

    if not rn_model and not rf_model:
        print(f"\n  {RED}[ERROR]{RESET} No models loaded. Run train_model.py first.")
        sys.exit(1)

    print(f"\n  {'─'*60}")
    print(f"  Enter image path(s) to classify (or 'q' to quit)")
    print(f"  Supported: {', '.join(sorted(SUPPORTED_EXTS))}")
    print(f"  {'─'*60}")

    while True:
        try:
            user_input = input(f"\n  {BLUE}Image path >{RESET} ").strip().strip('"').strip("'")
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye!")
            break

        if user_input.lower() in ("q", "quit", "exit"):
            print("  Goodbye!")
            break

        if not user_input:
            continue

        path = Path(user_input)

        # ── Check if it's a folder ──────────────────────────────────
        if path.is_dir():
            files = [f for f in path.rglob("*")
                     if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]
            print(f"\n  Folder detected: {len(files)} image(s) found")
            for fpath in files:
                print(f"\n  Checking: {fpath.name}")
                _classify(str(fpath), rn_model, rn_idx_to_class, rn_device, rf_model, rf_scaler)
            continue

        # ── Single file ─────────────────────────────────────────────
        if not path.exists():
            print(f"  {RED}[ERROR]{RESET} File not found: {path}")
            continue

        if path.suffix.lower() not in SUPPORTED_EXTS:
            print(f"  {RED}[ERROR]{RESET} Unsupported format. Use: {', '.join(SUPPORTED_EXTS)}")
            continue

        _classify(str(path), rn_model, rn_idx_to_class, rn_device, rf_model, rf_scaler)


def _classify(img_path, rn_model, rn_idx_to_class, rn_device, rf_model, rf_scaler):
    """Run all available models on one image."""
    print(f"\n  {'═'*60}")
    print(f"  Analysing: {Path(img_path).name}")
    print(f"  {'═'*60}")

    t0 = time.time()
    rn_result = rf_result = None

    if rn_model:
        try:
            label, conf, p_ai, p_real = predict_rn(img_path, rn_model, rn_idx_to_class, rn_device)
            rn_result = (label, conf, p_ai, p_real)
            print_verdict(label, conf, p_ai, p_real, "ResNet50")
        except Exception as e:
            print(f"  {RED}[ResNet50 error]{RESET} {e}")

    if rf_model:
        try:
            label, conf, p_ai, p_real = predict_rf(img_path, rf_model, rf_scaler)
            rf_result = (label, conf, p_ai, p_real)
            print_verdict(label, conf, p_ai, p_real, "Random Forest")
        except Exception as e:
            print(f"  {RED}[RF error]{RESET} {e}")

    if rn_result and rf_result:
        print_ensemble(rn_result, rf_result)

    print(f"\n  Done in {(time.time()-t0)*1000:.0f}ms")


if __name__ == "__main__":
    main()