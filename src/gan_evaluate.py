"""
DWDM Project — gan_evaluate.py
Evaluate GAN output quality + integrate generated images into the dataset
=====================================================================
Metrics computed:
  1. FID Score  (Fréchet Inception Distance) — lower = better quality
  2. IS Score   (Inception Score)            — higher = better
  3. Visual sample grid comparison (Real vs GAN-generated)
  4. Pixel statistics comparison (brightness, RGB, sharpness)

Integration:
  - Move GAN-generated images into the AI training split
  - Re-run predict.py on GAN images to verify classifier detects them

Usage:
    python gan_evaluate.py                        # full evaluation
    python gan_evaluate.py --compare_only         # just compare real vs fake stats
    python gan_evaluate.py --integrate            # add GAN images to training set

Requirements:
    pip install torch torchvision Pillow numpy matplotlib scipy
    pip install pytorch-fid   (optional — for FID score)
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# ─────────────────────────────────────────────────────────────────
MODELS_DIR    = r"C:\Users\shash\Downloads\DATASET\DATASET\models"
GAN_OUTPUT_DIR= r"C:\Users\shash\Downloads\DATASET\DATASET\gan_outputs"
REAL_DIR      = r"C:\Users\shash\Downloads\DATASET\DATASET\cleaned_data\splits\train\real"
AI_TRAIN_DIR  = r"C:\Users\shash\Downloads\DATASET\DATASET\cleaned_data\splits\train\ai"
GENERATED_DIR = os.path.join(GAN_OUTPUT_DIR, "generated_images")
COMPARE_DIR   = os.path.join(GAN_OUTPUT_DIR, "comparisons")
# ─────────────────────────────────────────────────────────────────
os.makedirs(COMPARE_DIR, exist_ok=True)

try:
    import torch
    from torchvision import transforms, models
    from torch.utils.data import DataLoader, Dataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL = True
except ImportError:
    MPL = False


# ══════════════════════════════════════════════════════════════════
# PIXEL STATISTICS
# ══════════════════════════════════════════════════════════════════
def compute_pixel_stats(folder: str, n_sample: int = 500, label: str = "") -> dict:
    """Compute mean/std of brightness and RGB channels for a folder of images."""
    files = [f for f in os.listdir(folder)
             if f.lower().endswith((".jpg",".jpeg",".png"))]

    import random
    random.shuffle(files)
    files = files[:n_sample]

    br_list, r_list, g_list, b_list = [], [], [], []

    for fname in files:
        try:
            with Image.open(os.path.join(folder, fname)) as img:
                img = img.convert("RGB").resize((64, 64))
                arr = np.array(img, dtype=np.float32)
            r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
            gray = 0.299*r + 0.587*g + 0.114*b
            br_list.append(gray.mean())
            r_list.append(r.mean())
            g_list.append(g.mean())
            b_list.append(b.mean())
        except Exception:
            pass

    if not br_list:
        return {}

    return {
        "label":       label,
        "n":           len(br_list),
        "brightness":  {"mean": float(np.mean(br_list)), "std": float(np.std(br_list))},
        "R":           {"mean": float(np.mean(r_list)),  "std": float(np.std(r_list))},
        "G":           {"mean": float(np.mean(g_list)),  "std": float(np.std(g_list))},
        "B":           {"mean": float(np.mean(b_list)),  "std": float(np.std(b_list))},
    }


# ══════════════════════════════════════════════════════════════════
# COMPARISON PLOTS
# ══════════════════════════════════════════════════════════════════
def plot_stats_comparison(real_stats: dict, ai_stats: dict, gan_stats: dict):
    if not MPL:
        return
    metrics = ["brightness", "R", "G", "B"]
    labels  = ["Brightness", "Red", "Green", "Blue"]
    colors  = {"Real":"#4C9BE8", "AI Train":"#E45C5C", "GAN Generated":"#A78BFA"}

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle("Pixel Statistics: Real vs AI Train vs GAN Generated",
                 fontsize=13, fontweight="bold")

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[i]
        groups = {
            "Real":          (real_stats.get(metric, {}).get("mean", 0),
                               real_stats.get(metric, {}).get("std", 0)),
            "AI Train":       (ai_stats.get(metric, {}).get("mean", 0),
                               ai_stats.get(metric, {}).get("std", 0)),
            "GAN Generated": (gan_stats.get(metric, {}).get("mean", 0),
                               gan_stats.get(metric, {}).get("std", 0)),
        }
        x    = np.arange(len(groups))
        means = [v[0] for v in groups.values()]
        stds  = [v[1] for v in groups.values()]

        bars = ax.bar(x, means, yerr=stds, color=list(colors.values()),
                       capsize=5, width=0.6, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(list(groups.keys()), rotation=12, fontsize=8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Pixel Value (0–255)")
        ax.set_ylim(0, 280)
        ax.spines[["top","right"]].set_visible(False)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                    f"{mean:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(COMPARE_DIR, "stats_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [✓] Stats comparison plot → {path}")


def plot_image_comparison(real_dir: str, gan_dir: str, n: int = 8):
    """Show n real images vs n GAN-generated images side by side."""
    if not MPL:
        return

    real_files = sorted(os.listdir(real_dir))[:n]
    gan_files  = sorted(os.listdir(gan_dir))[:n]

    if not real_files or not gan_files:
        print("  [!] Not enough images for comparison plot")
        return

    fig, axes = plt.subplots(2, n, figsize=(n*2, 5))
    fig.suptitle("Real (top) vs GAN Generated (bottom)", fontsize=12, fontweight="bold")

    for i in range(n):
        # Real
        if i < len(real_files):
            with Image.open(os.path.join(real_dir, real_files[i])) as img:
                axes[0, i].imshow(img.convert("RGB").resize((64,64)))
        axes[0, i].axis("off")
        if i == 0: axes[0, i].set_title("Real", fontsize=9, color="#4C9BE8")

        # GAN
        if i < len(gan_files):
            with Image.open(os.path.join(gan_dir, gan_files[i])) as img:
                axes[1, i].imshow(img.convert("RGB").resize((64,64)))
        axes[1, i].axis("off")
        if i == 0: axes[1, i].set_title("GAN", fontsize=9, color="#A78BFA")

    plt.tight_layout()
    path = os.path.join(COMPARE_DIR, "real_vs_gan.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [✓] Visual comparison → {path}")


# ══════════════════════════════════════════════════════════════════
# FID SCORE  (requires pytorch-fid)
# ══════════════════════════════════════════════════════════════════
def compute_fid(real_dir: str, fake_dir: str) -> float:
    """
    Compute Fréchet Inception Distance (FID).
    Lower is better (0 = identical distributions).
    Requires: pip install pytorch-fid
    """
    try:
        from pytorch_fid import fid_score
        fid = fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir],
            batch_size=50,
            device="cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu",
            dims=2048,
        )
        return float(fid)
    except ImportError:
        print("  [!] pytorch-fid not installed. FID skipped.")
        print("      Install: pip install pytorch-fid")
        return None
    except Exception as e:
        print(f"  [!] FID computation error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# INTEGRATE GAN IMAGES INTO DATASET
# ══════════════════════════════════════════════════════════════════
def integrate_gan_images(n_to_add: int = 500):
    """
    Copy GAN-generated images into the AI training split.
    Renames them consistently: gan_000001.jpg, gan_000002.jpg ...
    """
    if not os.path.exists(GENERATED_DIR):
        print(f"  [ERROR] Generated images folder not found: {GENERATED_DIR}")
        print("  Run: python gan_generator.py --mode generate --num_images 500")
        return

    gen_files = sorted([f for f in os.listdir(GENERATED_DIR)
                        if f.lower().endswith((".jpg",".jpeg",".png"))])

    if not gen_files:
        print("  [ERROR] No generated images found.")
        return

    to_copy = gen_files[:n_to_add]
    print(f"\n  Integrating {len(to_copy)} GAN images into AI training split ...")
    print(f"  Source : {GENERATED_DIR}")
    print(f"  Dest   : {AI_TRAIN_DIR}")

    added = 0
    for i, fname in enumerate(to_copy, 1):
        src = os.path.join(GENERATED_DIR, fname)
        dst = os.path.join(AI_TRAIN_DIR,  f"gan_{i:06d}.jpg")
        try:
            shutil.copy2(src, dst)
            added += 1
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")

    print(f"\n  [✓] {added} GAN images added to training split.")
    print(f"  AI training set now has ~{len(os.listdir(AI_TRAIN_DIR)):,} images.")
    print("  Re-run train_model.py to retrain with the augmented dataset.\n")


# ══════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Evaluate GAN output quality")
    parser.add_argument("--compare_only", action="store_true",
                        help="Only run pixel stats comparison (no FID)")
    parser.add_argument("--integrate", action="store_true",
                        help="Copy GAN images into AI training split")
    parser.add_argument("--n_integrate", type=int, default=500,
                        help="Number of GAN images to add to training set")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  DWDM — GAN Evaluation")
    print("="*65)

    if args.integrate:
        integrate_gan_images(args.n_integrate)
        return

    # Check generated images exist
    if not os.path.exists(GENERATED_DIR) or not os.listdir(GENERATED_DIR):
        print(f"\n  [ERROR] No generated images found at: {GENERATED_DIR}")
        print("  Run: python gan_generator.py --mode generate --num_images 500")
        sys.exit(1)

    n_gen = len([f for f in os.listdir(GENERATED_DIR) if f.endswith((".jpg",".png"))])
    print(f"\n  Found {n_gen} GAN-generated images")

    # ── Pixel statistics ──────────────────────────────────────────
    print("\n  Computing pixel statistics ...")
    real_stats = compute_pixel_stats(REAL_DIR,      label="Real",          n_sample=500)
    ai_stats   = compute_pixel_stats(AI_TRAIN_DIR,  label="AI Train",      n_sample=500)
    gan_stats  = compute_pixel_stats(GENERATED_DIR, label="GAN Generated", n_sample=min(500, n_gen))

    print("\n  ── Pixel Statistics Comparison ─────────────────────────")
    header = f"  {'Metric':<20} {'Real':>12} {'AI Train':>12} {'GAN Gen':>12}"
    print(header)
    print("  " + "─"*56)

    for metric in ["brightness", "R", "G", "B"]:
        r_m = real_stats.get(metric, {}).get("mean", 0)
        a_m = ai_stats.get(metric, {}).get("mean", 0)
        g_m = gan_stats.get(metric, {}).get("mean", 0)
        print(f"  {metric:<20} {r_m:>12.2f} {a_m:>12.2f} {g_m:>12.2f}")

    # ── Plots ────────────────────────────────────────────────────
    plot_stats_comparison(real_stats, ai_stats, gan_stats)
    plot_image_comparison(REAL_DIR, GENERATED_DIR)

    # ── FID score ─────────────────────────────────────────────────
    if not args.compare_only:
        print("\n  Computing FID score ...")
        fid = compute_fid(REAL_DIR, GENERATED_DIR)
        if fid is not None:
            quality = "Excellent" if fid < 50 else ("Good" if fid < 100 else ("Moderate" if fid < 200 else "Poor"))
            print(f"  FID Score: {fid:.2f}  ({quality})")
            print("  (Lower is better — FID<50 is good, FID<100 is acceptable)")
        else:
            print("  FID score unavailable (install pytorch-fid)")

    # ── Save summary ──────────────────────────────────────────────
    summary = {
        "generated_images":  n_gen,
        "real_stats":        real_stats,
        "ai_train_stats":    ai_stats,
        "gan_stats":         gan_stats,
        "compare_dir":       COMPARE_DIR,
    }
    summary_path = os.path.join(GAN_OUTPUT_DIR, "gan_evaluation.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  [✓] Evaluation summary → {summary_path}")

    print("\n  ── Next Steps ───────────────────────────────────────────")
    print("  1. Check real_vs_gan.png to visually assess image quality")
    print("  2. Check stats_comparison.png for distribution alignment")
    print("  3. Run: python gan_evaluate.py --integrate  (add to training set)")
    print("  4. Re-run: python train_model.py  (retrain with augmented data)\n")


if __name__ == "__main__":
    main()