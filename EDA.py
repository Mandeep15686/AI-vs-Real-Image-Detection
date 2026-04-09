"""
DWDM Project - Step 5: Exploratory Data Analysis (EDA)
Data-Driven Detection and Analysis of AI-Generated vs Real Visual Media
===========================================================================
Purpose: Compute rich statistics and generate visualisation plots for the
         cleaned dataset:
         · Class distribution bar chart
         · Image resolution scatter & histogram
         · Mean pixel intensity & brightness distributions
         · RGB channel mean comparisons (AI vs Real)
         · Aspect ratio distribution
         · File size distribution
         · Sample image grid

Requires: Pillow, numpy, pandas, matplotlib  (pip install all)
"""

import os
import json
import random
import csv
import numpy as np
from pathlib import Path
from collections import Counter

try:
    import matplotlib
    matplotlib.use("Agg")    # non-interactive backend for servers
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    print("[WARNING] matplotlib not installed — plots will be skipped.")

from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
STD_AI_DIR    = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\cleaned_data/standardised/ai"
STD_REAL_DIR  = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\cleaned_data/standardised/real"
OUTPUT_DIR    = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\cleaned_data"
PLOTS_DIR     = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\cleaned_data/eda_plots"
SAMPLE_SIZE   = 500   # images to sample per class for pixel stats (full run is slow)
GRID_SAMPLES  = 9     # images per class shown in the grid
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {"ai": "#E45C5C", "real": "#4C9BE8"}


def collect_files(folder: str) -> list[Path]:
    p = Path(folder)
    if not p.exists():
        return []
    return sorted(f for f in p.glob("*.jpg") if f.is_file())


def image_stats(fpath: Path) -> dict:
    """Extract per-image statistics."""
    with Image.open(fpath) as img:
        arr   = np.array(img, dtype=np.float32)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        gray  = 0.299*r + 0.587*g + 0.114*b

        return {
            "width":        img.width,
            "height":       img.height,
            "aspect_ratio": round(img.width / img.height, 3),
            "size_kb":      round(fpath.stat().st_size / 1024, 2),
            "mean_r":       float(r.mean()),
            "mean_g":       float(g.mean()),
            "mean_b":       float(b.mean()),
            "brightness":   float(gray.mean()),
            "std_brightness": float(gray.std()),
        }


def compute_stats(files: list[Path], label: str, sample_size: int) -> list[dict]:
    """Sample files and compute statistics."""
    sampled = random.sample(files, min(sample_size, len(files)))
    total   = len(sampled)
    results = []

    print(f"\n  Computing pixel statistics for [{label}] — {total} samples ...")
    for i, f in enumerate(sampled, 1):
        if i % 100 == 0 or i == total:
            print(f"    {i}/{total}")
        try:
            s = image_stats(f)
            s["label"] = label
            s["path"]  = str(f)
            results.append(s)
        except Exception as e:
            pass

    return results


def save_eda_csv(all_stats: list[dict]):
    """Save EDA stats to CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fpath = os.path.join(OUTPUT_DIR, "eda_stats.csv")
    fields = ["label", "width", "height", "aspect_ratio", "size_kb",
              "mean_r", "mean_g", "mean_b", "brightness", "std_brightness", "path"]

    with open(fpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_stats)
    print(f"\n  [✓] EDA CSV saved → {fpath}")
    return fpath


def plot_class_distribution(ai_count: int, real_count: int):
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["AI-Generated", "Real"], [ai_count, real_count],
                  color=[COLORS["ai"], COLORS["real"]], width=0.5, edgecolor="white")
    for bar, count in zip(bars, [ai_count, real_count]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f"{count:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Class Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Images")
    ax.set_ylim(0, max(ai_count, real_count) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_class_distribution.png"), dpi=150)
    plt.close()
    print("  [✓] Plot: class distribution")


def plot_brightness_distribution(ai_stats: list[dict], real_stats: list[dict]):
    fig, ax = plt.subplots(figsize=(8, 4))
    ai_bright   = [s["brightness"]   for s in ai_stats]
    real_bright = [s["brightness"]   for s in real_stats]
    bins = np.linspace(0, 255, 50)
    ax.hist(ai_bright,   bins=bins, alpha=0.6, color=COLORS["ai"],   label="AI-Generated", density=True)
    ax.hist(real_bright, bins=bins, alpha=0.6, color=COLORS["real"], label="Real",          density=True)
    ax.axvline(np.mean(ai_bright),   color=COLORS["ai"],   linestyle="--", linewidth=1.5, label=f"AI mean={np.mean(ai_bright):.1f}")
    ax.axvline(np.mean(real_bright), color=COLORS["real"], linestyle="--", linewidth=1.5, label=f"Real mean={np.mean(real_bright):.1f}")
    ax.set_title("Brightness Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Pixel Brightness (0–255)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_brightness_distribution.png"), dpi=150)
    plt.close()
    print("  [✓] Plot: brightness distribution")


def plot_rgb_channel_means(ai_stats: list[dict], real_stats: list[dict]):
    channels = ["mean_r", "mean_g", "mean_b"]
    labels   = ["Red", "Green", "Blue"]
    ch_colors = ["#E05050", "#50C050", "#5070E0"]

    x = np.arange(len(channels))
    ai_means   = [np.mean([s[c] for s in ai_stats])   for c in channels]
    real_means = [np.mean([s[c] for s in real_stats]) for c in channels]

    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.35
    ax.bar(x - w/2, ai_means,   width=w, color=COLORS["ai"],   label="AI-Generated", alpha=0.85)
    ax.bar(x + w/2, real_means, width=w, color=COLORS["real"], label="Real",          alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Mean RGB Channel Values (AI vs Real)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Pixel Value (0–255)")
    ax.set_ylim(0, 255)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_rgb_channel_means.png"), dpi=150)
    plt.close()
    print("  [✓] Plot: RGB channel means")


def plot_file_size_distribution(ai_stats: list[dict], real_stats: list[dict]):
    fig, ax = plt.subplots(figsize=(8, 4))
    ai_sizes   = [s["size_kb"]   for s in ai_stats]
    real_sizes = [s["size_kb"]   for s in real_stats]
    bins = np.linspace(0, max(max(ai_sizes), max(real_sizes)), 50)
    ax.hist(ai_sizes,   bins=bins, alpha=0.6, color=COLORS["ai"],   label="AI-Generated", density=True)
    ax.hist(real_sizes, bins=bins, alpha=0.6, color=COLORS["real"], label="Real",          density=True)
    ax.set_title("File Size Distribution (KB)", fontsize=13, fontweight="bold")
    ax.set_xlabel("File Size (KB)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_file_size_distribution.png"), dpi=150)
    plt.close()
    print("  [✓] Plot: file size distribution")


def plot_sample_grid(ai_files: list[Path], real_files: list[Path], n: int = 9):
    """Plot a grid of sample images: top row AI, bottom row Real."""
    ai_sample   = random.sample(ai_files,   min(n, len(ai_files)))
    real_sample = random.sample(real_files, min(n, len(real_files)))

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 5))
    fig.suptitle("Sample Images: AI-Generated (top) vs Real (bottom)",
                 fontsize=13, fontweight="bold")

    for i, f in enumerate(ai_sample):
        with Image.open(f) as img:
            axes[0, i].imshow(img)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("AI", fontsize=9, color=COLORS["ai"])

    for i, f in enumerate(real_sample):
        with Image.open(f) as img:
            axes[1, i].imshow(img)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Real", fontsize=9, color=COLORS["real"])

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_sample_grid.png"), dpi=150)
    plt.close()
    print("  [✓] Plot: sample grid")


def print_summary_table(ai_stats: list[dict], real_stats: list[dict]):
    metrics = ["brightness", "std_brightness", "mean_r", "mean_g", "mean_b", "size_kb"]
    print("\n  ── EDA Summary Table ──────────────────────────────────────")
    print(f"  {'Metric':<22} {'AI mean':>10} {'Real mean':>10} {'Diff':>10}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10}")
    for m in metrics:
        ai_m   = np.mean([s[m] for s in ai_stats])
        real_m = np.mean([s[m] for s in real_stats])
        diff   = ai_m - real_m
        print(f"  {m:<22} {ai_m:>10.2f} {real_m:>10.2f} {diff:>+10.2f}")
    print()


def main():
    print("\n" + "="*60)
    print("  DWDM PROJECT — STEP 5: EXPLORATORY DATA ANALYSIS")
    print("="*60)

    ai_files   = collect_files(STD_AI_DIR)
    real_files = collect_files(STD_REAL_DIR)

    print(f"  AI images found   : {len(ai_files):,}")
    print(f"  Real images found : {len(real_files):,}")

    if not ai_files or not real_files:
        print("\n  [ERROR] No images found. Run step4_standardise_images.py first.")
        return

    # ── Pixel-level statistics ────────────────────────────────────────────────
    ai_stats   = compute_stats(ai_files,   "ai",   SAMPLE_SIZE)
    real_stats = compute_stats(real_files, "real", SAMPLE_SIZE)

    save_eda_csv(ai_stats + real_stats)
    print_summary_table(ai_stats, real_stats)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if MPL_AVAILABLE:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        print(f"\n  Saving plots to: {PLOTS_DIR}")

        plot_class_distribution(len(ai_files), len(real_files))
        plot_brightness_distribution(ai_stats, real_stats)
        plot_rgb_channel_means(ai_stats, real_stats)
        plot_file_size_distribution(ai_stats, real_stats)
        plot_sample_grid(ai_files, real_files, n=GRID_SAMPLES)
    else:
        print("\n  [SKIP] matplotlib not available — install it to generate plots.")

    print("\n  Step 5 complete. Dataset is ready for ML model training.\n")


if __name__ == "__main__":
    main()