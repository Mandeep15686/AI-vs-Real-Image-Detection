"""
DWDM Project - Step 6: Train / Validation / Test Split
Data-Driven Detection and Analysis of AI-Generated vs Real Visual Media
===========================================================================
Purpose: Split the cleaned standardised dataset into:
         · Training set    (70%)
         · Validation set  (15%)
         · Test set        (15%)
         Maintains class balance in each split.
         Generates a master CSV manifest and folder structure for ML pipelines.
"""

import os
import csv
import json
import shutil
import random
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
STD_AI_DIR      = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\cleaned_data/standardised/ai"
STD_REAL_DIR    = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\cleaned_data/standardised/real"

SPLIT_DIR       = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\cleaned_data/splits"       # destination for the 3 splits
OUTPUT_DIR      = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\cleaned_data"
MANIFEST_CSV    = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\dataset_manifest.csv"      # master file list with labels & split

TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15                        # must sum to 1.0

RANDOM_SEED     = 42
COPY_FILES      = True   # True = copy files into split folders; False = manifest only
# ─────────────────────────────────────────────────────────────────────────────

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, "Ratios must sum to 1.0"


def collect_files(folder: str, label: str, label_id: int) -> list[dict]:
    """Collect all .jpg files from a standardised folder."""
    p = Path(folder)
    if not p.exists():
        print(f"[WARNING] Not found: {folder}")
        return []
    files = sorted(f for f in p.glob("*.jpg") if f.is_file())
    return [{"path": str(f), "filename": f.name, "label": label, "label_id": label_id}
            for f in files]


def stratified_split(records: list[dict], train_r: float, val_r: float,
                      seed: int = 42) -> tuple[list, list, list]:
    """Split records into train/val/test maintaining class balance."""
    random.seed(seed)

    # Group by class
    by_class = {}
    for rec in records:
        by_class.setdefault(rec["label"], []).append(rec)

    train, val, test = [], [], []

    for label, items in by_class.items():
        random.shuffle(items)
        n      = len(items)
        n_train = int(n * train_r)
        n_val   = int(n * val_r)

        train += items[:n_train]
        val   += items[n_train : n_train + n_val]
        test  += items[n_train + n_val :]

        print(f"    [{label}] total={n:,} → train={n_train:,} / val={n_val:,} / test={n - n_train - n_val:,}")

    # Shuffle within each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def copy_split_files(records: list[dict], split_name: str, base_dir: str):
    """Optionally copy files into split subdirectories."""
    errors = 0
    for rec in records:
        src = Path(rec["path"])
        dst = Path(base_dir) / split_name / rec["label"] / rec["filename"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"  [ERROR] Copy failed: {src} → {dst}: {e}")
            errors += 1
    return errors


def save_manifest(train, val, test, output_dir: str, filename: str):
    """Save master CSV manifest with split assignments."""
    os.makedirs(output_dir, exist_ok=True)
    fpath = os.path.join(output_dir, filename)

    all_records = (
        [{"split": "train", **r} for r in train] +
        [{"split": "val",   **r} for r in val]   +
        [{"split": "test",  **r} for r in test]
    )

    fields = ["split", "label", "label_id", "filename", "path"]
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n  [✓] Manifest saved → {fpath}  ({len(all_records):,} rows)")
    return fpath


def print_split_summary(train, val, test):
    def class_counts(records):
        counts = {}
        for r in records:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        return counts

    print("\n  ── Final Split Summary ─────────────────────────────────────")
    for name, records in [("Train", train), ("Val", val), ("Test", test)]:
        cc = class_counts(records)
        total = len(records)
        ai_n   = cc.get("ai",   0)
        real_n = cc.get("real", 0)
        print(f"  {name:<8} : {total:>6,} total  |  AI={ai_n:,}  Real={real_n:,}")
    print()


def main():
    print("\n" + "="*60)
    print("  DWDM PROJECT — STEP 6: TRAIN/VAL/TEST SPLIT")
    print("="*60)
    print(f"  Split ratios: train={TRAIN_RATIO:.0%} / val={VAL_RATIO:.0%} / test={TEST_RATIO:.0%}")
    print(f"  Random seed : {RANDOM_SEED}")
    print(f"  Copy files  : {COPY_FILES}")

    ai_records   = collect_files(STD_AI_DIR,   label="ai",   label_id=0)
    real_records = collect_files(STD_REAL_DIR, label="real", label_id=1)
    all_records  = ai_records + real_records

    print(f"\n  Total records: {len(all_records):,} (AI={len(ai_records):,}, Real={len(real_records):,})")

    if not all_records:
        print("\n  [ERROR] No images found. Run step4_standardise_images.py first.")
        return

    print(f"\n  Splitting ...")
    train, val, test = stratified_split(all_records, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)

    print_split_summary(train, val, test)
    save_manifest(train, val, test, OUTPUT_DIR, MANIFEST_CSV)

    if COPY_FILES:
        print(f"  Copying files into split folders → {SPLIT_DIR} ...")
        for split_name, records in [("train", train), ("val", val), ("test", test)]:
            errors = copy_split_files(records, split_name, SPLIT_DIR)
            print(f"    {split_name}: {len(records):,} files  (errors: {errors})")

    summary = {
        "seed":       RANDOM_SEED,
        "train_size": len(train),
        "val_size":   len(val),
        "test_size":  len(test),
        "total":      len(all_records),
        "split_dir":  SPLIT_DIR,
        "manifest":   os.path.join(OUTPUT_DIR, MANIFEST_CSV),
    }
    with open(os.path.join(OUTPUT_DIR, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n  [✓] Dataset split complete and ready for ML training.")
    print(f"  Use '{OUTPUT_DIR}/{MANIFEST_CSV}' as your dataset manifest.\n")


if __name__ == "__main__":
    main()