"""
DWDM Project - Step 1: Data Audit
Data-Driven Detection and Analysis of AI-Generated vs Real Visual Media
===========================================================================
Purpose: Audit both AI and Real image folders to get an overview of the dataset.
         Reports file counts, formats, sizes, and folder structure.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
import json

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — update these paths to point to your actual folders
# ─────────────────────────────────────────────────────────────────────────────
AI_IMAGES_DIR   = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\AI"      # Folder containing AI-generated images
REAL_IMAGES_DIR = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\REAL"    # Folder containing real/authentic images
OUTPUT_DIR      = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\cleaned_data"           # Where cleaned data will be saved
REPORT_JSON     = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\audit_report.json"      # JSON summary of this audit

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff", ".tif"}
# ─────────────────────────────────────────────────────────────────────────────


def audit_folder(folder_path: str, label: str) -> dict:
    """Walk a folder and collect metadata about every image file."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"[WARNING] Folder not found: {folder_path}")
        return {}

    stats = {
        "label":          label,
        "folder":         str(folder.resolve()),
        "total_files":    0,
        "image_files":    0,
        "non_image":      0,
        "formats":        Counter(),
        "sizes_bytes":    [],
        "file_list":      [],          # (path, size, extension)
        "empty_files":    [],
        "hidden_files":   [],
    }

    for root, dirs, files in os.walk(folder):
        # Skip hidden directories (e.g. .DS_Store folders)
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for fname in files:
            fpath = Path(root) / fname
            stats["total_files"] += 1

            # Hidden file check
            if fname.startswith("."):
                stats["hidden_files"].append(str(fpath))
                continue

            ext = fpath.suffix.lower()
            size = fpath.stat().st_size

            if ext in SUPPORTED_FORMATS:
                stats["image_files"] += 1
                stats["formats"][ext] += 1
                stats["sizes_bytes"].append(size)
                stats["file_list"].append((str(fpath), size, ext))

                if size == 0:
                    stats["empty_files"].append(str(fpath))
            else:
                stats["non_image"] += 1

    # Summary metrics
    sizes = stats["sizes_bytes"]
    if sizes:
        stats["min_size_kb"]  = round(min(sizes) / 1024, 2)
        stats["max_size_kb"]  = round(max(sizes) / 1024, 2)
        stats["mean_size_kb"] = round(sum(sizes) / len(sizes) / 1024, 2)
        stats["total_size_mb"]= round(sum(sizes) / (1024 * 1024), 2)
    else:
        stats["min_size_kb"] = stats["max_size_kb"] = stats["mean_size_kb"] = stats["total_size_mb"] = 0

    return stats


def print_report(stats: dict):
    """Pretty-print audit results to the console."""
    label = stats.get("label", "Unknown")
    sep = "─" * 60

    print(f"\n{'═'*60}")
    print(f"  AUDIT REPORT — {label.upper()} IMAGES")
    print(f"{'═'*60}")
    print(f"  Folder       : {stats.get('folder')}")
    print(sep)
    print(f"  Total files  : {stats.get('total_files', 0)}")
    print(f"  Image files  : {stats.get('image_files', 0)}")
    print(f"  Non-image    : {stats.get('non_image', 0)}")
    print(f"  Hidden files : {len(stats.get('hidden_files', []))}")
    print(f"  Empty files  : {len(stats.get('empty_files', []))}")
    print(sep)
    print(f"  Formats found:")
    for ext, count in stats.get("formats", {}).most_common():
        print(f"      {ext:<8} → {count:,} files")
    print(sep)
    print(f"  Size (min)   : {stats.get('min_size_kb', 0)} KB")
    print(f"  Size (max)   : {stats.get('max_size_kb', 0)} KB")
    print(f"  Size (mean)  : {stats.get('mean_size_kb', 0)} KB")
    print(f"  Total size   : {stats.get('total_size_mb', 0)} MB")
    print(f"{'═'*60}\n")


def main():
    print("\n" + "="*60)
    print("  DWDM PROJECT — STEP 1: DATA AUDIT")
    print("="*60)

    ai_stats   = audit_folder(AI_IMAGES_DIR,   label="AI-Generated")
    real_stats = audit_folder(REAL_IMAGES_DIR, label="Real")

    print_report(ai_stats)
    print_report(real_stats)

    # Compare class balance
    ai_count   = ai_stats.get("image_files", 0)
    real_count = real_stats.get("image_files", 0)
    total      = ai_count + real_count

    print(f"\n{'─'*60}")
    print("  CLASS BALANCE SUMMARY")
    print(f"{'─'*60}")
    print(f"  AI images   : {ai_count:,}  ({ai_count/total*100:.1f}% of total)" if total else "  AI images   : 0")
    print(f"  Real images : {real_count:,}  ({real_count/total*100:.1f}% of total)" if total else "  Real images : 0")
    print(f"  Grand total : {total:,}")
    if ai_count and real_count:
        ratio = max(ai_count, real_count) / min(ai_count, real_count)
        print(f"  Balance ratio (max/min): {ratio:.3f}  {'[BALANCED ✓]' if ratio < 1.2 else '[IMBALANCED — consider augmentation]'}")
    print(f"{'─'*60}\n")

    # Save JSON report (convert Counter to dict for JSON serialisation)
    report = {
        "ai_images":   {**ai_stats,   "formats": dict(ai_stats.get("formats", {})),   "file_list": []},
        "real_images": {**real_stats, "formats": dict(real_stats.get("formats", {})), "file_list": []},
    }
    # Remove large lists from JSON to keep it readable
    for key in ("sizes_bytes",):
        report["ai_images"].pop(key, None)
        report["real_images"].pop(key, None)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, REPORT_JSON), "w") as f:
        json.dump(report, f, indent=2)

    print(f"  [✓] Audit report saved → {OUTPUT_DIR}/{REPORT_JSON}")
    print("  Step 1 complete. Proceed to step2_validate_images.py\n")


if __name__ == "__main__":
    main()
