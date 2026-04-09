"""
DWDM Project - Step 2: Image Validation & Corruption Detection
Data-Driven Detection and Analysis of AI-Generated vs Real Visual Media
===========================================================================
Purpose: Open every image with Pillow to detect:
         - Corrupted / unreadable files
         - Truncated images
         - Unsupported or misnamed files
         - Images with suspicious modes (e.g. palette mode P)
         Outputs a CSV log of all issues found.
"""

import os
import csv
import json
from pathlib import Path
from PIL import Image, UnidentifiedImageError

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
AI_IMAGES_DIR    = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\AI"
REAL_IMAGES_DIR  = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\REAL"
OUTPUT_DIR       = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\cleaned_data"
CORRUPT_LOG_CSV  = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\corrupt_images.csv"
VALID_LOG_CSV    = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\valid_images.csv"
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
# ─────────────────────────────────────────────────────────────────────────────


def validate_image(fpath: Path) -> dict:
    """
    Try to fully open and verify an image.
    Returns a dict with validation result.
    """
    result = {
        "path":    str(fpath),
        "label":   None,           # set by caller
        "status":  "ok",
        "reason":  "",
        "width":   None,
        "height":  None,
        "mode":    None,
        "format":  None,
        "size_kb": round(fpath.stat().st_size / 1024, 2),
    }

    try:
        with Image.open(fpath) as img:
            # Force full decode — catches truncated images
            img.verify()           # structural check (cannot get size after verify)

        # Re-open to read metadata (verify() closes the stream)
        with Image.open(fpath) as img:
            img.load()             # decode all pixel data
            result["width"]  = img.width
            result["height"] = img.height
            result["mode"]   = img.mode
            result["format"] = img.format

            # Flag palette-mode images — may lose info on conversion
            if img.mode == "P":
                result["status"] = "warning"
                result["reason"] = "Palette mode (P) — will be converted to RGB"

            # Flag grayscale
            if img.mode == "L":
                result["status"] = "warning"
                result["reason"] = "Grayscale (L) — will be converted to RGB"

            # Flag very small images (likely thumbnails / icons)
            if img.width < 32 or img.height < 32:
                result["status"] = "warning"
                result["reason"] = f"Tiny image: {img.width}×{img.height}"

    except UnidentifiedImageError:
        result["status"] = "corrupt"
        result["reason"] = "Cannot identify image file (not a valid image or wrong extension)"

    except OSError as e:
        result["status"] = "corrupt"
        result["reason"] = f"OS/IO error: {e}"

    except Exception as e:
        result["status"] = "corrupt"
        result["reason"] = f"Unexpected error: {e}"

    return result


def validate_folder(folder_path: str, label: str) -> list[dict]:
    """Validate all supported images in a folder."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"[WARNING] Folder not found: {folder_path}")
        return []

    results = []
    files = [f for f in folder.rglob("*")
             if f.is_file()
             and not f.name.startswith(".")
             and f.suffix.lower() in SUPPORTED_FORMATS]

    total = len(files)
    print(f"\n  Validating {total:,} images in [{label}] ...")

    for i, fpath in enumerate(files, 1):
        if i % 1000 == 0 or i == total:
            print(f"    Progress: {i:,}/{total:,}  ({i/total*100:.0f}%)")

        res = validate_image(fpath)
        res["label"] = label
        results.append(res)

    return results


def summarise(results: list[dict], label: str):
    """Print a summary of validation results for one class."""
    ok       = [r for r in results if r["status"] == "ok"]
    warnings = [r for r in results if r["status"] == "warning"]
    corrupt  = [r for r in results if r["status"] == "corrupt"]

    print(f"\n  ── {label} Validation Summary ──")
    print(f"     Total checked  : {len(results):,}")
    print(f"     ✓ Valid        : {len(ok):,}")
    print(f"     ⚠ Warnings     : {len(warnings):,}")
    print(f"     ✗ Corrupt      : {len(corrupt):,}")

    if warnings:
        reasons = {}
        for r in warnings:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        print("     Warning breakdown:")
        for reason, count in reasons.items():
            print(f"       · {reason}: {count}")

    if corrupt:
        print("     First 5 corrupt files:")
        for r in corrupt[:5]:
            print(f"       · {r['path']}  →  {r['reason']}")


def save_csv(results: list[dict], output_dir: str, filename: str):
    """Save results list to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    fpath = os.path.join(output_dir, filename)

    if not results:
        print(f"  (no records to save for {filename})")
        return

    fieldnames = ["label", "status", "reason", "width", "height", "mode", "format", "size_kb", "path"]
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"  [✓] Saved: {fpath}  ({len(results):,} rows)")


def main():
    print("\n" + "="*60)
    print("  DWDM PROJECT — STEP 2: IMAGE VALIDATION")
    print("="*60)

    ai_results   = validate_folder(AI_IMAGES_DIR,   label="ai")
    real_results = validate_folder(REAL_IMAGES_DIR, label="real")

    all_results = ai_results + real_results

    summarise(ai_results,   "AI-Generated")
    summarise(real_results, "Real")

    # Split into corrupt/warning vs valid
    problem_list = [r for r in all_results if r["status"] != "ok"]
    valid_list   = [r for r in all_results if r["status"] == "ok"]

    print(f"\n  Overall: {len(valid_list):,} valid  |  {len(problem_list):,} problematic")

    save_csv(problem_list, OUTPUT_DIR, CORRUPT_LOG_CSV)
    save_csv(valid_list,   OUTPUT_DIR, VALID_LOG_CSV)

    # Save summary JSON for use in next steps
    summary = {
        "ai_total":   len(ai_results),
        "real_total": len(real_results),
        "ai_valid":   sum(1 for r in ai_results   if r["status"] == "ok"),
        "real_valid": sum(1 for r in real_results if r["status"] == "ok"),
        "corrupt_count":  sum(1 for r in all_results if r["status"] == "corrupt"),
        "warning_count":  sum(1 for r in all_results if r["status"] == "warning"),
    }
    with open(os.path.join(OUTPUT_DIR, "validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  [✓] Validation summary saved.")
    print("  Step 2 complete. Proceed to step3_remove_duplicates.py\n")


if __name__ == "__main__":
    main()
