"""
DWDM Project - Step 3: Duplicate Image Detection & Removal
Data-Driven Detection and Analysis of AI-Generated vs Real Visual Media
===========================================================================
Purpose: Detect exact duplicates using MD5 / SHA-256 hashing, and near-duplicate
         images using perceptual hashing (pHash / dHash).
         Logs all duplicates and removes them (or moves to a quarantine folder).

Requires: Pillow  (pip install Pillow)
Optional: imagehash  (pip install imagehash) for perceptual hashing
"""

import os
import csv
import json
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict

try:
    import imagehash
    from PIL import Image
    PHASH_AVAILABLE = True
except ImportError:
    PHASH_AVAILABLE = False
    print("[INFO] imagehash not installed — perceptual hashing disabled.")
    print("       Run: pip install imagehash")

from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
AI_IMAGES_DIR      = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\AI"
REAL_IMAGES_DIR    = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\REAL"
OUTPUT_DIR         = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\cleaned_data"
QUARANTINE_DIR     = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\cleaned_data/quarantine_duplicates"  # duplicates moved here
DUPLICATES_CSV     = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\duplicates_log.csv"
PHASH_THRESHOLD    = 8       # Hamming distance ≤ this → near-duplicate (0 = identical)
SUPPORTED_FORMATS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
DRY_RUN            = True   # True = only log, False = actually move duplicates
# ─────────────────────────────────────────────────────────────────────────────


def md5_hash(fpath: Path, chunk_size: int = 65536) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(fpath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def collect_files(folder: str, label: str) -> list[dict]:
    """Return list of {path, label} for all supported images."""
    p = Path(folder)
    if not p.exists():
        print(f"[WARNING] Not found: {folder}")
        return []
    return [
        {"path": f, "label": label}
        for f in p.rglob("*")
        if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in SUPPORTED_FORMATS
    ]


def detect_exact_duplicates(file_records: list[dict]) -> tuple[dict, list[dict]]:
    """
    Hash every file with MD5.
    Returns:
      - hash_map  : {md5: [file_record, ...]}
      - duplicates: list of file_records to be removed (keep first occurrence)
    """
    hash_map    = defaultdict(list)
    duplicates  = []

    total = len(file_records)
    print(f"\n  Computing MD5 hashes for {total:,} files ...")

    for i, rec in enumerate(file_records, 1):
        if i % 2000 == 0 or i == total:
            print(f"    {i:,}/{total:,}")

        digest = md5_hash(rec["path"])
        rec["md5"] = digest
        hash_map[digest].append(rec)

    # Collect duplicates (all except the first occurrence of each hash)
    for digest, group in hash_map.items():
        if len(group) > 1:
            for rec in group[1:]:   # keep group[0], flag the rest
                rec["duplicate_type"] = "exact"
                rec["duplicate_of"]   = str(group[0]["path"])
                duplicates.append(rec)

    return hash_map, duplicates


def detect_near_duplicates(file_records: list[dict], threshold: int = 8) -> list[dict]:
    """
    Compute perceptual hash (pHash) for each image.
    Any pair with Hamming distance <= threshold is flagged as near-duplicate.
    NOTE: O(n²) — use on reduced set after exact-dup removal for performance.
    """
    if not PHASH_AVAILABLE:
        print("\n  [SKIP] Perceptual hashing skipped (imagehash not installed).")
        return []

    print(f"\n  Computing perceptual hashes (pHash) for {len(file_records):,} files ...")
    hashes = []

    for i, rec in enumerate(file_records, 1):
        if i % 2000 == 0 or i == len(file_records):
            print(f"    {i:,}/{len(file_records):,}")
        try:
            with Image.open(rec["path"]) as img:
                ph = imagehash.phash(img)
            hashes.append((rec, ph))
        except Exception:
            pass  # already caught in validation step

    print(f"  Running pairwise comparison (this may take a while for large datasets) ...")

    near_dups = []
    seen_paths = set()

    for i in range(len(hashes)):
        rec_i, ph_i = hashes[i]
        path_i = str(rec_i["path"])
        if path_i in seen_paths:
            continue

        for j in range(i + 1, len(hashes)):
            rec_j, ph_j = hashes[j]
            path_j = str(rec_j["path"])
            if path_j in seen_paths:
                continue

            dist = abs(ph_i - ph_j)
            if dist <= threshold:
                rec_j["duplicate_type"] = f"near (pHash dist={dist})"
                rec_j["duplicate_of"]   = path_i
                near_dups.append(rec_j)
                seen_paths.add(path_j)

    return near_dups


def quarantine_files(duplicates: list[dict], quarantine_dir: str, dry_run: bool = True):
    """Move duplicate files to a quarantine folder."""
    qdir = Path(quarantine_dir)
    qdir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for rec in duplicates:
        src = rec["path"]
        dst = qdir / rec["label"] / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dry_run:
            pass  # just log
        else:
            try:
                shutil.move(str(src), str(dst))
                moved += 1
            except Exception as e:
                print(f"  [ERROR] Could not move {src}: {e}")

    action = "would be moved" if dry_run else "moved"
    print(f"  {len(duplicates):,} duplicate files {action} to quarantine.")
    if dry_run:
        print("  Set DRY_RUN = False to actually move them.")
    return moved


def save_duplicate_log(duplicates: list[dict], output_dir: str, filename: str):
    """Save duplicate list to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    fpath = os.path.join(output_dir, filename)

    fieldnames = ["label", "duplicate_type", "duplicate_of", "md5", "path"]
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in duplicates:
            row = {k: rec.get(k, "") for k in fieldnames}
            row["path"] = str(rec["path"])
            row["duplicate_of"] = str(rec.get("duplicate_of", ""))
            writer.writerow(row)

    print(f"  [✓] Duplicate log saved → {fpath}  ({len(duplicates):,} records)")


def main():
    print("\n" + "="*60)
    print("  DWDM PROJECT — STEP 3: DUPLICATE REMOVAL")
    print("="*60)
    print(f"  Mode: {'DRY RUN (no files moved)' if DRY_RUN else 'LIVE (files will be moved)'}")
    print(f"  pHash threshold: {PHASH_THRESHOLD}")

    ai_files   = collect_files(AI_IMAGES_DIR,   "ai")
    real_files = collect_files(REAL_IMAGES_DIR, "real")
    all_files  = ai_files + real_files

    print(f"\n  Total images to scan: {len(all_files):,}")
    print(f"    AI   : {len(ai_files):,}")
    print(f"    Real : {len(real_files):,}")

    # ── EXACT DUPLICATES ──────────────────────────────────────────────────────
    hash_map, exact_dups = detect_exact_duplicates(all_files)
    print(f"\n  Exact duplicates found: {len(exact_dups):,}")

    # ── NEAR DUPLICATES (perceptual) ─────────────────────────────────────────
    # Run on unique files only (post exact-dup removal)
    exact_dup_paths = {str(r["path"]) for r in exact_dups}
    unique_files    = [r for r in all_files if str(r["path"]) not in exact_dup_paths]
    near_dups       = detect_near_duplicates(unique_files, threshold=PHASH_THRESHOLD)
    print(f"  Near-duplicates found : {len(near_dups):,}")

    all_dups = exact_dups + near_dups

    # ── QUARANTINE ────────────────────────────────────────────────────────────
    quarantine_files(all_dups, QUARANTINE_DIR, dry_run=DRY_RUN)

    # ── LOGGING ───────────────────────────────────────────────────────────────
    save_duplicate_log(all_dups, OUTPUT_DIR, DUPLICATES_CSV)

    summary = {
        "total_scanned":    len(all_files),
        "exact_duplicates": len(exact_dups),
        "near_duplicates":  len(near_dups),
        "total_duplicates": len(all_dups),
        "unique_remaining": len(all_files) - len(all_dups),
        "dry_run":          DRY_RUN,
    }
    with open(os.path.join(OUTPUT_DIR, "duplicate_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n  ── Duplicate Removal Summary ──")
    for k, v in summary.items():
        print(f"     {k:<25}: {v}")

    print("\n  Step 3 complete. Proceed to step4_standardise_images.py\n")


if __name__ == "__main__":
    main()
