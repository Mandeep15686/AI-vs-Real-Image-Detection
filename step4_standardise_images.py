"""
DWDM Project - Step 4: Image Standardisation & Normalisation
Data-Driven Detection and Analysis of AI-Generated vs Real Visual Media
===========================================================================
Purpose: Convert all images to a uniform format ready for ML pipelines:
         · Convert all images to RGB JPEG (removes alpha, palette modes, etc.)
         · Resize to a target resolution (default 224×224 for CNN compatibility)
         · Rename files with a consistent scheme: ai_000001.jpg / real_000001.jpg
         · Copy standardised images to output folders
         · Strip EXIF metadata (privacy + noise reduction)
         · Save a metadata CSV mapping original → standardised filename

Requires: Pillow  (pip install Pillow)
"""

import os
import csv
import json
import shutil
from pathlib import Path
from PIL import Image, ExifTags

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
AI_IMAGES_DIR     = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\AI"
REAL_IMAGES_DIR   = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\REAL"

OUTPUT_DIR        = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\cleaned_data"
STD_AI_DIR        = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\cleaned_data/standardised/ai"
STD_REAL_DIR      = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\cleaned_data/standardised/real"
METADATA_CSV      = "C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\standardised_metadata.csv"

TARGET_SIZE       = (224, 224)   # Width × Height  (change to 256/299/384 as needed)
OUTPUT_FORMAT     = "JPEG"       # JPEG for universal compatibility
OUTPUT_EXT        = ".jpg"
JPEG_QUALITY      = 95           # 85–95 is a good balance of quality vs size
RESIZE_STRATEGY   = "center_crop"  # "resize" | "center_crop" | "pad"

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
# ─────────────────────────────────────────────────────────────────────────────


def center_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Crop the centre of an image to match target aspect ratio, then resize."""
    src_w, src_h = img.size
    target_ratio  = target_w / target_h
    src_ratio     = src_w / src_h

    if src_ratio > target_ratio:
        # Image is wider than target — crop sides
        new_w = int(src_h * target_ratio)
        left  = (src_w - new_w) // 2
        img   = img.crop((left, 0, left + new_w, src_h))
    elif src_ratio < target_ratio:
        # Image is taller than target — crop top/bottom
        new_h = int(src_w / target_ratio)
        top   = (src_h - new_h) // 2
        img   = img.crop((0, top, src_w, top + new_h))

    return img.resize((target_w, target_h), Image.LANCZOS)


def pad_to_square(img: Image.Image, target_w: int, target_h: int,
                  fill_color=(0, 0, 0)) -> Image.Image:
    """Resize image keeping aspect ratio, then pad to target size."""
    img.thumbnail((target_w, target_h), Image.LANCZOS)
    out = Image.new("RGB", (target_w, target_h), fill_color)
    offset = ((target_w - img.width) // 2, (target_h - img.height) // 2)
    out.paste(img, offset)
    return out


def strip_exif(img: Image.Image) -> Image.Image:
    """Return a new image with all EXIF metadata removed."""
    data = img.getdata()
    clean = Image.new(img.mode, img.size)
    clean.putdata(data)
    return clean


def standardise_image(src: Path,
                       dst: Path,
                       target_size: tuple[int, int],
                       strategy: str = "center_crop") -> dict:
    """
    Load, convert, resize, and save one image.
    Returns a result dict.
    """
    result = {
        "source_path": str(src),
        "output_path": str(dst),
        "original_mode":   None,
        "original_size":   None,
        "status":          "ok",
        "error":           "",
    }

    try:
        with Image.open(src) as img:
            result["original_mode"] = img.mode
            result["original_size"] = f"{img.width}x{img.height}"

            # ── 1. Convert to RGB ──────────────────────────────────────────
            if img.mode != "RGB":
                if img.mode == "RGBA":
                    # Composite on white background
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                else:
                    img = img.convert("RGB")

            # ── 2. Resize / crop ──────────────────────────────────────────
            tw, th = target_size
            if strategy == "center_crop":
                img = center_crop(img, tw, th)
            elif strategy == "pad":
                img = pad_to_square(img, tw, th)
            else:  # simple resize
                img = img.resize((tw, th), Image.LANCZOS)

            # ── 3. Strip EXIF ─────────────────────────────────────────────
            img = strip_exif(img)

            # ── 4. Save ───────────────────────────────────────────────────
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst, format="JPEG", quality=JPEG_QUALITY, optimize=True)

    except Exception as e:
        result["status"] = "error"
        result["error"]  = str(e)

    return result


def process_folder(src_dir: str, dst_dir: str, label: str,
                   target_size: tuple, strategy: str) -> list[dict]:
    """Standardise all images in a folder."""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    if not src_path.exists():
        print(f"[WARNING] Source folder not found: {src_dir}")
        return []

    files = sorted(
        f for f in src_path.rglob("*")
        if f.is_file() and not f.name.startswith(".")
        and f.suffix.lower() in SUPPORTED_FORMATS
    )

    total   = len(files)
    results = []
    errors  = 0

    print(f"\n  Standardising [{label}] — {total:,} images → {dst_dir}")
    print(f"  Target: {target_size[0]}×{target_size[1]}px | Strategy: {strategy} | Quality: {JPEG_QUALITY}")

    for i, src_file in enumerate(files, 1):
        new_name = f"{label}_{i:06d}{OUTPUT_EXT}"
        dst_file = dst_path / new_name

        res = standardise_image(src_file, dst_file, target_size, strategy)
        res["label"]    = label
        res["new_name"] = new_name
        results.append(res)

        if res["status"] == "error":
            errors += 1
            print(f"  [ERROR] {src_file.name}: {res['error']}")

        if i % 2000 == 0 or i == total:
            print(f"    Progress: {i:,}/{total:,}  (errors: {errors})")

    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"  Done — {ok_count:,} saved, {errors} errors.")
    return results


def save_metadata_csv(results: list[dict], output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    fpath = os.path.join(output_dir, filename)

    fields = ["label", "new_name", "original_mode", "original_size", "status", "error",
              "source_path", "output_path"]

    with open(fpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"  [✓] Metadata CSV saved → {fpath}  ({len(results):,} rows)")


def main():
    print("\n" + "="*60)
    print("  DWDM PROJECT — STEP 4: STANDARDISATION")
    print("="*60)

    ai_results   = process_folder(AI_IMAGES_DIR,   STD_AI_DIR,   "ai",   TARGET_SIZE, RESIZE_STRATEGY)
    real_results = process_folder(REAL_IMAGES_DIR, STD_REAL_DIR, "real", TARGET_SIZE, RESIZE_STRATEGY)

    all_results = ai_results + real_results
    save_metadata_csv(all_results, OUTPUT_DIR, METADATA_CSV)

    summary = {
        "target_size":      f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
        "resize_strategy":  RESIZE_STRATEGY,
        "jpeg_quality":     JPEG_QUALITY,
        "ai_total":         len(ai_results),
        "real_total":       len(real_results),
        "ai_errors":        sum(1 for r in ai_results   if r["status"] == "error"),
        "real_errors":      sum(1 for r in real_results if r["status"] == "error"),
        "ai_output_dir":    STD_AI_DIR,
        "real_output_dir":  STD_REAL_DIR,
    }
    with open(os.path.join(OUTPUT_DIR, "standardisation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n  ── Standardisation Summary ──")
    for k, v in summary.items():
        print(f"     {k:<22}: {v}")

    print("\n  Step 4 complete. Proceed to step5_eda.py\n")


if __name__ == "__main__":
    main()
