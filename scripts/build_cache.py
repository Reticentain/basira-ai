"""
scripts/build_cache.py
======================
Build a local cache of Stage-1 preprocessed images (256x256 lossless RGB PNG).

Key characteristics:
  - Keyed by preprocessing settings (<tag>): /content/cache/<tag>/<source>/<image_id stem>.png
  - Atomic file writes (temp file in same folder + os.replace) to prevent truncated files on interrupt
  - Deduplicated by (source, image_id): each unique image is preprocessed and stored exactly once
  - Multiprocessing Pool sized to os.cpu_count(), with cv2.setNumThreads(1) per worker
  - Resumable: skips already-cached files (fast resumption on Colab disconnect)
  - Drive archiving: stores uncompressed ZIP to Google Drive with 2 GB safety margin check
  - --restore: unzips complete archives from Drive, validating file count before skipping build
  - Manifest: writes manifest.json in the tag folder detailing parameters, git commit, and counts
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import pathlib
import shutil
import sys
import zipfile
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Ensure src/ is on sys.path
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from basira.config import load_config
from basira.data.cache import (
    cache_dir,
    cache_path,
    compute_cache_tag,
    write_manifest,
)
from basira.data.loaders import read_split_csv, rebuild_path
from basira.data.preprocess import crop_and_enhance

SUPPORTED_SOURCES = [
    "aptos",
    "eyepacs_train",
    "eyepacs_test",
    "messidor2",
    "idrid",
]


def _worker_init() -> None:
    """Ensure OpenCV threading doesn't conflict with Python multiprocessing."""
    cv2.setNumThreads(1)


def _process_single_image(args: tuple[str, str, str, str, dict[str, Any]]) -> tuple[str, str, Optional[str]]:
    """
    Worker function executed in child processes.

    Parameters
    ----------
    args : tuple
        (source, image_id, raw_path, target_path, preprocess_kwargs)

    Returns
    -------
    status : str
        'written', 'skipped', or 'failed'
    image_id : str
        Image identifier
    error_msg : str or None
        Error details if failed
    """
    _worker_init()
    source, image_id, raw_path, target_path_str, prep_kwargs = args
    target_path = pathlib.Path(target_path_str)

    # Fast skip if already exists and is non-empty
    if target_path.exists() and target_path.stat().st_size > 0:
        return "skipped", image_id, None

    try:
        raw_p = pathlib.Path(raw_path)
        if not raw_p.exists():
            return "failed", image_id, f"Raw image not found: {raw_path}"

        # Open image with PIL, convert to RGB numpy array
        with Image.open(raw_p) as pil_img:
            img_rgb = pil_img.convert("RGB")

        # Run Stage-1 preprocessing
        enhanced = crop_and_enhance(img_rgb, **prep_kwargs)

        # Atomic write: write to temp file in the SAME directory, then os.replace
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.parent / f".tmp_{os.getpid()}_{target_path.name}"
        Image.fromarray(enhanced).save(tmp_path, format="PNG")
        os.replace(tmp_path, target_path)

        return "written", image_id, None
    except Exception as exc:
        return "failed", image_id, str(exc)


def find_split_file(filename: str, cfg: Any) -> pathlib.Path:
    """Search for a split CSV file in Drive output or local reports/splits directory."""
    candidates = [
        pathlib.Path(cfg.output.splits) / filename,
        _REPO_ROOT / "reports" / "splits" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Split file '{filename}' not found. Checked:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + "\nPlease run scripts/create_splits.py first."
    )


def load_unique_source_images(source: str, cfg: Any) -> list[tuple[str, str]]:
    """
    Read split CSVs to determine unique (image_id, raw_path) pairs for a source.

    Returns
    -------
    list of (image_id, raw_path)
    """
    unique_items: dict[str, str] = {}

    if source == "aptos":
        # Look for aptos_deduped.csv first; fallback to combining aptos splits
        try:
            p_dedup = find_split_file("aptos_deduped.csv", cfg)
            df = read_split_csv(p_dedup)
            for _, r in df.iterrows():
                iid = str(r["image_id"])
                unique_items[iid] = rebuild_path("aptos", iid, cfg)
        except FileNotFoundError:
            for fname in ["aptos_train.csv", "aptos_val.csv", "aptos_test.csv"]:
                df = read_split_csv(find_split_file(fname, cfg))
                for _, r in df.iterrows():
                    iid = str(r["image_id"])
                    unique_items[iid] = rebuild_path("aptos", iid, cfg)

    elif source == "eyepacs_train":
        # Combines train and val splits of official EyePACS train
        for fname in ["eyepacs_train.csv", "eyepacs_val.csv"]:
            df = read_split_csv(find_split_file(fname, cfg))
            for _, r in df.iterrows():
                iid = str(r["image_id"])
                unique_items[iid] = rebuild_path("eyepacs_train", iid, cfg)

    elif source == "eyepacs_test":
        df = read_split_csv(find_split_file("eyepacs_test.csv", cfg))
        for _, r in df.iterrows():
            iid = str(r["image_id"])
            unique_items[iid] = rebuild_path("eyepacs_test", iid, cfg)

    elif source == "messidor2":
        try:
            p = find_split_file("messidor2_test.csv", cfg)
        except FileNotFoundError:
            p = find_split_file("messidor2.csv", cfg)
        df = read_split_csv(p)
        for _, r in df.iterrows():
            iid = str(r["image_id"])
            unique_items[iid] = rebuild_path("messidor2", iid, cfg)

    elif source == "idrid":
        try:
            p = find_split_file("idrid_test.csv", cfg)
        except FileNotFoundError:
            p = find_split_file("idrid.csv", cfg)
        df = read_split_csv(p)
        for _, r in df.iterrows():
            iid = str(r["image_id"])
            unique_items[iid] = rebuild_path("idrid", iid, cfg)

    else:
        raise ValueError(f"Unknown source '{source}'")

    return sorted(unique_items.items(), key=lambda x: x[0])


def get_dir_size_bytes(directory: pathlib.Path) -> int:
    """Calculate total size of all regular files in a directory."""
    if not directory.exists():
        return 0
    total = 0
    for root, _, files in os.walk(directory):
        for f in files:
            p = pathlib.Path(root) / f
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def format_bytes(n: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"


def get_drive_free_space(drive_path: pathlib.Path) -> int:
    """Return free disk space in bytes for the drive containing drive_path."""
    curr = drive_path.resolve()
    while not curr.exists() and curr != curr.parent:
        curr = curr.parent
    try:
        return shutil.disk_usage(curr).free
    except Exception:
        # Fallback if shutil.disk_usage fails
        return 0


def try_restore_from_drive(source: str, expected_count: int, cfg: Any) -> bool:
    """
    Attempt to restore source cache from finished Google Drive zip archive.

    Returns True if successfully restored, False if rebuild is needed.
    """
    tag = compute_cache_tag(cfg)
    drive_cache_dir = pathlib.Path(cfg.output.drive_root) / "cache" / tag
    zip_path = drive_cache_dir / f"{source}.zip"

    if not zip_path.exists():
        print(f"[{source}] No finished zip found at {zip_path}. Will build from raw.")
        return False

    dest_dir = cache_dir(cfg) / source
    print(f"[{source}] Found Drive zip: {zip_path} ({format_bytes(zip_path.stat().st_size)})")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = [name for name in zf.namelist() if not name.endswith("/")]
            zip_count = len(namelist)
            if zip_count != expected_count:
                print(
                    f"[{source}] WARNING: Archive contains {zip_count} files, "
                    f"expected {expected_count}. Rejecting zip, will rebuild."
                )
                return False

            print(f"[{source}] Extracting {zip_count} files to {dest_dir}...")
            dest_dir.mkdir(parents=True, exist_ok=True)
            zf.extractall(dest_dir)

        # Verify extracted files on disk
        extracted_pngs = list(dest_dir.glob("*.png"))
        if len(extracted_pngs) == expected_count:
            print(f"[{source}] ✓ Restore verified: {len(extracted_pngs)}/{expected_count} files intact.")
            return True
        else:
            print(
                f"[{source}] WARNING: Extracted count ({len(extracted_pngs)}) != expected ({expected_count}). "
                "Will rebuild."
            )
            return False
    except Exception as exc:
        print(f"[{source}] Error reading archive: {exc}. Will rebuild.")
        return False


def zip_source_to_drive(source: str, source_dir: pathlib.Path, cfg: Any) -> None:
    """
    Create an uncompressed ZIP archive on Google Drive, verifying >= 2 GB free space margin.
    """
    tag = compute_cache_tag(cfg)
    drive_cache_dir = pathlib.Path(cfg.output.drive_root) / "cache" / tag
    drive_cache_dir.mkdir(parents=True, exist_ok=True)

    final_zip = drive_cache_dir / f"{source}.zip"
    partial_zip = drive_cache_dir / f"{source}.zip.partial"

    # Measure source directory size
    source_size = get_dir_size_bytes(source_dir)
    drive_free = get_drive_free_space(drive_cache_dir)

    print(f"\n--- Drive Archiving: {source} ---")
    print(f"  Source size on disk : {format_bytes(source_size)}")
    print(f"  Drive free space    : {format_bytes(drive_free)}")

    min_required_margin = 2 * (1024 ** 3)  # 2 GB
    if drive_free - source_size < min_required_margin:
        print(
            f"  [REFUSED] Archiving refused: Free space after zipping would be "
            f"{format_bytes(drive_free - source_size)}, which is less than the required 2.00 GB margin."
        )
        return

    print(f"  Archiving {source} to {partial_zip.name} (stored, uncompressed)...")
    try:
        with zipfile.ZipFile(partial_zip, "w", compression=zipfile.ZIP_STORED) as zf:
            for root, _, files in os.walk(source_dir):
                for f in files:
                    file_p = pathlib.Path(root) / f
                    arcname = file_p.relative_to(source_dir)
                    zf.write(file_p, arcname=str(arcname))

        # Atomic rename once complete
        if partial_zip.exists():
            os.replace(partial_zip, final_zip)
            print(f"  ✓ Saved to Drive: {final_zip} ({format_bytes(final_zip.stat().st_size)})")
    except Exception as exc:
        print(f"  [ERROR] Archiving failed: {exc}")
        if partial_zip.exists():
            try:
                partial_zip.unlink()
            except OSError:
                pass


def build_cache_for_source(
    source: str,
    cfg: Any,
    pool: multiprocessing.Pool,
    restore: bool = False,
    dry_run: bool = False,
) -> tuple[int, int, list[tuple[str, str]]]:
    """
    Build or restore cache for a single dataset source.

    Returns
    -------
    written_count : int
    skipped_count : int
    failed_items : list of (image_id, error_msg)
    """
    items = load_unique_source_images(source, cfg)
    expected_count = len(items)

    print(f"\n{'=' * 70}")
    print(f"SOURCE: {source.upper()} — {expected_count} unique images expected")
    print(f"{'=' * 70}")

    dest_dir = cache_dir(cfg) / source

    # Try restore if requested
    if restore:
        if dest_dir.exists() and len(list(dest_dir.glob("*.png"))) == expected_count:
            print(f"[{source}] Cache already populated on disk with {expected_count} files. Skipping.")
            return 0, expected_count, []
        restored = try_restore_from_drive(source, expected_count, cfg)
        if restored:
            return 0, expected_count, []

    if dry_run:
        print(f"  [DRY RUN] Would process {expected_count} images for source '{source}'.")
        return 0, 0, []

    prep_kwargs = {
        "target_size": int(cfg.image.size),
        "apply_clahe": bool(cfg.image.apply_clahe),
        "clahe_clip_limit": float(cfg.image.clahe_clip_limit),
        "clahe_tile_grid": tuple(cfg.image.clahe_tile_grid),
        "tol": 10,
    }

    work_items = []
    for image_id, raw_path in items:
        t_path = cache_path(source, image_id, cfg)
        work_items.append((source, image_id, raw_path, str(t_path), prep_kwargs))

    written_count = 0
    skipped_count = 0
    failed_items: list[tuple[str, str]] = []

    dest_dir.mkdir(parents=True, exist_ok=True)

    with tqdm(total=expected_count, desc=f"Caching {source}", unit="img") as pbar:
        for status, iid, err in pool.imap_unordered(_process_single_image, work_items, chunksize=32):
            if status == "written":
                written_count += 1
            elif status == "skipped":
                skipped_count += 1
            elif status == "failed":
                failed_items.append((iid, str(err)))
            pbar.update(1)

    # Verification: check total cached files on disk against expected unique images
    cached_pngs = {p.name for p in dest_dir.glob("*.png")}
    expected_names = {f"{pathlib.Path(iid).stem}.png" for iid, _ in items}
    missing_names = expected_names - cached_pngs

    dir_size = get_dir_size_bytes(dest_dir)
    print(f"\n--- {source} Summary ---")
    print(f"  Written : {written_count}")
    print(f"  Skipped : {skipped_count}")
    print(f"  Failed  : {len(failed_items)}")
    print(f"  Total on disk: {len(cached_pngs)} / {expected_count} ({format_bytes(dir_size)})")

    if missing_names:
        print(f"  [WARNING] {len(missing_names)} expected images are missing from cache!")
        for m in sorted(list(missing_names))[:10]:
            print(f"    - Missing: {m}")
        if len(missing_names) > 10:
            print(f"    ... and {len(missing_names) - 10} more.")

    if failed_items:
        print(f"  [FAILURES LIST] {len(failed_items)} failed:")
        for iid, err in failed_items:
            print(f"    - {iid}: {err}")

    # Zip to Google Drive if no failures and all files present
    if not failed_items and not missing_names and not dry_run:
        zip_source_to_drive(source, dest_dir, cfg)

    return written_count, skipped_count, failed_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage-1 preprocessing cache for Basira AI.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=SUPPORTED_SOURCES,
        help="Specify one or more sources to build (e.g. --only aptos eyepacs_train). Defaults to all.",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore complete archives from Google Drive if local cache is absent.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List work and verify paths without reading/writing cached images.",
    )
    args = parser.parse_args()

    cfg = load_config()
    tag = compute_cache_tag(cfg)
    c_root = cache_dir(cfg)

    print("=" * 70)
    print("BASIRA AI — Stage-1 Preprocessing Cache Builder")
    print(f"Cache tag       : {tag}")
    print(f"Cache directory : {c_root}")
    print(f"CPU count       : {os.cpu_count()}")
    print("=" * 70)

    sources_to_build = args.only if args.only else SUPPORTED_SOURCES

    total_failures: list[tuple[str, str, str]] = []
    source_counts: dict[str, int] = {}

    num_workers = max(1, os.cpu_count() or 1)
    with multiprocessing.Pool(processes=num_workers, initializer=_worker_init) as pool:
        for src in sources_to_build:
            written, skipped, fails = build_cache_for_source(
                src,
                cfg=cfg,
                pool=pool,
                restore=args.restore,
                dry_run=args.dry_run,
            )
            for iid, err in fails:
                total_failures.append((src, iid, err))

            dest = cache_dir(cfg) / src
            source_counts[src] = len(list(dest.glob("*.png"))) if dest.exists() else 0

    if not args.dry_run:
        manifest_p = write_manifest(source_counts, cfg=cfg)
        print(f"\n✓ Manifest written to: {manifest_p}")

    print("\n" + "=" * 70)
    if total_failures:
        print(f"Cache build completed with {len(total_failures)} TOTAL FAILURES across sources.")
        sys.exit(1)
    else:
        print("Cache build completed successfully for all requested sources.")
    print("=" * 70)


if __name__ == "__main__":
    main()
