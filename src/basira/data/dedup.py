"""
src/basira/data/dedup.py
========================
Perceptual hash deduplication for APTOS 2019 using pure PIL + NumPy.

Design specifications:
  - Hash is computed AFTER circular border crop & square padding.
  - Uses 16x16 (256-bit) difference hash (dHash).
  - Evaluates nearest-neighbour Hamming distance distribution across the dataset.
  - Inspects flagged candidate duplicate pairs for grade conflict.
  - Group resolution:
      * If grades conflict within a duplicate group -> DROP ENTIRE GROUP (label noise/unsafe).
      * If grades agree -> KEEP ONE representative image, drop duplicates.
  - Saves side-by-side comparison figure (~20 pairs) to cfg.output.reports.
  - Saves full deduplication statistics and list of dropped images.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from PIL import Image

from basira.config import load_config
from basira.data.preprocess import crop_black_borders, pad_to_square


def compute_dhash_256(img: Image.Image | np.ndarray, hash_size: int = 16) -> np.ndarray:
    """
    Compute a 256-bit (hash_size x hash_size) difference hash (dHash).
    Applied after black border crop and square padding.

    Parameters
    ----------
    img : PIL.Image or np.ndarray
        RGB image.
    hash_size : int
        Dimension of square hash (16 -> 256 bits).

    Returns
    -------
    np.ndarray
        Boolean array of shape (hash_size * hash_size,) representing 256 bits.
    """
    if isinstance(img, Image.Image):
        img_np = np.array(img.convert("RGB"))
    else:
        img_np = img

    # 1. Border crop and square pad before hashing
    cropped = crop_black_borders(img_np)
    squared = pad_to_square(cropped)

    # 2. Convert to grayscale via standard luminance weights
    pil_img = Image.fromarray(squared).convert("L")

    # 3. Resize to (hash_size + 1, hash_size): 17 columns, 16 rows
    resized = pil_img.resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.int16)

    # 4. Compare adjacent horizontal pixels (16 comparisons per row -> 256 bits total)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten()


def compute_all_hashes(df: pd.DataFrame) -> np.ndarray:
    """Compute 256-bit dHash for all images listed in DataFrame `df`."""
    hashes = []
    for idx, row in df.iterrows():
        img_path = row["path"]
        try:
            with Image.open(img_path) as im:
                h = compute_dhash_256(im, hash_size=16)
                hashes.append(h)
        except Exception as e:
            # Fallback for unreadable images: random/zeros
            hashes.append(np.zeros(256, dtype=bool))
    return np.array(hashes, dtype=bool)


def compute_nn_distances(hashes: np.ndarray, batch_size: int = 250) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute nearest-neighbour Hamming distances for all images in batches
    to maintain low memory footprint in Colab.

    Returns
    -------
    nn_distances : np.ndarray of shape (N,)
        Distance to closest other image.
    nn_indices : np.ndarray of shape (N,)
        Index of closest other image.
    """
    n = len(hashes)
    nn_distances = np.full(n, fill_value=256, dtype=np.int32)
    nn_indices = np.zeros(n, dtype=np.int32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # Shape: (B, 1, 256) != (1, N, 256) -> (B, N)
        batch = hashes[start:end]
        dist_batch = np.count_nonzero(batch[:, None, :] != hashes[None, :, :], axis=2)

        # Mask diagonal (distance to self is 0)
        for i_local, i_global in enumerate(range(start, end)):
            dist_batch[i_local, i_global] = 256

        batch_min_dist = dist_batch.min(axis=1)
        batch_min_idx = dist_batch.argmin(axis=1)

        nn_distances[start:end] = batch_min_dist
        nn_indices[start:end] = batch_min_idx

    return nn_distances, nn_indices


def find_duplicate_pairs(
    hashes: np.ndarray,
    threshold: int,
    batch_size: int = 250,
) -> List[Tuple[int, int, int]]:
    """Find all unique pairs (i, j) with i < j having Hamming distance <= threshold."""
    n = len(hashes)
    pairs = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = hashes[start:end]
        dist_batch = np.count_nonzero(batch[:, None, :] != hashes[None, :, :], axis=2)

        for i_local, i_global in enumerate(range(start, end)):
            # Look only at j > i_global
            matches = np.where(dist_batch[i_local, i_global + 1 :] <= threshold)[0]
            for m in matches:
                j_global = i_global + 1 + int(m)
                pairs.append((i_global, j_global, int(dist_batch[i_local, j_global])))

    return pairs


def save_flagged_pairs_figure(
    df: pd.DataFrame,
    pairs: List[Tuple[int, int, int]],
    save_path: pathlib.Path,
    max_pairs: int = 20,
) -> None:
    """Save side-by-side comparison visualization for up to max_pairs flagged duplicate pairs."""
    if not pairs:
        return

    n_show = min(len(pairs), max_pairs)
    fig, axes = plt.subplots(n_show, 2, figsize=(7, 2.5 * n_show), squeeze=False)

    for k in range(n_show):
        i, j, dist = pairs[k]
        row_i = df.iloc[i]
        row_j = df.iloc[j]

        conflict = row_i["grade"] != row_j["grade"]
        status = "CONFLICT" if conflict else "AGREE"

        try:
            im_i = Image.open(row_i["path"]).convert("RGB")
            axes[k, 0].imshow(im_i)
        except Exception:
            axes[k, 0].text(0.5, 0.5, "Image Load Error", ha="center")
        axes[k, 0].set_title(f"A: {row_i['image_id']} | Grade {row_i['grade']}", fontsize=9)
        axes[k, 0].axis("off")

        try:
            im_j = Image.open(row_j["path"]).convert("RGB")
            axes[k, 1].imshow(im_j)
        except Exception:
            axes[k, 1].text(0.5, 0.5, "Image Load Error", ha="center")
        axes[k, 1].set_title(f"B: {row_j['image_id']} | Grade {row_j['grade']} (d={dist}, {status})", fontsize=9,
                             color="red" if conflict else "green")
        axes[k, 1].axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def deduplicate_aptos(
    df_aptos: pd.DataFrame,
    cfg: Optional[DictConfig] = None,
    distance_threshold: int = 6,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main entry point for APTOS deduplication.

    Parameters
    ----------
    df_aptos : pd.DataFrame
        Unified DataFrame with columns: path, image_id, grade, referable, source, etc.
    cfg : DictConfig, optional
        Loaded configuration.
    distance_threshold : int
        Maximum Hamming distance (out of 256) to be considered duplicate. Default 6 bits.

    Returns
    -------
    df_clean : pd.DataFrame
        Deduplicated DataFrame.
    summary : dict
        Deduplication metrics, NN distance stats, and list of dropped image IDs.
    """
    if cfg is None:
        cfg = load_config()

    reports_dir = pathlib.Path(cfg.output.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DEDUP] Computing 256-bit dHash for {len(df_aptos)} images...")
    hashes = compute_all_hashes(df_aptos)

    # 1. Nearest-neighbour distribution
    print("[DEDUP] Computing nearest-neighbour Hamming distances...")
    nn_dist, nn_idx = compute_nn_distances(hashes)

    dist_stats = {
        "min": int(np.min(nn_dist)),
        "p10": float(np.percentile(nn_dist, 10)),
        "p25": float(np.percentile(nn_dist, 25)),
        "median": float(np.median(nn_dist)),
        "p75": float(np.percentile(nn_dist, 75)),
        "max": int(np.max(nn_dist)),
    }
    print(f"[DEDUP] NN Distance Distribution: {dist_stats}")

    # 2. Find candidate pairs below threshold
    pairs = find_duplicate_pairs(hashes, threshold=distance_threshold)
    print(f"[DEDUP] Found {len(pairs)} candidate duplicate pairs at Hamming dist <= {distance_threshold}")

    # 3. Check for grade conflicts
    conflicting_pairs = 0
    for i, j, dist in pairs:
        if df_aptos.iloc[i]["grade"] != df_aptos.iloc[j]["grade"]:
            conflicting_pairs += 1
    print(f"[DEDUP] Candidate pairs with conflicting DR grades: {conflicting_pairs}")

    # 4. Save side-by-side visualization of flagged pairs (~20 pairs)
    fig_path = reports_dir / "aptos_flagged_duplicates.png"
    save_flagged_pairs_figure(df_aptos, pairs, fig_path, max_pairs=20)
    print(f"[DEDUP] Saved flagged pairs side-by-side figure to: {fig_path}")

    # 5. Connected components resolution
    # Build adjacency
    from collections import defaultdict
    adj = defaultdict(set)
    for i, j, _ in pairs:
        adj[i].add(j)
        adj[j].add(i)

    visited = set()
    drop_indices = set()
    conflict_dropped_count = 0
    agreement_dropped_count = 0

    for idx in range(len(df_aptos)):
        if idx in visited:
            continue
        if idx not in adj:
            continue

        # BFS component discovery
        component = []
        queue = [idx]
        visited.add(idx)

        while queue:
            curr = queue.pop(0)
            component.append(curr)
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Inspect component grades
        grades = [df_aptos.iloc[k]["grade"] for k in component]
        all_same_grade = len(set(grades)) == 1

        if not all_same_grade:
            # Rule: DROP ENTIRE GROUP if grades conflict (noisy labels)
            for k in component:
                drop_indices.add(k)
            conflict_dropped_count += len(component)
        else:
            # Rule: KEEP ONE, DROP REST if grades agree
            # Keep first by index
            for k in component[1:]:
                drop_indices.add(k)
            agreement_dropped_count += (len(component) - 1)

    df_clean = df_aptos.drop(index=list(drop_indices)).reset_index(drop=True)
    dropped_image_ids = df_aptos.iloc[list(drop_indices)]["image_id"].tolist()

    summary = {
        "original_count": len(df_aptos),
        "deduped_count": len(df_clean),
        "total_dropped": len(drop_indices),
        "conflict_dropped": conflict_dropped_count,
        "agreement_dropped": agreement_dropped_count,
        "distance_threshold": distance_threshold,
        "nn_distance_distribution": dist_stats,
        "conflicting_pairs_count": conflicting_pairs,
        "dropped_image_ids": dropped_image_ids,
        "flagged_figure_path": str(fig_path),
    }

    # Save summary to reports path from config
    summary_path = reports_dir / "aptos_dedup_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[DEDUP] Deduplication complete: {len(df_aptos)} -> {len(df_clean)} images "
          f"({len(drop_indices)} dropped: {conflict_dropped_count} by conflict, {agreement_dropped_count} by agreement).")
    print(f"[DEDUP] Saved summary to: {summary_path}")

    return df_clean, summary
