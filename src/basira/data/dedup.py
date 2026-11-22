"""
src/basira/data/dedup.py
========================
Perceptual hash deduplication for APTOS 2019 using pure PIL + NumPy.

Design specifications:
  - Hash is computed AFTER circular border crop & square padding.
  - Uses 16x16 (256-bit) difference hash (dHash).
  - Caches computed hashes to `cfg.output.reports/aptos_hashes_256.npy` so threshold
    tuning does not recompute them.
  - Prints counts of candidate duplicate pairs at each Hamming distance: 0, 1, ..., 8.
  - Generates calibration figure showing 4 random pairs for EACH distance from 0 to 8
    with DR grades and distance labels, clearly visualizing where retinas cease being identical.
  - Threshold is configurable via `cfg.dedup.max_hamming` (never hardcoded).
  - Group resolution:
      * If grades conflict within a duplicate group -> DROP ENTIRE GROUP (label noise/unsafe).
      * If grades agree -> KEEP ONE representative image, drop duplicates.
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

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


def compute_all_hashes(
    df: pd.DataFrame,
    cache_path: Optional[pathlib.Path] = None,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute or load cached 256-bit dHash for all images listed in DataFrame `df`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least the 'path' column.
    cache_path : pathlib.Path, optional
        Path to .npy cache file. If it exists and matches row count, it is loaded.
    show_progress : bool
        Whether to show tqdm progress bar on computation.

    Returns
    -------
    np.ndarray of shape (N, 256), dtype bool.
    """
    if cache_path is not None and cache_path.exists():
        try:
            cached = np.load(cache_path)
            if len(cached) == len(df) and cached.shape[1] == 256:
                print(f"[DEDUP] Loaded {len(cached)} cached 256-bit hashes from: {cache_path}")
                return cached.astype(bool)
            else:
                print(f"[DEDUP] Cache shape mismatch ({cached.shape} vs {len(df)}, 256). Recomputing...")
        except Exception as e:
            print(f"[DEDUP] Failed to load cache ({e}). Recomputing...")

    hashes = []
    iterator = tqdm(df.iterrows(), total=len(df), desc="[DEDUP] Computing 256-bit dHash", disable=not show_progress)
    for idx, row in iterator:
        img_path = row["path"]
        try:
            with Image.open(img_path) as im:
                h = compute_dhash_256(im, hash_size=16)
                hashes.append(h)
        except Exception:
            # Fallback for unreadable images: zeros
            hashes.append(np.zeros(256, dtype=bool))

    hashes_arr = np.array(hashes, dtype=bool)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, hashes_arr)
        print(f"[DEDUP] Cached {len(hashes_arr)} computed hashes to: {cache_path}")

    return hashes_arr


def compute_nn_distances(hashes: np.ndarray, batch_size: int = 250) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute nearest-neighbour Hamming distances for all images in batches.

    Returns
    -------
    nn_distances : np.ndarray of shape (N,)
    nn_indices : np.ndarray of shape (N,)
    """
    n = len(hashes)
    nn_distances = np.full(n, fill_value=256, dtype=np.int32)
    nn_indices = np.zeros(n, dtype=np.int32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = hashes[start:end]
        dist_batch = np.count_nonzero(batch[:, None, :] != hashes[None, :, :], axis=2)

        for i_local, i_global in enumerate(range(start, end)):
            dist_batch[i_local, i_global] = 256

        batch_min_dist = dist_batch.min(axis=1)
        batch_min_idx = dist_batch.argmin(axis=1)

        nn_distances[start:end] = batch_min_dist
        nn_indices[start:end] = batch_min_idx

    return nn_distances, nn_indices


def find_pairs_grouped_by_distance(
    hashes: np.ndarray,
    max_dist: int = 8,
    batch_size: int = 250,
) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    Find all unique image pairs (i, j) with i < j having Hamming distance <= max_dist,
    grouped by exact distance d in {0, 1, ..., max_dist}.
    """
    n = len(hashes)
    pairs_by_dist: Dict[int, List[Tuple[int, int, int]]] = {d: [] for d in range(max_dist + 1)}

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = hashes[start:end]
        dist_batch = np.count_nonzero(batch[:, None, :] != hashes[None, :, :], axis=2)

        for i_local, i_global in enumerate(range(start, end)):
            for d in range(max_dist + 1):
                matches = np.where(dist_batch[i_local, i_global + 1 :] == d)[0]
                for m in matches:
                    j_global = i_global + 1 + int(m)
                    pairs_by_dist[d].append((i_global, j_global, d))

    return pairs_by_dist


def save_pairs_by_distance_figure(
    df: pd.DataFrame,
    pairs_by_dist: Dict[int, List[Tuple[int, int, int]]],
    save_path: pathlib.Path,
    pairs_per_dist: int = 4,
    seed: int = 42,
) -> None:
    """
    Save a calibration figure with up to `pairs_per_dist` random pairs for EACH distance
    from 0 to 8, labeled with their distance and DR grades.
    """
    selected_pairs: List[Tuple[int, int, int]] = []
    rng = np.random.default_rng(seed)

    for d in sorted(pairs_by_dist.keys()):
        p_list = pairs_by_dist[d]
        if not p_list:
            continue
        if len(p_list) <= pairs_per_dist:
            chosen = p_list
        else:
            chosen_idx = rng.choice(len(p_list), size=pairs_per_dist, replace=False)
            chosen = [p_list[k] for k in chosen_idx]
        selected_pairs.extend(chosen)

    if not selected_pairs:
        print("[DEDUP] No candidate pairs found to plot.")
        return

    m = len(selected_pairs)
    fig, axes = plt.subplots(m, 2, figsize=(8, 2.5 * m), squeeze=False)

    for k, (i, j, dist) in enumerate(selected_pairs):
        row_i = df.iloc[i]
        row_j = df.iloc[j]

        conflict = row_i["grade"] != row_j["grade"]
        status = "CONFLICT" if conflict else "AGREE"
        status_color = "red" if conflict else "green"

        try:
            im_i = Image.open(row_i["path"]).convert("RGB")
            axes[k, 0].imshow(im_i)
        except Exception:
            axes[k, 0].text(0.5, 0.5, "Image Load Error", ha="center")
        axes[k, 0].set_title(f"[d={dist}] A: {row_i['image_id']} (Grade {row_i['grade']})", fontsize=9, fontweight="bold")
        axes[k, 0].axis("off")

        try:
            im_j = Image.open(row_j["path"]).convert("RGB")
            axes[k, 1].imshow(im_j)
        except Exception:
            axes[k, 1].text(0.5, 0.5, "Image Load Error", ha="center")
        axes[k, 1].set_title(
            f"[d={dist}] B: {row_j['image_id']} (Grade {row_j['grade']}) -> {status}",
            fontsize=9,
            fontweight="bold",
            color=status_color,
        )
        axes[k, 1].axis("off")

    plt.suptitle("APTOS Candidate Pairs by Hamming Distance (0 to 8) - Calibration Grid", fontsize=11, y=1.00)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[DEDUP] Saved distance calibration figure ({m} pairs across distances) to: {save_path}")


def deduplicate_aptos(
    df_aptos: pd.DataFrame,
    cfg: Optional[DictConfig] = None,
    distance_threshold: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main entry point for APTOS deduplication with cached hashes and distance-based reporting.

    Parameters
    ----------
    df_aptos : pd.DataFrame
        Unified DataFrame with columns: path, image_id, grade, referable, source, etc.
    cfg : DictConfig, optional
        Loaded configuration.
    distance_threshold : int, optional
        Threshold distance. If None, read from `cfg.dedup.max_hamming`.

    Returns
    -------
    df_clean : pd.DataFrame
        Deduplicated DataFrame.
    summary : dict
        Deduplication metrics, pair counts per distance, and list of dropped image IDs.
    """
    if cfg is None:
        cfg = load_config()

    reports_dir = pathlib.Path(cfg.output.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve distance threshold from config (never hardcoded)
    if distance_threshold is None:
        if hasattr(cfg, "dedup") and hasattr(cfg.dedup, "max_hamming"):
            distance_threshold = int(cfg.dedup.max_hamming)
        else:
            distance_threshold = 8
    print(f"[DEDUP] Active Hamming distance threshold: {distance_threshold}")

    # 2. Compute or load cached 256-bit hashes
    cache_file = reports_dir / "aptos_hashes_256.npy"
    hashes = compute_all_hashes(df_aptos, cache_path=cache_file, show_progress=True)

    # 3. Nearest-neighbour distribution
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

    # 4. Find pairs across range 0 to 8
    max_range = max(8, distance_threshold)
    pairs_by_dist = find_pairs_grouped_by_distance(hashes, max_dist=max_range)

    print("\n[DEDUP] Candidate pair counts by Hamming distance:")
    print("-" * 45)
    counts_by_distance: Dict[int, int] = {}
    for d in range(max_range + 1):
        cnt = len(pairs_by_dist[d])
        counts_by_distance[d] = cnt
        status_flag = " [CONSIDERED DUPLICATE]" if d <= distance_threshold else " [RETAINED]"
        print(f"  d = {d:>2} : {cnt:>5} pairs{status_flag}")
    print("-" * 45)

    # 5. Save calibration figure with 4 random pairs for EACH distance from 0 to 8
    fig_path = reports_dir / "aptos_flagged_duplicates_by_distance.png"
    save_pairs_by_distance_figure(df_aptos, pairs_by_dist, fig_path, pairs_per_dist=4, seed=int(cfg.seeds.global_seed))

    # Also save as aptos_flagged_duplicates.png for standard reference
    standard_fig_path = reports_dir / "aptos_flagged_duplicates.png"
    if fig_path.exists() and fig_path != standard_fig_path:
        import shutil
        shutil.copyfile(fig_path, standard_fig_path)

    # 6. Gather all candidate duplicate pairs up to active threshold
    active_pairs: List[Tuple[int, int, int]] = []
    for d in range(distance_threshold + 1):
        active_pairs.extend(pairs_by_dist[d])

    print(f"[DEDUP] Total candidate duplicate pairs at d <= {distance_threshold}: {len(active_pairs)}")

    # 7. Check for grade conflicts
    conflicting_pairs = 0
    for i, j, dist in active_pairs:
        if df_aptos.iloc[i]["grade"] != df_aptos.iloc[j]["grade"]:
            conflicting_pairs += 1
    print(f"[DEDUP] Candidate duplicate pairs with conflicting DR grades: {conflicting_pairs}")

    # 8. Connected components resolution
    adj = defaultdict(set)
    for i, j, _ in active_pairs:
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

        grades = [df_aptos.iloc[k]["grade"] for k in component]
        all_same_grade = len(set(grades)) == 1

        if not all_same_grade:
            # Rule: DROP ENTIRE GROUP if grades conflict (noisy labels)
            for k in component:
                drop_indices.add(k)
            conflict_dropped_count += len(component)
        else:
            # Rule: KEEP ONE, DROP REST if grades agree
            for k in component[1:]:
                drop_indices.add(k)
            agreement_dropped_count += len(component) - 1

    df_clean = df_aptos.drop(index=list(drop_indices)).reset_index(drop=True)
    dropped_image_ids = df_aptos.iloc[list(drop_indices)]["image_id"].tolist()

    summary = {
        "original_count": len(df_aptos),
        "deduped_count": len(df_clean),
        "total_dropped": len(drop_indices),
        "conflict_dropped": conflict_dropped_count,
        "agreement_dropped": agreement_dropped_count,
        "distance_threshold": distance_threshold,
        "counts_by_distance": counts_by_distance,
        "nn_distance_distribution": dist_stats,
        "conflicting_pairs_count": conflicting_pairs,
        "dropped_image_ids": dropped_image_ids,
        "flagged_figure_path": str(fig_path),
        "hashes_cache_path": str(cache_file),
    }

    # Save summary to reports path from config
    summary_path = reports_dir / "aptos_dedup_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[DEDUP] Deduplication complete: {len(df_aptos)} -> {len(df_clean)} images "
        f"({len(drop_indices)} dropped: {conflict_dropped_count} by conflict, {agreement_dropped_count} by agreement)."
    )
    print(f"[DEDUP] Saved summary to: {summary_path}")

    return df_clean, summary
