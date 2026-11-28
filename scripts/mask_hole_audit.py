"""scripts/mask_hole_audit.py – read-only interior-hole audit of cached retina PNGs.

For every cached 256-px image in a run's datasets, computes the fraction of the
filled retina disc that is occupied by interior holes (pixels that are zero inside
the retina boundary).  Joins with the run's saved predictions to report false-negative
and false-positive rates for images with/without holes.

This script is strictly READ-ONLY: no cached PNGs or any cache files are modified.

Usage
-----
    python scripts/mask_hole_audit.py --run-name b0_referable_s42
    python scripts/mask_hole_audit.py --run-name b0_referable_s42 \\
        --datasets eyepacs_test aptos_deduped \\
        --set paths.cache_dir=/mnt/gdrive/cache

Outputs (all in <report_dir>/mask_hole_audit/)
----------------------------------------------
  <dataset>_hole_audit.csv   – per-image metrics
  mask_hole_audit.md         – per-dataset summary table (also saved to report_dir root)
  mask_hole_audit.json       – full structured results for downstream use
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes

# ---------------------------------------------------------------------------
# Repo root bootstrap (same pattern as all other scripts in this repo)
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import pandas as pd  # noqa: E402  (after sys.path setup)

from basira.config import apply_overrides, load_config  # noqa: E402
from basira.data.cache import cache_path  # noqa: E402
from basira.data.dataset import resolve_cache_source  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS = [
    "eyepacs_val",
    "eyepacs_test",
    "aptos_deduped",
    "messidor2_test",
    "idrid_test",
]

HOLE_THRESHOLDS = [0.01, 0.05, 0.10]  # 1%, 5%, 10%

DEFAULT_OPERATING_THRESHOLD = 0.2785

# ---------------------------------------------------------------------------
# Core hole-fraction computation
# ---------------------------------------------------------------------------


def compute_hole_fraction(img_rgb: np.ndarray) -> tuple[float, int, int]:
    """Return ``(hole_fraction, hole_pixels, filled_disc_pixels)`` for a cached image.

    Parameters
    ----------
    img_rgb:
        HxWx3 uint8 numpy array.  Background pixels have all three channels == 0
        (as guaranteed by the Stage-1 preprocessing pipeline).

    Returns
    -------
    hole_fraction:
        ``hole_pixels / filled_disc_pixels``, or 0.0 when the image is all black.
    hole_pixels:
        Count of interior-hole pixels.
    filled_disc_pixels:
        Count of pixels inside the morphologically filled retina disc.
    """
    # Foreground mask: any channel > 0
    nonzero: np.ndarray = np.any(img_rgb > 0, axis=-1)  # (H, W) bool
    filled: np.ndarray = binary_fill_holes(nonzero)      # (H, W) bool
    holes: np.ndarray = filled & ~nonzero                # interior only

    filled_area = int(np.sum(filled))
    hole_area = int(np.sum(holes))

    if filled_area == 0:
        return 0.0, 0, 0

    return float(hole_area / filled_area), hole_area, filled_area


# ---------------------------------------------------------------------------
# Per-dataset audit logic
# ---------------------------------------------------------------------------


def audit_dataset(
    dataset_name: str,
    predictions_csv: pathlib.Path,
    cfg: Any,
    operating_threshold: float,
    out_dir: pathlib.Path,
) -> dict[str, Any]:
    """Audit one dataset.

    Parameters
    ----------
    dataset_name:
        String key (e.g. ``"eyepacs_test"``).
    predictions_csv:
        Path to ``<run_name>/<dataset_name>_predictions.csv``.
    cfg:
        OmegaConf config (for cache path resolution).
    operating_threshold:
        Frozen classification threshold (from calibration.json).
    out_dir:
        Directory for per-dataset output CSV.

    Returns
    -------
    A dict with summary statistics suitable for JSON / markdown serialisation.
    """
    df = pd.read_csv(predictions_csv)

    records: list[dict[str, Any]] = []
    skipped = 0

    for _, row in df.iterrows():
        img_id = str(row["image_id"])
        raw_src = str(row.get("source", dataset_name))

        try:
            c_src = resolve_cache_source(raw_src)
        except Exception:
            c_src = raw_src

        try:
            c_path = pathlib.Path(cache_path(c_src, img_id, cfg))
        except Exception:
            skipped += 1
            continue

        if not c_path.exists():
            skipped += 1
            continue

        try:
            img = np.array(Image.open(c_path).convert("RGB"), dtype=np.uint8)
        except Exception:
            skipped += 1
            continue

        hole_frac, hole_px, filled_px = compute_hole_fraction(img)

        records.append(
            {
                "image_id": img_id,
                "dataset": dataset_name,
                "grade": int(row["grade"]),
                "calibrated_prob": float(row["calibrated_prob"]),
                "hole_fraction": hole_frac,
                "hole_pixels": hole_px,
                "filled_pixels": filled_px,
            }
        )

    if not records:
        return _empty_dataset_result(dataset_name, skipped)

    audit_df = pd.DataFrame(records)
    audit_df["is_referable"] = (audit_df["grade"] >= 2).astype(int)
    audit_df["predicted_referable"] = (
        audit_df["calibrated_prob"] >= operating_threshold
    ).astype(int)

    # ---- Write per-dataset CSV (atomic) ----
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{dataset_name}_hole_audit.csv"
    _atomic_write_csv(audit_df, csv_path)

    # ---- Summary statistics ----
    n_total = len(audit_df)
    fracs = audit_df["hole_fraction"].values

    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "n_total": n_total,
        "n_skipped": skipped,
        "mean_hole_fraction": float(np.mean(fracs)),
        "median_hole_fraction": float(np.median(fracs)),
    }

    for thresh in HOLE_THRESHOLDS:
        key = f"n_hole_gt{int(thresh*100)}pct"
        summary[key] = int(np.sum(fracs > thresh))

    # ---- FN / FP analysis per hole threshold ----
    fn_fp: dict[str, Any] = {}
    for thresh in HOLE_THRESHOLDS:
        label = f"gt{int(thresh*100)}pct"
        has_holes = audit_df["hole_fraction"] > thresh
        fn_fp[label] = {
            "has_holes": _fn_fp_rates(audit_df[has_holes]),
            "no_holes": _fn_fp_rates(audit_df[~has_holes]),
        }
    summary["fn_fp_by_hole_threshold"] = fn_fp

    return summary


def _fn_fp_rates(sub: pd.DataFrame) -> dict[str, Any]:
    """Compute FN and FP rates for a subset of the audit DataFrame."""
    if len(sub) == 0:
        return {"n": 0, "fn_rate": None, "fp_rate": None}

    referable = sub["is_referable"] == 1
    pred_referable = sub["predicted_referable"] == 1

    tp = int(np.sum(referable & pred_referable))
    fn = int(np.sum(referable & ~pred_referable))
    fp = int(np.sum(~referable & pred_referable))
    tn = int(np.sum(~referable & ~pred_referable))

    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else None
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else None

    return {
        "n": len(sub),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "fn_rate": fn_rate,
        "fp_rate": fp_rate,
    }


def _empty_dataset_result(dataset_name: str, skipped: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "n_total": 0,
        "n_skipped": skipped,
        "mean_hole_fraction": None,
        "median_hole_fraction": None,
    }
    for thresh in HOLE_THRESHOLDS:
        summary[f"n_hole_gt{int(thresh*100)}pct"] = 0
    summary["fn_fp_by_hole_threshold"] = {}
    return summary


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------


def build_markdown(results: list[dict[str, Any]], operating_threshold: float) -> str:
    lines: list[str] = []
    lines.append("# Mask Interior-Hole Audit")
    lines.append("")
    lines.append(
        "> **Read-only audit.** No cached PNGs or cache files were modified. "
        "Interior holes are defined as pixels that are zero (background) but lie "
        "inside the morphologically filled retina disc boundary."
    )
    lines.append("")
    lines.append(f"Operating threshold: **{operating_threshold:.4f}**")
    lines.append("")

    # --- Summary table ---
    lines.append("## Per-Dataset Hole Statistics")
    lines.append("")
    lines.append(
        "| Dataset | N imgs | Mean hole% | Median hole% "
        "| >1% | >5% | >10% |"
    )
    lines.append("|---------|--------|-----------|--------------|-----|-----|------|")
    for r in results:
        name = r["dataset"]
        n = r["n_total"]
        mean_h = (
            f"{r['mean_hole_fraction']*100:.3f}"
            if r["mean_hole_fraction"] is not None
            else "—"
        )
        med_h = (
            f"{r['median_hole_fraction']*100:.3f}"
            if r["median_hole_fraction"] is not None
            else "—"
        )
        gt1 = r.get("n_hole_gt1pct", 0)
        gt5 = r.get("n_hole_gt5pct", 0)
        gt10 = r.get("n_hole_gt10pct", 0)
        lines.append(
            f"| {name} | {n} | {mean_h} | {med_h} | {gt1} | {gt5} | {gt10} |"
        )
    lines.append("")

    # --- FN / FP tables ---
    lines.append("## FN / FP Rates by Hole Presence")
    lines.append("")
    lines.append(
        "FN rate = missed referables / all referables.  "
        "FP rate = false alarms / all non-referables."
    )
    lines.append("")

    for thresh_label, thresh_pct in [("gt1pct", "1%"), ("gt5pct", "5%"), ("gt10pct", "10%")]:
        lines.append(f"### Hole fraction > {thresh_pct}")
        lines.append("")
        lines.append(
            "| Dataset | Group | N | FN rate | FP rate |"
        )
        lines.append("|---------|-------|---|---------|---------|")
        for r in results:
            fn_fp = r.get("fn_fp_by_hole_threshold", {}).get(thresh_label, {})
            if not fn_fp:
                continue
            for group_key, group_label in [("has_holes", "has holes"), ("no_holes", "no holes")]:
                g = fn_fp.get(group_key, {})
                n_g = g.get("n", 0)
                fn_r = g.get("fn_rate")
                fp_r = g.get("fp_rate")
                fn_str = f"{fn_r:.3f}" if fn_r is not None else "—"
                fp_str = f"{fp_r:.3f}" if fp_r is not None else "—"
                lines.append(
                    f"| {r['dataset']} | {group_label} | {n_g} | {fn_str} | {fp_str} |"
                )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Atomic file helpers
# ---------------------------------------------------------------------------


def _atomic_write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{os.getpid()}_{path.name}"
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: pathlib.Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{os.getpid()}_{path.name}"
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{os.getpid()}_{path.name}"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only interior-hole audit of cached retina PNGs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-name", required=True, help="Run name (directory under reports/).")
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Override for the report directory (default: cfg.output.reports/<run-name>).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="Dataset names to audit.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Config key=value overrides.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    cfg = load_config()
    if args.overrides:
        apply_overrides(cfg, args.overrides)

    # Resolve report directory
    if args.report_dir:
        report_dir = pathlib.Path(args.report_dir)
    else:
        report_dir = pathlib.Path(cfg.output.reports) / args.run_name

    # Load operating threshold from calibration.json if present
    cal_path = report_dir / "calibration.json"
    operating_threshold = DEFAULT_OPERATING_THRESHOLD
    if cal_path.exists():
        with open(cal_path, encoding="utf-8") as fh:
            cal = json.load(fh)
        operating_threshold = float(
            cal.get("operating_threshold", cal.get("threshold", DEFAULT_OPERATING_THRESHOLD))
        )

    out_dir = report_dir / "mask_hole_audit"

    results: list[dict[str, Any]] = []

    for dname in args.datasets:
        pred_csv = report_dir / f"{dname}_predictions.csv"
        if not pred_csv.exists():
            print(f"[mask_hole_audit] {dname}: predictions CSV not found, skipping.")
            results.append(_empty_dataset_result(dname, 0))
            continue

        print(f"[mask_hole_audit] Auditing {dname} …")
        summary = audit_dataset(
            dataset_name=dname,
            predictions_csv=pred_csv,
            cfg=cfg,
            operating_threshold=operating_threshold,
            out_dir=out_dir,
        )
        results.append(summary)
        n = summary["n_total"]
        sk = summary["n_skipped"]
        mean_h = summary["mean_hole_fraction"]
        mean_str = f"{mean_h*100:.3f}%" if mean_h is not None else "n/a"
        print(
            f"  {dname}: {n} images audited, {sk} skipped, mean hole fraction={mean_str}"
        )

    # Write markdown to both out_dir and report_dir root
    md_content = build_markdown(results, operating_threshold)
    _atomic_write_text(out_dir / "mask_hole_audit.md", md_content)
    _atomic_write_text(report_dir / "mask_hole_audit.md", md_content)

    # Write JSON
    _atomic_write_json(
        out_dir / "mask_hole_audit.json",
        {
            "run_name": args.run_name,
            "operating_threshold": operating_threshold,
            "datasets": results,
        },
    )

    print(f"\n[mask_hole_audit] Done. Reports written to {report_dir / 'mask_hole_audit.md'}")


if __name__ == "__main__":
    main()
