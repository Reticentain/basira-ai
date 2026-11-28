"""
scripts/analyze_failures.py
===========================
Failure analysis and error profiling for trained Diabetic Retinopathy models.

Key functions:
  1. For each dataset and for the frozen operating threshold:
       - Confusion counts per true grade (0 to 4).
       - Per-grade flagged-referable rate with 95% Wilson score confidence intervals.
  2. Identifies and exports lists of the top-K (default K=24) most confident:
       - FALSE NEGATIVES (grade >= 2, lowest calibrated_prob)
       - FALSE POSITIVES (grade 0/1, highest calibrated_prob)
  3. Reports separately missed grade-4 (PDR) and grade-3 (Severe NPDR) cases on eyepacs_test
     (counts and image IDs).
  4. Generates a comprehensive failure_summary.md report with Markdown tables.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig

# Ensure repo root and src/ are on sys.path
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from basira.config import apply_overrides, load_config
from basira.eval.metrics import wilson_score_interval


GRADE_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR (PDR)",
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Failure analysis for Basira AI models.")
    parser.add_argument("--run-name", type=str, required=True, help="Run name (e.g. b0_referable_s42)")
    parser.add_argument("--report-dir", type=str, default=None, help="Directory containing prediction CSVs")
    parser.add_argument("--k", type=int, default=24, help="Top-K false negatives/positives to export (default: 24)")
    parser.add_argument("--threshold", type=float, default=None, help="Operating threshold override")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific datasets to analyze (default: auto-detect from reports directory)",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Config overrides as key=value pairs",
    )
    return parser.parse_args(argv)


def compute_grade_breakdown(
    df: pd.DataFrame,
    threshold: float,
    confidence: float = 0.95,
) -> list[dict[str, Any]]:
    """
    Compute confusion counts and Wilson CI flagged-referable rates for grades 0-4.

    Parameters
    ----------
    df : pd.DataFrame
        Predictions DataFrame with 'grade' and 'calibrated_prob'.
    threshold : float
        Decision threshold.
    confidence : float
        Confidence level for Wilson score interval.

    Returns
    -------
    list of dict
        One dictionary per grade (0 to 4).
    """
    preds_referable = (df["calibrated_prob"].values >= threshold).astype(int)
    grades = df["grade"].values.astype(int)

    breakdown = []
    for g in range(5):
        mask_g = (grades == g)
        total_g = int(np.sum(mask_g))
        if total_g > 0:
            flagged = int(np.sum(preds_referable[mask_g] == 1))
            not_flagged = total_g - flagged
            rate = float(flagged / total_g)
            ci_lower, ci_upper = wilson_score_interval(flagged, total_g, confidence=confidence)
        else:
            flagged = 0
            not_flagged = 0
            rate = float("nan")
            ci_lower = float("nan")
            ci_upper = float("nan")

        breakdown.append({
            "grade": g,
            "grade_name": GRADE_NAMES.get(g, f"Grade {g}"),
            "clinical_status": "Referable (≥2)" if g >= 2 else "Non-Referable (0-1)",
            "total": total_g,
            "flagged_referable": flagged,
            "flagged_non_referable": not_flagged,
            "rate": rate,
            "wilson_ci_lower": ci_lower,
            "wilson_ci_upper": ci_upper,
        })
    return breakdown


def select_top_k_failures(
    df: pd.DataFrame,
    threshold: float,
    k: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select top-K most confident False Negatives and False Positives.

    - False Negatives: True grade >= 2, calibrated_prob < threshold,
      sorted by calibrated_prob ascending (most confident errors first).
    - False Positives: True grade <= 1, calibrated_prob >= threshold,
      sorted by calibrated_prob descending (most confident errors first).

    Returns
    -------
    top_fn : pd.DataFrame
    top_fp : pd.DataFrame
    """
    # False Negatives
    mask_fn = (df["grade"] >= 2) & (df["calibrated_prob"] < threshold)
    df_fn = df[mask_fn].copy()
    df_fn.sort_values(by="calibrated_prob", ascending=True, inplace=True)
    top_fn = df_fn.head(k).reset_index(drop=True)

    # False Positives
    mask_fp = (df["grade"] <= 1) & (df["calibrated_prob"] >= threshold)
    df_fp = df[mask_fp].copy()
    df_fp.sort_values(by="calibrated_prob", ascending=False, inplace=True)
    top_fp = df_fp.head(k).reset_index(drop=True)

    return top_fn, top_fp


def extract_severe_misses_eyepacs(
    df: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    """
    Extract missed Grade 4 (PDR) and Grade 3 (Severe NPDR) cases from EyePACS test.
    """
    mask_g4_miss = (df["grade"] == 4) & (df["calibrated_prob"] < threshold)
    mask_g3_miss = (df["grade"] == 3) & (df["calibrated_prob"] < threshold)

    df_g4 = df[mask_g4_miss].sort_values(by="calibrated_prob", ascending=True)
    df_g3 = df[mask_g3_miss].sort_values(by="calibrated_prob", ascending=True)

    total_g4 = int(np.sum(df["grade"] == 4))
    total_g3 = int(np.sum(df["grade"] == 3))

    return {
        "grade_4": {
            "total_samples": total_g4,
            "missed_count": len(df_g4),
            "miss_rate": float(len(df_g4) / total_g4) if total_g4 > 0 else 0.0,
            "image_ids": df_g4["image_id"].astype(str).tolist(),
            "records": df_g4[["image_id", "grade", "calibrated_prob"]].to_dict(orient="records"),
        },
        "grade_3": {
            "total_samples": total_g3,
            "missed_count": len(df_g3),
            "miss_rate": float(len(df_g3) / total_g3) if total_g3 > 0 else 0.0,
            "image_ids": df_g3["image_id"].astype(str).tolist(),
            "records": df_g3[["image_id", "grade", "calibrated_prob"]].to_dict(orient="records"),
        },
    }


def generate_failure_markdown(
    run_name: str,
    threshold: float,
    all_breakdowns: dict[str, list[dict[str, Any]]],
    eyepacs_severe_misses: Optional[dict[str, Any]],
    k: int,
) -> str:
    """Generate Markdown failure summary document."""
    lines = [
        f"# Basira AI — Failure Analysis & Error Profiling: `{run_name}`",
        "",
        f"- **Operating Decision Threshold:** `{threshold:.4f}` (fit on validation set for ≥90% specificity)",
        f"- **Analysis Target:** Grade-level confusion, 95% Wilson CIs, and top-{k} failure profiling.",
        "",
        "---",
        "",
        "## 1. Grade-Level Confusion & Flagged Referable Rates",
        "",
    ]

    for dname, breakdown in all_breakdowns.items():
        lines.extend([
            f"### Dataset: `{dname}`",
            "",
            "| Grade | Clinical Description | Status | Total N | Flagged Referable | Flagged Non-Referable | Referable Rate | 95% Wilson CI |",
            "|:---:|:---|:---:|---:|---:|---:|---:|:---:|",
        ])
        for row in breakdown:
            g = row["grade"]
            g_name = row["grade_name"]
            status = row["clinical_status"]
            tot = f"{row['total']:,}"
            flagged = f"{row['flagged_referable']:,}"
            non_flagged = f"{row['flagged_non_referable']:,}"
            if not np.isnan(row["rate"]):
                rate_str = f"{row['rate'] * 100:.2f}%"
                ci_str = f"[{row['wilson_ci_lower'] * 100:.2f}%, {row['wilson_ci_upper'] * 100:.2f}%]"
            else:
                rate_str = "N/A"
                ci_str = "N/A"
            lines.append(f"| {g} | {g_name} | {status} | {tot} | {flagged} | {non_flagged} | {rate_str} | {ci_str} |")
        lines.append("")

    if eyepacs_severe_misses:
        g4 = eyepacs_severe_misses["grade_4"]
        g3 = eyepacs_severe_misses["grade_3"]
        lines.extend([
            "---",
            "",
            "## 2. Critical Safety Misses on EyePACS Test Set",
            "",
            "In diabetic retinopathy screening, missing proliferative (Grade 4) or severe non-proliferative (Grade 3) disease "
            "represents the highest clinical safety hazard due to imminent risk of vision loss.",
            "",
            f"- **Grade 4 (Proliferative DR):** {g4['missed_count']} missed out of {g4['total_samples']:,} ({g4['miss_rate'] * 100:.2f}% miss rate)",
            f"- **Grade 3 (Severe NPDR):** {g3['missed_count']} missed out of {g3['total_samples']:,} ({g3['miss_rate'] * 100:.2f}% miss rate)",
            "",
        ])

        if g4["missed_count"] > 0:
            lines.extend([
                "#### Missed Grade-4 Images:",
                "| Image ID | True Grade | Calibrated Prob | Decision Threshold |",
                "|:---|:---:|:---:|:---:|",
            ])
            for rec in g4["records"][:15]:
                lines.append(f"| `{rec['image_id']}` | {rec['grade']} | {rec['calibrated_prob']:.4f} | {threshold:.4f} |")
            if g4["missed_count"] > 15:
                lines.append(f"| *... and {g4['missed_count'] - 15} additional cases* | | | |")
            lines.append("")

        if g3["missed_count"] > 0:
            lines.extend([
                "#### Missed Grade-3 Images (Sample):",
                "| Image ID | True Grade | Calibrated Prob | Decision Threshold |",
                "|:---|:---:|:---:|:---:|",
            ])
            for rec in g3["records"][:15]:
                lines.append(f"| `{rec['image_id']}` | {rec['grade']} | {rec['calibrated_prob']:.4f} | {threshold:.4f} |")
            if g3["missed_count"] > 15:
                lines.append(f"| *... and {g3['missed_count'] - 15} additional cases* | | | |")
            lines.append("")

    lines.extend([
        "---",
        "",
        f"## 3. Top-{k} Failure Exports",
        "",
        f"Lists of the top-{k} most confident false negatives (lowest predicted risk for referable eyes) "
        f"and false positives (highest predicted risk for healthy/mild eyes) have been exported as CSVs under `failures/`.",
        "",
        "These image cohorts serve as the input for Grad-CAM visual interpretability audits and clinician review.",
    ])

    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_config()
    if args.set:
        apply_overrides(cfg, args.set)

    run_name = args.run_name.strip()
    k = int(args.k)

    # 1. Resolve reports directory
    if args.report_dir:
        report_dir = pathlib.Path(args.report_dir)
    else:
        candidates = [
            pathlib.Path(cfg.output.reports) / run_name,
            _REPO_ROOT / "reports" / run_name,
        ]
        report_dir = None
        for c in candidates:
            if c.exists():
                report_dir = c
                break
        if report_dir is None:
            report_dir = pathlib.Path(cfg.output.reports) / run_name
            report_dir.mkdir(parents=True, exist_ok=True)

    # 2. Resolve operating threshold
    threshold = args.threshold
    if threshold is None:
        calib_file = report_dir / "calibration.json"
        if calib_file.exists():
            with open(calib_file, "r", encoding="utf-8") as f:
                calib_data = json.load(f)
            threshold = float(calib_data.get("operating_threshold", 0.2785))
            print(f"Loaded operating threshold {threshold:.4f} from: {calib_file}")
        else:
            threshold = 0.2785
            print(f"Calibration JSON not found at '{calib_file}'. Using fallback threshold: {threshold:.4f}")
    else:
        print(f"Using CLI threshold: {threshold:.4f}")

    # 3. Locate prediction CSVs
    pred_files: dict[str, pathlib.Path] = {}
    if args.datasets:
        for dname in args.datasets:
            candidates = [
                report_dir / f"{dname}_predictions.csv",
                report_dir / f"{dname}.csv",
            ]
            for c in candidates:
                if c.exists():
                    pred_files[dname] = c
                    break
    else:
        for f in report_dir.glob("*_predictions.csv"):
            dname = f.stem.replace("_predictions", "")
            pred_files[dname] = f

    if not pred_files:
        raise FileNotFoundError(
            f"No prediction CSVs found in '{report_dir}'. Please ensure scripts/evaluate.py has been run."
        )

    print("=" * 75)
    print(f"Basira AI — Failure Analysis: {run_name}")
    print(f"Report Dir  : {report_dir}")
    print(f"Threshold   : {threshold:.4f}")
    print(f"Datasets    : {list(pred_files.keys())}")
    print(f"Top-K       : {k}")
    print("=" * 75)

    failures_dir = report_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)

    all_breakdowns: dict[str, list[dict[str, Any]]] = {}
    eyepacs_severe_misses: Optional[dict[str, Any]] = None

    for dname, ppath in pred_files.items():
        print(f"\nProcessing dataset: '{dname}' ({ppath.name})...")
        df = pd.read_csv(ppath)

        # Grade-level confusion breakdown
        breakdown = compute_grade_breakdown(df, threshold=threshold)
        all_breakdowns[dname] = breakdown

        # Top-K False Negatives & False Positives
        top_fn, top_fp = select_top_k_failures(df, threshold=threshold, k=k)

        fn_csv_path = failures_dir / f"{dname}_false_negatives_top{k}.csv"
        fp_csv_path = failures_dir / f"{dname}_false_positives_top{k}.csv"
        top_fn.to_csv(fn_csv_path, index=False)
        top_fp.to_csv(fp_csv_path, index=False)

        print(f"  - Flagged referable per grade:")
        for row in breakdown:
            pct = f"{row['rate'] * 100:.1f}%" if not np.isnan(row["rate"]) else "N/A"
            ci = f"[{row['wilson_ci_lower'] * 100:.1f}%, {row['wilson_ci_upper'] * 100:.1f}%]" if not np.isnan(row["wilson_ci_lower"]) else ""
            print(f"    Grade {row['grade']} ({row['grade_name']}): {row['flagged_referable']}/{row['total']} ({pct}) {ci}")

        print(f"  - Saved top-{k} false negatives: {fn_csv_path} (found {len(top_fn)})")
        print(f"  - Saved top-{k} false positives: {fp_csv_path} (found {len(top_fp)})")

        if dname == "eyepacs_test":
            eyepacs_severe_misses = extract_severe_misses_eyepacs(df, threshold=threshold)
            g4_miss = eyepacs_severe_misses["grade_4"]["missed_count"]
            g4_tot = eyepacs_severe_misses["grade_4"]["total_samples"]
            g3_miss = eyepacs_severe_misses["grade_3"]["missed_count"]
            g3_tot = eyepacs_severe_misses["grade_3"]["total_samples"]
            print(f"  - Safety-critical misses on eyepacs_test:")
            print(f"    * Grade 4 (PDR)        : {g4_miss}/{g4_tot} missed")
            print(f"    * Grade 3 (Severe NPDR): {g3_miss}/{g3_tot} missed")

    # Generate failure_summary.md
    md_content = generate_failure_markdown(
        run_name=run_name,
        threshold=threshold,
        all_breakdowns=all_breakdowns,
        eyepacs_severe_misses=eyepacs_severe_misses,
        k=k,
    )
    summary_md_path = report_dir / "failure_summary.md"
    tmp_md_path = report_dir / f".tmp_{os.getpid()}_failure_summary.md"
    with open(tmp_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    os.replace(tmp_md_path, summary_md_path)
    print(f"\n✓ Saved failure summary markdown to: {summary_md_path}")

    # Export structured JSON
    analysis_payload = {
        "run_name": run_name,
        "operating_threshold": float(threshold),
        "top_k": k,
        "breakdowns": all_breakdowns,
        "eyepacs_test_severe_misses": eyepacs_severe_misses,
    }
    json_path = report_dir / "failure_analysis.json"
    tmp_json_path = report_dir / f".tmp_{os.getpid()}_failure_analysis.json"
    with open(tmp_json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_payload, f, indent=2)
    os.replace(tmp_json_path, json_path)
    print(f"✓ Saved failure analysis JSON to: {json_path}")


if __name__ == "__main__":
    main()
