"""
scripts/quality_analysis.py
===========================
Image quality analysis, distribution shifts, and Quality Gate evaluation.

Key functionality:
  1. Computes retina-specific image quality metrics from cached 256px PNGs:
     - Sharpness (variance of Laplacian on the green channel)
     - Mean Brightness (mean pixel intensity on retina)
     - Contrast (std of luminance on retina)
     - Retina Fill Fraction (fraction of 256x256 image covered by retina)
     - Mean R, G, B & Color Ratios (quantifying camera color cast shifts)
  2. Generates distribution plots across datasets to quantify camera/spectral shift.
  3. Joins quality metrics with prediction reports:
     - Diagnostic performance (AUC, Sens/Spec @ frozen threshold) per quality quartile.
     - Enrichment analysis: How enriched low-quality images are among False Negatives & False Positives.
  4. Quality Gate:
     - Composite quality score with rejection thresholds fit on EyePACS VAL only.
     - Evaluates coverage vs. diagnostic performance trade-offs across all external test sets.

Outputs:
  - Drive reports/<run-name>/quality/<dataset>_quality.csv
  - Drive reports/<run-name>/quality_analysis.md
  - Drive reports/<run-name>/quality_gate_table.md
  - Drive reports/<run-name>/quality_results.json
  - Drive reports/<run-name>/figures/quality_distributions.png
  - Drive reports/<run-name>/figures/quality_gate_coverage.png
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Repository bootstrap
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from basira.config import apply_overrides, load_config
from basira.data.cache import cache_path
from basira.data.dataset import resolve_cache_source
from basira.eval.metrics import compute_auc
from basira.eval.quality import QualityGate, compute_image_quality_metrics


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fundus image quality analysis and Quality Gate screening.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-name", type=str, required=True, help="Run name (e.g. b0_referable_s42)")
    parser.add_argument("--report-dir", type=str, default=None, help="Directory containing prediction reports")
    parser.add_argument("--figures-dir", type=str, default=None, help="Output directory for quality figures")
    parser.add_argument("--quality-dir", type=str, default=None, help="Output directory for quality CSVs")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["eyepacs_val", "eyepacs_test", "aptos_deduped", "messidor2_test", "idrid_test"],
        help="Datasets to analyze",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Config overrides as key=value pairs",
    )
    return parser.parse_args(argv)


def compute_dataset_quality_df(
    df_preds: pd.DataFrame,
    dataset_name: str,
    cfg: Any,
) -> pd.DataFrame:
    """Compute per-image quality metrics for all images in a dataset."""
    records = []
    print(f"  Computing quality metrics for {len(df_preds):,} images in '{dataset_name}'...")

    for idx, row in df_preds.iterrows():
        img_id = str(row["image_id"]).strip()
        raw_src = str(row.get("source", dataset_name))
        c_src = resolve_cache_source(raw_src)
        c_path = cache_path(c_src, img_id, cfg)

        if not c_path.exists():
            continue

        try:
            with Image.open(c_path) as pil_img:
                img_rgb = np.array(pil_img.convert("RGB"))
            metrics = compute_image_quality_metrics(img_rgb)
            record = {
                "image_id": img_id,
                "dataset": dataset_name,
                "grade": int(row.get("grade", 0)),
                "calibrated_prob": float(row.get("calibrated_prob", 0.0)),
                **metrics,
            }
            records.append(record)
        except Exception:
            continue

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def plot_quality_distributions(
    quality_dfs: dict[str, pd.DataFrame],
    output_path: pathlib.Path,
) -> None:
    """Plot cross-dataset distributions for key quality metrics quantifying camera/spectral shifts."""
    metrics_to_plot = [
        ("sharpness", "Sharpness (Var of Laplacian)", True),
        ("contrast", "Contrast (Luminance Std)", False),
        ("mean_brightness", "Mean Brightness", False),
        ("retina_fill_fraction", "Retina Fill Fraction", False),
        ("rg_ratio", "Colour Cast (R/G Ratio)", False),
        ("mean_b", "Blue Channel Intensity", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=300)
    axes = axes.flatten()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (metric_key, metric_label, log_scale) in enumerate(metrics_to_plot):
        ax = axes[i]
        data_series = []
        labels = []

        for dname, df in quality_dfs.items():
            if df.empty or metric_key not in df.columns:
                continue
            vals = df[metric_key].values
            vals = vals[~np.isnan(vals)]
            data_series.append(vals)
            labels.append(dname)

        if data_series:
            box = ax.boxplot(
                data_series,
                labels=labels,
                patch_artist=True,
                showmeans=True,
                meanline=True,
                flierprops={"marker": ".", "markersize": 2, "alpha": 0.25},
            )
            for patch, col in zip(box["boxes"], colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.6)

            if log_scale:
                ax.set_yscale("log")
            ax.set_title(metric_label, fontweight="bold", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.tick_params(axis="x", rotation=25, labelsize=9)

    fig.suptitle("Fundus Image Quality & Camera Shift Distributions Across Datasets", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".tmp_{os.getpid()}_{output_path.name}"
    fig.savefig(tmp_path, dpi=300)
    plt.close(fig)
    os.replace(tmp_path, output_path)


def plot_coverage_vs_performance(
    coverage_results: dict[str, dict[str, Any]],
    output_path: pathlib.Path,
) -> None:
    """Plot diagnostic AUC and Sensitivity vs. Quality Gate acceptance coverage."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    colors = {
        "eyepacs_val": "#1f77b4",
        "eyepacs_test": "#ff7f0e",
        "aptos_deduped": "#2ca02c",
        "messidor2_test": "#d62728",
        "idrid_test": "#9467bd",
    }

    ax_auc = axes[0]
    ax_sens = axes[1]

    for dname, cov_data in coverage_results.items():
        color = colors.get(dname, "#333333")
        tiers = cov_data.get("tiers", {})
        all_img = cov_data.get("all_images", {})

        coverages = [100.0]
        aucs = [all_img["auc"]]
        sensitivities = [all_img["sensitivity"]]

        # Sort tiers by descending coverage (100 -> 95 -> 90 -> 85)
        sorted_tiers = sorted(tiers.items(), key=lambda x: x[1]["coverage_pct"], reverse=True)
        for _, tdata in sorted_tiers:
            coverages.append(tdata["coverage_pct"])
            aucs.append(tdata["auc"])
            sensitivities.append(tdata["sensitivity"])

        ax_auc.plot(coverages, aucs, "o-", label=dname, color=color, lw=2)
        ax_sens.plot(coverages, sensitivities, "s-", label=dname, color=color, lw=2)

    ax_auc.set_title("AUC vs. Quality Gate Coverage", fontweight="bold", fontsize=11)
    ax_auc.set_xlabel("Coverage (% Images Accepted)", fontsize=10)
    ax_auc.set_ylabel("AUC", fontsize=10)
    ax_auc.grid(True, linestyle=":", alpha=0.6)
    ax_auc.legend(fontsize=8, loc="lower left")

    ax_sens.set_title("Sensitivity @ frozen threshold vs. Coverage", fontweight="bold", fontsize=11)
    ax_sens.set_xlabel("Coverage (% Images Accepted)", fontsize=10)
    ax_sens.set_ylabel("Sensitivity @ frozen threshold", fontsize=10)
    ax_sens.grid(True, linestyle=":", alpha=0.6)
    ax_sens.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".tmp_{os.getpid()}_{output_path.name}"
    fig.savefig(tmp_path, dpi=300)
    plt.close(fig)
    os.replace(tmp_path, output_path)


def compute_quartile_performance(
    df: pd.DataFrame,
    operating_threshold: float,
    metric_col: str = "quality_score",
) -> list[dict[str, Any]]:
    """Compute diagnostic metrics for each quartile of a given quality score."""
    if len(df) < 20:
        return []

    y_true = (df["grade"].values >= 2).astype(int)
    probs = df["calibrated_prob"].values
    scores = df[metric_col].values

    q25, q50, q75 = np.percentile(scores, [25, 50, 75])

    quartile_masks = [
        ("Q1 (Lowest Quality)", scores <= q25),
        ("Q2 (Moderate-Low)", (scores > q25) & (scores <= q50)),
        ("Q3 (Moderate-High)", (scores > q50) & (scores <= q75)),
        ("Q4 (Highest Quality)", scores > q75),
    ]

    quartile_records = []
    for label, mask in quartile_masks:
        n = int(np.sum(mask))
        if n == 0:
            continue
        y_q = y_true[mask]
        p_q = probs[mask]
        prev = float(np.mean(y_q))

        if len(np.unique(y_q)) >= 2:
            auc = float(compute_auc(y_q, p_q))
        else:
            auc = float("nan")

        preds_q = (p_q >= operating_threshold).astype(int)
        tp = int(np.sum((y_q == 1) & (preds_q == 1)))
        fp = int(np.sum((y_q == 0) & (preds_q == 1)))
        tn = int(np.sum((y_q == 0) & (preds_q == 0)))
        fn = int(np.sum((y_q == 1) & (preds_q == 0)))

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        quartile_records.append({
            "quartile": label,
            "n": n,
            "prevalence": prev,
            "auc": auc,
            "sensitivity": sens,
            "specificity": spec,
        })

    return quartile_records


def compute_enrichment_analysis(
    df: pd.DataFrame,
    operating_threshold: float,
    lq_flag_col: str = "is_reject_10%",
) -> dict[str, Any]:
    """
    Compute how enriched low-quality images are among False Negatives and False Positives.
    """
    y_true = (df["grade"].values >= 2).astype(int)
    probs = df["calibrated_prob"].values
    preds = (probs >= operating_threshold).astype(int)

    is_tp = (y_true == 1) & (preds == 1)
    is_fn = (y_true == 1) & (preds == 0)
    is_tn = (y_true == 0) & (preds == 0)
    is_fp = (y_true == 0) & (preds == 1)

    is_lq = df[lq_flag_col].values

    def calc_group_stats(mask: np.ndarray, group_name: str) -> dict[str, Any]:
        count = int(np.sum(mask))
        if count == 0:
            return {"name": group_name, "count": 0, "lq_count": 0, "lq_rate": 0.0}
        lq_count = int(np.sum(mask & is_lq))
        return {
            "name": group_name,
            "count": count,
            "lq_count": lq_count,
            "lq_rate": float(lq_count / count),
        }

    overall_lq_rate = float(np.mean(is_lq)) if len(is_lq) > 0 else 0.0
    tp_stats = calc_group_stats(is_tp, "True Positives")
    fn_stats = calc_group_stats(is_fn, "False Negatives")
    tn_stats = calc_group_stats(is_tn, "True Negatives")
    fp_stats = calc_group_stats(is_fp, "False Positives")

    fn_enrichment = (fn_stats["lq_rate"] / (overall_lq_rate + 1e-6)) if overall_lq_rate > 0 else 1.0
    fp_enrichment = (fp_stats["lq_rate"] / (overall_lq_rate + 1e-6)) if overall_lq_rate > 0 else 1.0

    return {
        "overall_lq_rate": overall_lq_rate,
        "tp": tp_stats,
        "fn": fn_stats,
        "tn": tn_stats,
        "fp": fp_stats,
        "fn_enrichment_ratio": float(fn_enrichment),
        "fp_enrichment_ratio": float(fp_enrichment),
    }


def _format_tier_label(tier_key: str) -> str:
    """Format gate tier key (e.g. 'reject_5%') as a descriptive label."""
    pct = tier_key.replace("reject_", "")
    return f"gate fitted for {pct} rejection on val"


def generate_quality_markdown_report(
    run_name: str,
    operating_threshold: float,
    gate: QualityGate,
    quartile_results: dict[str, list[dict[str, Any]]],
    enrichment_results: dict[str, dict[str, Any]],
    coverage_results: dict[str, dict[str, Any]],
) -> str:
    """Generate comprehensive Markdown report for Phase 6 quality analysis."""
    lines = [
        f"# Basira AI — Phase 6: Image Quality & Quality Gate Analysis",
        f"**Run:** `{run_name}`  |  **Operating Threshold:** `{operating_threshold:.4f}` (fit on EyePACS VAL only)",
        "",
        "> [!IMPORTANT]",
        "> **Validation-Only Gate Policy:** All Quality Gate thresholds and reference statistics "
        "were determined exclusively on `eyepacs_val`. External test sets were strictly evaluated "
        "using these frozen thresholds without fine-tuning.",
        "",
        "---",
        "",
        "## 1. Quality Gate Coverage vs. Diagnostic Performance",
        "",
        "| Dataset | Rejection Tier | Coverage (%) | Rejection Rate (%) | Evaluated N | AUC | Sensitivity @ frozen threshold | Specificity | Errors Filtered (FN / FP) |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for dname, cov_data in coverage_results.items():
        base = cov_data["all_images"]
        lines.append(
            f"| **`{dname}`** | All Images (100% Base) | `100.0%` | `0.0%` | {base['n_evaluated']:,} | "
            f"`{base['auc']:.4f}` | `{base['sensitivity']:.4f}` | `{base['specificity']:.4f}` | Baseline (0 / 0) |"
        )
        for tier_key, tdata in cov_data["tiers"].items():
            tier_label = _format_tier_label(tier_key)
            lines.append(
                f"| `{dname}` | {tier_label} | `{tdata['coverage_pct']:.1f}%` | `{tdata['rejection_rate_pct']:.1f}%` | {tdata['n_evaluated']:,} | "
                f"`{tdata['auc']:.4f}` | `{tdata['sensitivity']:.4f}` | `{tdata['specificity']:.4f}` | {tdata['fn_filtered_out']} FN / {tdata['fp_filtered_out']} FP |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Diagnostic Performance by Quality Quartiles",
        "",
        "> [!NOTE]",
        "> **Within-Dataset Quartiles & Severity Confounding:** Quality quartiles are computed within each dataset independently. "
        "Prevalence differs strongly across quartiles (see the Prevalence (%) column), so quartile AUCs are confounded by severity mix.",
        "",
        "| Dataset | Quality Quartile | N | Prevalence (%) | AUC | Sensitivity @ frozen threshold | Specificity |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|",
    ])

    for dname, qlist in quartile_results.items():
        for q in qlist:
            lines.append(
                f"| `{dname}` | {q['quartile']} | {q['n']:,} | `{q['prevalence'] * 100:.1f}%` | "
                f"`{q['auc']:.4f}` | `{q['sensitivity']:.4f}` | `{q['specificity']:.4f}` |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Low-Quality Enrichment Among Diagnostic Errors",
        "",
        "> [!NOTE]",
        "> **Enrichment Ratio:** Ratio of low-quality rate in error subset (FN or FP) vs. overall dataset baseline.",
        "",
        "| Dataset | Baseline LQ Rate (%) | n_FN | LQ Rate in FN (%) | FN Enrichment Ratio | n_FP | LQ Rate in FP (%) | FP Enrichment Ratio |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    small_count_warnings: list[str] = []
    for dname, enr in enrichment_results.items():
        base_rate = enr["overall_lq_rate"] * 100
        n_fn = int(enr["fn"]["count"])
        n_fp = int(enr["fp"]["count"])
        fn_rate = enr["fn"]["lq_rate"] * 100
        fp_rate = enr["fp"]["lq_rate"] * 100
        fn_enr = enr["fn_enrichment_ratio"]
        fp_enr = enr["fp_enrichment_ratio"]
        lines.append(
            f"| `{dname}` | `{base_rate:.1f}%` | {n_fn:,} | `{fn_rate:.1f}%` | **`{fn_enr:.2f}x`** | {n_fp:,} | `{fp_rate:.1f}%` | **`{fp_enr:.2f}x`** |"
        )
        low_groups = []
        if n_fn < 30:
            low_groups.append(f"n_FN={n_fn}")
        if n_fp < 30:
            low_groups.append(f"n_FP={n_fp}")
        if low_groups:
            small_count_warnings.append(f"`{dname}` ({', '.join(low_groups)})")

    if small_count_warnings:
        lines.extend([
            "",
            "> [!WARNING]",
            f"> **Small Error Counts (n < 30):** {', '.join(small_count_warnings)}. "
            "Enrichment ratios computed from fewer than 30 errors have high sampling variance and should be interpreted with caution.",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 4. Data-Driven Gate Summary",
        "- **Quality Metrics:** Evaluated sharpness (green-channel Laplacian variance on interior retina), contrast (luminance standard deviation), brightness, retina fill fraction, and R/G color ratio.",
    ])

    for dname, cov_data in coverage_results.items():
        base = cov_data["all_images"]
        base_auc = base["auc"]
        base_sens = base["sensitivity"]
        tier_summaries = []
        all_neutral = True
        for tier_key, tdata in cov_data["tiers"].items():
            d_auc = tdata["auc"] - base_auc
            d_sens = tdata["sensitivity"] - base_sens
            tier_label = _format_tier_label(tier_key)
            if np.isnan(d_auc):
                effect = "AUC undefined"
            elif d_auc > 0.01:
                effect = "improves AUC"
                all_neutral = False
            elif d_auc < -0.01:
                effect = "reduces AUC"
                all_neutral = False
            else:
                effect = "neutral on AUC"
            tier_summaries.append(
                f"{tier_label} (rej {tdata['rejection_rate_pct']:.1f}%): ΔAUC={d_auc:+.4f}, ΔSens={d_sens:+.4f} [{effect}]"
            )

        if all_neutral:
            overall_note = "Gate is neutral on AUC (|ΔAUC| <= 0.01 across all tiers)."
        else:
            overall_note = "Gate produces |ΔAUC| > 0.01 on at least one tier."
        lines.append(f"- **`{dname}`:** {overall_note} " + "; ".join(tier_summaries) + ".")

    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_config()
    if args.set:
        apply_overrides(cfg, args.set)

    run_name = args.run_name.strip()
    report_dir = pathlib.Path(args.report_dir) if args.report_dir else pathlib.Path(cfg.output.reports) / run_name
    figures_dir = pathlib.Path(args.figures_dir) if args.figures_dir else report_dir / "figures"
    quality_dir = pathlib.Path(args.quality_dir) if args.quality_dir else report_dir / "quality"
    figures_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)

    # 1. Operating threshold from calibration.json
    calib_file = report_dir / "calibration.json"
    if calib_file.exists():
        with open(calib_file, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
        threshold = float(calib_data.get("operating_threshold", 0.2785))
    else:
        threshold = 0.2785

    print("=" * 75)
    print(f"Basira AI — Phase 6: Image Quality & Quality Gate Pipeline")
    print(f"Run Name    : {run_name}")
    print(f"Report Dir  : {report_dir}")
    print(f"Quality Dir : {quality_dir}")
    print(f"Threshold   : {threshold:.4f} (Validation frozen)")
    print("=" * 75)

    # 2. Compute quality metrics per dataset
    quality_dfs: dict[str, pd.DataFrame] = {}

    for dname in args.datasets:
        pred_csv = report_dir / f"{dname}_predictions.csv"
        if not pred_csv.exists():
            print(f"Predictions CSV '{pred_csv.name}' not found. Skipping '{dname}'.")
            continue

        q_csv = quality_dir / f"{dname}_quality.csv"
        if q_csv.exists():
            print(f"Loading existing quality CSV: '{q_csv.name}'...")
            df_q = pd.read_csv(q_csv)
        else:
            df_preds = pd.read_csv(pred_csv)
            df_q = compute_dataset_quality_df(df_preds, dname, cfg)
            if not df_q.empty:
                tmp_q = quality_dir / f".tmp_{os.getpid()}_{q_csv.name}"
                df_q.to_csv(tmp_q, index=False)
                os.replace(tmp_q, q_csv)
                print(f"  ✓ Saved quality metrics to: {q_csv}")

        if not df_q.empty:
            quality_dfs[dname] = df_q

    if not quality_dfs:
        raise RuntimeError("No quality metrics could be computed. Please check prediction CSVs and cache.")

    # 3. Fit Quality Gate on eyepacs_val only
    gate_fit_dataset = "eyepacs_val" if "eyepacs_val" in quality_dfs else list(quality_dfs.keys())[0]
    print(f"\nFitting Quality Gate on '{gate_fit_dataset}' only...")
    val_q_df = quality_dfs[gate_fit_dataset]
    gate = QualityGate(rejection_rates=(0.05, 0.10, 0.15))
    gate.fit(val_q_df)

    # Apply Quality Gate scoring and flags across all datasets
    for dname, df_q in quality_dfs.items():
        decisions = gate.predict(df_q)
        for col_name, arr in decisions.items():
            df_q[col_name] = arr

        # Save updated CSV with quality score and rejection flags
        q_csv = quality_dir / f"{dname}_quality.csv"
        tmp_q = quality_dir / f".tmp_{os.getpid()}_{q_csv.name}"
        df_q.to_csv(tmp_q, index=False)
        os.replace(tmp_q, q_csv)

    # Save Gate configuration JSON
    gate_json_path = quality_dir / "quality_gate.json"
    tmp_gate = quality_dir / f".tmp_{os.getpid()}_{gate_json_path.name}"
    with open(tmp_gate, "w", encoding="utf-8") as f:
        json.dump(gate.to_dict(), f, indent=2)
    os.replace(tmp_gate, gate_json_path)
    print(f"  ✓ Saved Quality Gate parameters to: {gate_json_path}")

    # 4. Generate Quality Distribution Figure
    print("\nGenerating cross-dataset quality distribution plots...")
    dist_fig_path = figures_dir / "quality_distributions.png"
    plot_quality_distributions(quality_dfs, dist_fig_path)
    print(f"  ✓ Saved distribution figure: {dist_fig_path}")

    # 5. Quartile and Enrichment Analysis
    print("\nComputing quality quartile and error enrichment analyses...")
    quartile_results: dict[str, list[dict[str, Any]]] = {}
    enrichment_results: dict[str, dict[str, Any]] = {}
    coverage_results: dict[str, dict[str, Any]] = {}

    for dname, df_q in quality_dfs.items():
        q_records = compute_quartile_performance(df_q, operating_threshold=threshold)
        quartile_results[dname] = q_records

        enr_records = compute_enrichment_analysis(df_q, operating_threshold=threshold)
        enrichment_results[dname] = enr_records

        cov_data = gate.evaluate_coverage(df_q, operating_threshold=threshold)
        coverage_results[dname] = cov_data

    # 6. Generate Coverage vs Performance Figure
    print("Generating coverage vs. performance figure...")
    cov_fig_path = figures_dir / "quality_gate_coverage.png"
    plot_coverage_vs_performance(coverage_results, cov_fig_path)
    print(f"  ✓ Saved coverage figure: {cov_fig_path}")

    # 7. Save JSON & Markdown Reports
    payload = {
        "run_name": run_name,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "operating_threshold": threshold,
        "gate": gate.to_dict(),
        "coverage_analysis": coverage_results,
        "quartile_analysis": quartile_results,
        "enrichment_analysis": enrichment_results,
    }

    results_json_path = report_dir / "quality_results.json"
    tmp_res = report_dir / f".tmp_{os.getpid()}_{results_json_path.name}"
    with open(tmp_res, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_res, results_json_path)
    print(f"  ✓ Saved quality analysis JSON: {results_json_path}")

    md_report = generate_quality_markdown_report(
        run_name=run_name,
        operating_threshold=threshold,
        gate=gate,
        quartile_results=quartile_results,
        enrichment_results=enrichment_results,
        coverage_results=coverage_results,
    )
    md_report_path = report_dir / "quality_analysis.md"
    tmp_md = report_dir / f".tmp_{os.getpid()}_{md_report_path.name}"
    with open(tmp_md, "w", encoding="utf-8") as f:
        f.write(md_report)
    os.replace(tmp_md, md_report_path)
    print(f"  ✓ Saved quality analysis Markdown report: {md_report_path}")

    # Also save dedicated quality gate coverage table markdown
    gate_table_path = report_dir / "quality_gate_table.md"
    tmp_gt = report_dir / f".tmp_{os.getpid()}_{gate_table_path.name}"
    with open(tmp_gt, "w", encoding="utf-8") as f:
        f.write(md_report.split("## 2. Diagnostic Performance by Quality Quartiles")[0])
    os.replace(tmp_gt, gate_table_path)
    print(f"  ✓ Saved Quality Gate coverage table: {gate_table_path}")

    print("\n" + "=" * 75)
    print(f"Quality and Quality Gate analysis complete for '{run_name}'.")
    print("=" * 75)


if __name__ == "__main__":
    main()
