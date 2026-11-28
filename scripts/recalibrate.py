"""
scripts/recalibrate.py
======================
Local recalibration experiment for domain-adapted Diabetic Retinopathy screening.

Key features:
  1. APTOS 2019 Local Recalibration:
     - For sample sizes n in {25, 50, 100, 200, 500, 'all'} drawn from aptos_train.csv
       (stratified by referable, 20 seeded repetitions):
       * Refits Platt scaling (slope + bias) on the local sample logits.
       * Selects operating threshold achieving >= 90% specificity strictly on that local sample.
       * Evaluates Sensitivity, Specificity, and AUC on the independent aptos_test.csv.
     - Benchmarked against:
       * "no recalibration (frozen)": EyePACS val scaler and frozen threshold (0.2785).
       * "oracle": Scaler and threshold fit directly on aptos_test.csv.
  2. IDRiD Local Recalibration:
     - 20 repeated stratified 50/50 splits of idrid_test.csv (noting the small N of 516 images).
     - Fits Platt + 90% threshold on calibration split; evaluates on test split.
  3. Messidor-2 Policy:
     - Skipped due to paired eye correlations without patient identifiers (preventing data leakage).
  4. Exports:
     - recalibration_results.json
     - recalibration_results.md
     - figures/recalibration_curve.png (Sensitivity and Specificity vs n with ±1 SD bands).
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
from sklearn.model_selection import StratifiedShuffleSplit
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure repo root and src/ are on sys.path
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from basira.config import apply_overrides, load_config
from basira.data.dataset import CachedFundusDataset
from basira.data.loaders import read_split_csv
from basira.eval.calibration import PlattScaler
from basira.eval.inference import find_split_file, load_eval_checkpoint, run_inference
from basira.eval.metrics import compute_auc, sensitivity_at_specificity


SAMPLE_SIZES = [25, 50, 100, 200, 500, "all"]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local recalibration experiments for Basira AI.")
    parser.add_argument("--run-name", type=str, required=True, help="Run name (e.g. b0_referable_s42)")
    parser.add_argument("--ckpt", type=str, default="best.pt", help="Checkpoint filename (default: best.pt)")
    parser.add_argument("--report-dir", type=str, default=None, help="Directory for reports")
    parser.add_argument("--figures-dir", type=str, default=None, help="Directory for output figures")
    parser.add_argument("--n-repeats", type=int, default=20, help="Number of repetitions per sample size (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Target device ('cuda' or 'cpu')")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Config overrides as key=value pairs",
    )
    return parser.parse_args(argv)


def get_dataset_logits_and_labels(
    df: pd.DataFrame,
    dataset_name: str,
    model: nn.Module,
    device: torch.device,
    cfg: Any,
    report_dir: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve model logits and ground-truth binary labels for a dataset split.
    Uses cached prediction CSV if available; otherwise computes forward pass.
    """
    pred_csv = report_dir / f"{dataset_name}_predictions.csv"
    if pred_csv.exists():
        df_p = pd.read_csv(pred_csv)
        # Verify alignment by image_id
        if "image_id" in df.columns and "image_id" in df_p.columns:
            merged = pd.merge(df[["image_id"]], df_p[["image_id", "logit", "grade"]], on="image_id", how="left")
            if not merged["logit"].isna().any():
                logits = merged["logit"].values.astype(np.float64)
                labels = (merged["grade"].values >= 2).astype(np.int64)
                return logits, labels

    # Compute logits using model forward pass
    ds = CachedFundusDataset(df, cfg=cfg, train=False, task="referable", check_files=True)
    loader = DataLoader(ds, batch_size=int(cfg.training.batch_size), shuffle=False, num_workers=0)
    logits, _ = run_inference(model, loader, device)
    labels = (df["grade"].values >= 2).astype(np.int64)
    return logits, labels


def evaluate_threshold_performance(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calculate AUC, Sensitivity, and Specificity at a given operating threshold."""
    auc = compute_auc(y_true, probs)
    preds = (probs >= threshold).astype(int)

    tp = float(np.sum((y_true == 1) & (preds == 1)))
    fp = float(np.sum((y_true == 0) & (preds == 1)))
    tn = float(np.sum((y_true == 0) & (preds == 0)))
    fn = float(np.sum((y_true == 1) & (preds == 0)))

    sens = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "auc": float(auc),
        "sensitivity": float(sens),
        "specificity": float(spec),
    }


def run_single_calibration_trial(
    cal_logits: np.ndarray,
    cal_labels: np.ndarray,
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    target_spec: float = 0.90,
) -> dict[str, float]:
    """
    Fit Platt scaling on cal_logits/cal_labels, pick threshold on calibration set only,
    and evaluate on test_logits/test_labels.
    """
    scaler = PlattScaler()
    scaler.fit(cal_logits, cal_labels)

    # Pick threshold achieving target specificity on calibration set ONLY
    cal_probs = scaler.predict_proba(cal_logits)
    _, threshold = sensitivity_at_specificity(cal_labels, cal_probs, target_specificity=target_spec)

    # Evaluate on test set with frozen scaler and threshold
    test_probs = scaler.predict_proba(test_logits)
    perf = evaluate_threshold_performance(test_labels, test_probs, threshold=threshold)
    perf["operating_threshold"] = float(threshold)
    perf["slope"] = float(scaler.slope)
    perf["intercept"] = float(scaler.intercept)
    return perf


def run_aptos_recalibration(
    train_logits: np.ndarray,
    train_labels: np.ndarray,
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    frozen_scaler: PlattScaler,
    frozen_threshold: float,
    n_repeats: int = 20,
    seed: int = 42,
    target_spec: float = 0.90,
) -> dict[str, Any]:
    """Execute APTOS local recalibration experiment across sample sizes."""
    total_train = len(train_labels)

    # 1. Baseline: Frozen EyePACS model
    frozen_test_probs = frozen_scaler.predict_proba(test_logits)
    frozen_perf = evaluate_threshold_performance(test_labels, frozen_test_probs, threshold=frozen_threshold)

    # 2. Baseline: Oracle (fit and tuned on test set directly)
    oracle_scaler = PlattScaler().fit(test_logits, test_labels)
    oracle_test_probs = oracle_scaler.predict_proba(test_logits)
    _, oracle_threshold = sensitivity_at_specificity(test_labels, oracle_test_probs, target_specificity=target_spec)
    oracle_perf = evaluate_threshold_performance(test_labels, oracle_test_probs, threshold=oracle_threshold)
    oracle_perf["operating_threshold"] = float(oracle_threshold)

    results_by_n: dict[str, Any] = {}

    for n_val in SAMPLE_SIZES:
        n_key = str(n_val)
        repeats_data = []

        if n_val == "all":
            # Deterministic: full training set
            trial = run_single_calibration_trial(
                cal_logits=train_logits,
                cal_labels=train_labels,
                test_logits=test_logits,
                test_labels=test_labels,
                target_spec=target_spec,
            )
            trial["repeat"] = 0
            repeats_data.append(trial)
            summary = {
                "n": total_train,
                "n_label": "all",
                "repeats_count": 1,
                "sensitivity_mean": trial["sensitivity"],
                "sensitivity_sd": 0.0,
                "specificity_mean": trial["specificity"],
                "specificity_sd": 0.0,
                "auc_mean": trial["auc"],
                "auc_sd": 0.0,
                "threshold_mean": trial["operating_threshold"],
                "threshold_sd": 0.0,
            }
        else:
            n_int = int(n_val)
            # If n is greater than or equal to total available samples, skip it
            if n_int >= total_train:
                continue

            for r in range(n_repeats):
                r_seed = seed + r * 1000 + n_int
                # Stratified sample of size n_int from train
                sss = StratifiedShuffleSplit(n_splits=1, train_size=n_int, random_state=r_seed)
                cal_idx, _ = next(sss.split(train_logits, train_labels))

                sub_logits = train_logits[cal_idx]
                sub_labels = train_labels[cal_idx]

                # Ensure both classes present
                if len(np.unique(sub_labels)) < 2:
                    continue

                trial = run_single_calibration_trial(
                    cal_logits=sub_logits,
                    cal_labels=sub_labels,
                    test_logits=test_logits,
                    test_labels=test_labels,
                    target_spec=target_spec,
                )
                trial["repeat"] = r
                repeats_data.append(trial)

            sens_vals = [t["sensitivity"] for t in repeats_data]
            spec_vals = [t["specificity"] for t in repeats_data]
            auc_vals = [t["auc"] for t in repeats_data]
            thresh_vals = [t["operating_threshold"] for t in repeats_data]

            summary = {
                "n": n_int,
                "n_label": str(n_int),
                "repeats_count": len(repeats_data),
                "sensitivity_mean": float(np.mean(sens_vals)),
                "sensitivity_sd": float(np.std(sens_vals)),
                "specificity_mean": float(np.mean(spec_vals)),
                "specificity_sd": float(np.std(spec_vals)),
                "auc_mean": float(np.mean(auc_vals)),
                "auc_sd": float(np.std(auc_vals)),
                "threshold_mean": float(np.mean(thresh_vals)),
                "threshold_sd": float(np.std(thresh_vals)),
            }

        results_by_n[n_key] = {
            "summary": summary,
            "trials": repeats_data,
        }

    return {
        "dataset": "APTOS 2019",
        "n_train": len(train_labels),
        "n_test": len(test_labels),
        "frozen_baseline": frozen_perf,
        "oracle_upper_bound": oracle_perf,
        "sample_size_results": results_by_n,
    }


def run_idrid_recalibration(
    idrid_logits: np.ndarray,
    idrid_labels: np.ndarray,
    frozen_scaler: PlattScaler,
    frozen_threshold: float,
    n_repeats: int = 20,
    seed: int = 42,
    target_spec: float = 0.90,
) -> dict[str, Any]:
    """Execute IDRiD local recalibration using repeated 50/50 stratified splits."""
    # 1. Baseline: Frozen EyePACS model
    frozen_probs = frozen_scaler.predict_proba(idrid_logits)
    frozen_perf = evaluate_threshold_performance(idrid_labels, frozen_probs, threshold=frozen_threshold)

    # 2. Baseline: Oracle (fit and tuned on entire IDRiD set)
    oracle_scaler = PlattScaler().fit(idrid_logits, idrid_labels)
    oracle_probs = oracle_scaler.predict_proba(idrid_logits)
    _, oracle_threshold = sensitivity_at_specificity(idrid_labels, oracle_probs, target_specificity=target_spec)
    oracle_perf = evaluate_threshold_performance(idrid_labels, oracle_probs, threshold=oracle_threshold)
    oracle_perf["operating_threshold"] = float(oracle_threshold)

    # 3. 20 repeated 50/50 splits
    repeats_data = []
    sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=0.50, random_state=seed)

    for r, (cal_idx, test_idx) in enumerate(sss.split(idrid_logits, idrid_labels)):
        trial = run_single_calibration_trial(
            cal_logits=idrid_logits[cal_idx],
            cal_labels=idrid_labels[cal_idx],
            test_logits=idrid_logits[test_idx],
            test_labels=idrid_labels[test_idx],
            target_spec=target_spec,
        )
        trial["repeat"] = r
        trial["cal_size"] = len(cal_idx)
        trial["test_size"] = len(test_idx)
        repeats_data.append(trial)

    sens_vals = [t["sensitivity"] for t in repeats_data]
    spec_vals = [t["specificity"] for t in repeats_data]
    auc_vals = [t["auc"] for t in repeats_data]
    thresh_vals = [t["operating_threshold"] for t in repeats_data]

    summary = {
        "n_calibration": int(len(idrid_labels) // 2),
        "n_evaluation": int(len(idrid_labels) - len(idrid_labels) // 2),
        "repeats_count": len(repeats_data),
        "sensitivity_mean": float(np.mean(sens_vals)),
        "sensitivity_sd": float(np.std(sens_vals)),
        "specificity_mean": float(np.mean(spec_vals)),
        "specificity_sd": float(np.std(spec_vals)),
        "auc_mean": float(np.mean(auc_vals)),
        "auc_sd": float(np.std(auc_vals)),
        "threshold_mean": float(np.mean(thresh_vals)),
        "threshold_sd": float(np.std(thresh_vals)),
    }

    return {
        "dataset": "IDRiD",
        "n_total": len(idrid_labels),
        "note": f"Small sample size (total N={len(idrid_labels)}; {summary['n_calibration']} calibration / {summary['n_evaluation']} evaluation). Higher variance is expected.",
        "frozen_baseline": frozen_perf,
        "oracle_upper_bound": oracle_perf,
        "50_50_split_summary": summary,
        "trials": repeats_data,
    }


def generate_recalibration_markdown(
    run_name: str,
    aptos_res: dict[str, Any],
    idrid_res: dict[str, Any],
) -> str:
    """Generate Markdown report summarizing local recalibration results."""
    n_aptos_test = aptos_res.get("n_test", 524)
    n_idrid_total = idrid_res.get("n_total", 516)
    idrid_summary = idrid_res.get("50_50_split_summary", {})
    n_idrid_cal = idrid_summary.get("n_calibration", 258)
    n_idrid_eval = idrid_summary.get("n_evaluation", 258)

    lines = [
        f"# Basira AI — Local Recalibration Report: `{run_name}`",
        "",
        "This experiment evaluates domain adaptation via sample-efficient local probability recalibration.",
        "A clinic deploying the model without local retraining can calibrate probabilities using a small labelled",
        "reference sample, adapting the decision boundary to the local patient population.",
        "",
        "---",
        "",
        "## 1. APTOS 2019 Local Recalibration",
        "",
        "Platt scaling parameters and operating threshold (target specificity ≥ 0.90) were refit on $n$ labelled images",
        f"drawn from `aptos_train.csv` (20 seeded stratified repetitions) and evaluated on `aptos_test.csv` ({n_aptos_test:,} images).",
        "",
        "| Sample Size $n$ | Sensitivity (Mean ± SD) | Specificity (Mean ± SD) | AUC (Mean ± SD) | Selected Threshold |",
        "|:---|:---:|:---:|:---:|:---:|",
    ]

    # Frozen baseline row
    fb = aptos_res["frozen_baseline"]
    lines.append(
        f"| **Frozen (No Recalibration)** | `{fb['sensitivity']:.4f}` | `{fb['specificity']:.4f}` | `{fb['auc']:.4f}` | Baseline (0.2785) |"
    )

    # Sample size rows
    for n_key in SAMPLE_SIZES:
        if str(n_key) not in aptos_res.get("sample_size_results", {}):
            continue
        s = aptos_res["sample_size_results"][str(n_key)]["summary"]
        n_lbl = s["n_label"]
        sens = f"{s['sensitivity_mean']:.4f} ± {s['sensitivity_sd']:.4f}"
        spec = f"{s['specificity_mean']:.4f} ± {s['specificity_sd']:.4f}"
        auc = f"{s['auc_mean']:.4f} ± {s['auc_sd']:.4f}"
        thresh = f"{s['threshold_mean']:.4f} ± {s['threshold_sd']:.4f}"
        lines.append(f"| $n = {n_lbl}$ | {sens} | {spec} | {auc} | {thresh} |")

    # Oracle row
    ob = aptos_res["oracle_upper_bound"]
    lines.append(
        f"| **Oracle (Tuned on Test Set)** | `{ob['sensitivity']:.4f}` | `{ob['specificity']:.4f}` | `{ob['auc']:.4f}` | `{ob['operating_threshold']:.4f}` |"
    )
    lines.append("")

    lines.extend([
        "---",
        "",
        "## 2. IDRiD Local Recalibration (Repeated 50/50 Splits)",
        "",
        "> [!NOTE]",
        f"> **Small Sample Limitation:** IDRiD comprises {n_idrid_total:,} total images ({n_idrid_cal:,} calibration / {n_idrid_eval:,} test per split). "
        "Higher variance across repetitions reflects sample size constraints in rare clinical disease categories.",
        "",
        "| Configuration | Sensitivity (Mean ± SD) | Specificity (Mean ± SD) | AUC (Mean ± SD) | Selected Threshold |",
        "|:---|:---:|:---:|:---:|:---:|",
    ])

    ifb = idrid_res["frozen_baseline"]
    lines.append(
        f"| **Frozen (No Recalibration)** | `{ifb['sensitivity']:.4f}` | `{ifb['specificity']:.4f}` | `{ifb['auc']:.4f}` | Baseline (0.2785) |"
    )
    isum = idrid_res["50_50_split_summary"]
    isens = f"{isum['sensitivity_mean']:.4f} ± {isum['sensitivity_sd']:.4f}"
    ispec = f"{isum['specificity_mean']:.4f} ± {isum['specificity_sd']:.4f}"
    iauc = f"{isum['auc_mean']:.4f} ± {isum['auc_sd']:.4f}"
    ithresh = f"{isum['threshold_mean']:.4f} ± {isum['threshold_sd']:.4f}"
    lines.append(f"| **Local 50/50 Split ($N=258$)** | {isens} | {ispec} | {iauc} | {ithresh} |")

    iob = idrid_res["oracle_upper_bound"]
    lines.append(
        f"| **Oracle (Full IDRiD Upper Bound)** | `{iob['sensitivity']:.4f}` | `{iob['specificity']:.4f}` | `{iob['auc']:.4f}` | `{iob['operating_threshold']:.4f}` |"
    )
    lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. Messidor-2 Policy Statement",
        "",
        "> [!IMPORTANT]",
        "> **Messidor-2 Exclusion:** Messidor-2 was explicitly excluded from local recalibration splitting experiments. "
        "Messidor-2 consists of paired fundus photos (left and right eyes) from 874 patients without provided patient identifiers. "
        "Any random splitting would assign eyes from the same individual across both calibration and evaluation folds, "
        "causing patient-level data leakage and artificially deflating generalization error estimates. "
        "Messidor-2 is preserved strictly as an un-split external benchmark.",
    ])

    return "\n".join(lines) + "\n"


def plot_recalibration_curves(
    aptos_res: dict[str, Any],
    output_path: pathlib.Path,
) -> None:
    """Plot Sensitivity and Specificity vs sample size n with ±1 SD bands and baselines."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Extract numeric sample sizes (exclude 'all' from x-axis plotting, plot as vertical marker or max)
    sample_sizes = [
        n for n in [25, 50, 100, 200, 500]
        if str(n) in aptos_res.get("sample_size_results", {})
    ]
    if not sample_sizes:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        ax.text(0.5, 0.5, "No sub-sample sizes evaluated", ha="center", va="center")
        ax.axis("off")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.parent / f".tmp_{os.getpid()}_{output_path.name}"
        fig.savefig(tmp_path, dpi=300)
        plt.close(fig)
        os.replace(tmp_path, output_path)
        return

    sens_means = [aptos_res["sample_size_results"][str(n)]["summary"]["sensitivity_mean"] for n in sample_sizes]
    sens_sds = [aptos_res["sample_size_results"][str(n)]["summary"]["sensitivity_sd"] for n in sample_sizes]
    spec_means = [aptos_res["sample_size_results"][str(n)]["summary"]["specificity_mean"] for n in sample_sizes]
    spec_sds = [aptos_res["sample_size_results"][str(n)]["summary"]["specificity_sd"] for n in sample_sizes]

    frozen_sens = aptos_res["frozen_baseline"]["sensitivity"]
    frozen_spec = aptos_res["frozen_baseline"]["specificity"]
    oracle_sens = aptos_res["oracle_upper_bound"]["sensitivity"]
    oracle_spec = aptos_res["oracle_upper_bound"]["specificity"]

    # 1. Sensitivity vs n
    ax1 = axes[0]
    ax1.plot(sample_sizes, sens_means, "o-", color="#1f77b4", lw=2, label="Recalibrated (Mean)")
    ax1.fill_between(
        sample_sizes,
        np.array(sens_means) - np.array(sens_sds),
        np.array(sens_means) + np.array(sens_sds),
        color="#1f77b4",
        alpha=0.2,
        label="±1 SD (20 repeats)",
    )
    ax1.axhline(frozen_sens, color="#d62728", linestyle="--", lw=1.5, label=f"Frozen Baseline ({frozen_sens:.3f})")
    ax1.axhline(oracle_sens, color="#2ca02c", linestyle=":", lw=1.5, label=f"Oracle Upper Bound ({oracle_sens:.3f})")
    ax1.set_xscale("log")
    ax1.set_xticks(sample_sizes)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel("Local Calibration Sample Size $n$ (log scale)", fontsize=11)
    ax1.set_ylabel("Test Sensitivity @ 90% Specificity", fontsize=11)
    ax1.set_title("APTOS Test Sensitivity vs. Local Calibration Sample Size", fontweight="bold", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=9)

    # 2. Specificity vs n
    ax2 = axes[1]
    ax2.plot(sample_sizes, spec_means, "s-", color="#ff7f0e", lw=2, label="Recalibrated (Mean)")
    ax2.fill_between(
        sample_sizes,
        np.array(spec_means) - np.array(spec_sds),
        np.array(spec_means) + np.array(spec_sds),
        color="#ff7f0e",
        alpha=0.2,
        label="±1 SD (20 repeats)",
    )
    ax2.axhline(frozen_spec, color="#d62728", linestyle="--", lw=1.5, label=f"Frozen Baseline ({frozen_spec:.3f})")
    ax2.axhline(0.90, color="gray", linestyle="-.", lw=1.2, label="Target Specificity (0.90)")
    ax2.set_xscale("log")
    ax2.set_xticks(sample_sizes)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_xlabel("Local Calibration Sample Size $n$ (log scale)", fontsize=11)
    ax2.set_ylabel("Empirical Test Specificity", fontsize=11)
    ax2.set_title("APTOS Test Specificity vs. Local Calibration Sample Size", fontweight="bold", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".tmp_{os.getpid()}_{output_path.name}"
    fig.savefig(tmp_path, dpi=300)
    plt.close(fig)
    os.replace(tmp_path, output_path)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_config()
    if args.set:
        apply_overrides(cfg, args.set)

    run_name = args.run_name.strip()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 1. Resolve paths
    report_dir = pathlib.Path(args.report_dir) if args.report_dir else pathlib.Path(cfg.output.reports) / run_name
    figures_dir = pathlib.Path(args.figures_dir) if args.figures_dir else report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        pathlib.Path(cfg.output.checkpoints) / run_name / args.ckpt,
        _REPO_ROOT / "checkpoints" / run_name / args.ckpt,
    ]
    ckpt_path = None
    for c in candidates:
        if c.exists():
            ckpt_path = c
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"Checkpoint '{args.ckpt}' not found for run '{run_name}'.")

    # 2. Frozen calibration parameters
    calib_file = report_dir / "calibration.json"
    if calib_file.exists():
        with open(calib_file, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
        frozen_thresh = float(calib_data.get("operating_threshold", 0.2785))
        frozen_scaler = PlattScaler(
            slope=float(calib_data.get("slope", 1.0)),
            intercept=float(calib_data.get("intercept", 0.0)),
        )
        frozen_scaler.is_fitted = True
    else:
        frozen_thresh = 0.2785
        frozen_scaler = PlattScaler(slope=1.0, intercept=0.0)
        frozen_scaler.is_fitted = True

    print("=" * 75)
    print(f"Basira AI — Local Recalibration Experiments: {run_name}")
    print(f"Report Dir     : {report_dir}")
    print(f"Frozen Thresh  : {frozen_thresh:.4f}")
    print(f"Repeats        : {args.n_repeats}")
    print(f"Random Seed    : {args.seed}")
    print(f"Device         : {device}")
    print("=" * 75)

    # 3. Model setup
    ckpt = load_eval_checkpoint(ckpt_path, map_location="cpu")
    backbone = ckpt.get("config", {}).get("model", {}).get("backbone", cfg.model.backbone) if isinstance(ckpt, dict) else cfg.model.backbone
    timm_name = "mobilenetv3_large_100" if backbone in ("mobilenetv3", "mobilenetv3_large_100") else backbone
    model = timm.create_model(timm_name, pretrained=False, num_classes=1)
    state_dict = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ─────────────────────────────────────────────────────────────────────────
    # 4. APTOS Recalibration
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/3] Running APTOS local recalibration experiments...")
    df_apt_train = read_split_csv(find_split_file("aptos_train.csv", cfg))
    df_apt_test = read_split_csv(find_split_file("aptos_test.csv", cfg))

    apt_train_logits, apt_train_labels = get_dataset_logits_and_labels(
        df_apt_train, "aptos_train", model, device, cfg, report_dir
    )
    apt_test_logits, apt_test_labels = get_dataset_logits_and_labels(
        df_apt_test, "aptos_test", model, device, cfg, report_dir
    )

    aptos_results = run_aptos_recalibration(
        train_logits=apt_train_logits,
        train_labels=apt_train_labels,
        test_logits=apt_test_logits,
        test_labels=apt_test_labels,
        frozen_scaler=frozen_scaler,
        frozen_threshold=frozen_thresh,
        n_repeats=args.n_repeats,
        seed=args.seed,
    )

    print("  ✓ APTOS Results Summary:")
    print(f"    - Frozen Baseline   : Sens={aptos_results['frozen_baseline']['sensitivity']:.4f}, Spec={aptos_results['frozen_baseline']['specificity']:.4f}, AUC={aptos_results['frozen_baseline']['auc']:.4f}")
    for n_key in SAMPLE_SIZES:
        if str(n_key) in aptos_results.get("sample_size_results", {}):
            s = aptos_results["sample_size_results"][str(n_key)]["summary"]
            print(f"    - n={s['n_label']:<4}              : Sens={s['sensitivity_mean']:.4f}±{s['sensitivity_sd']:.4f}, Spec={s['specificity_mean']:.4f}±{s['specificity_sd']:.4f}, AUC={s['auc_mean']:.4f}")
    print(f"    - Oracle Upper Bound: Sens={aptos_results['oracle_upper_bound']['sensitivity']:.4f}, Spec={aptos_results['oracle_upper_bound']['specificity']:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. IDRiD Recalibration
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/3] Running IDRiD local recalibration experiments (20x 50/50 splits)...")
    df_idrid = read_split_csv(find_split_file("idrid_test.csv", cfg))
    idrid_logits, idrid_labels = get_dataset_logits_and_labels(
        df_idrid, "idrid_test", model, device, cfg, report_dir
    )

    idrid_results = run_idrid_recalibration(
        idrid_logits=idrid_logits,
        idrid_labels=idrid_labels,
        frozen_scaler=frozen_scaler,
        frozen_threshold=frozen_thresh,
        n_repeats=args.n_repeats,
        seed=args.seed,
    )

    isum = idrid_results["50_50_split_summary"]
    print(f"  ✓ IDRiD Results Summary (N=516):")
    print(f"    - Frozen Baseline   : Sens={idrid_results['frozen_baseline']['sensitivity']:.4f}, Spec={idrid_results['frozen_baseline']['specificity']:.4f}")
    print(f"    - 50/50 Split (N=258): Sens={isum['sensitivity_mean']:.4f}±{isum['sensitivity_sd']:.4f}, Spec={isum['specificity_mean']:.4f}±{isum['specificity_sd']:.4f}")
    print(f"    - Oracle Upper Bound: Sens={idrid_results['oracle_upper_bound']['sensitivity']:.4f}, Spec={idrid_results['oracle_upper_bound']['specificity']:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Messidor-2 Policy Statement
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/3] Messidor-2 Policy:")
    print("  ℹ Messidor-2 skipped for recalibration splitting (paired eyes without patient IDs prevent leakage-free splits).")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Save Artifacts
    # ─────────────────────────────────────────────────────────────────────────
    recal_payload = {
        "run_name": run_name,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "n_repeats": args.n_repeats,
        "frozen_threshold": frozen_thresh,
        "aptos": aptos_results,
        "idrid": idrid_results,
        "messidor2_policy": "Skipped due to paired eyes without patient IDs; prevents intra-patient cross-contamination.",
    }

    # Save JSON
    json_path = report_dir / "recalibration_results.json"
    tmp_json = report_dir / f".tmp_{os.getpid()}_{json_path.name}"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(recal_payload, f, indent=2)
    os.replace(tmp_json, json_path)
    print(f"\n✓ Saved recalibration results JSON: {json_path}")

    # Save Markdown
    md_content = generate_recalibration_markdown(run_name, aptos_results, idrid_results)
    md_path = report_dir / "recalibration_results.md"
    tmp_md = report_dir / f".tmp_{os.getpid()}_{md_path.name}"
    with open(tmp_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    os.replace(tmp_md, md_path)
    print(f"✓ Saved recalibration Markdown report: {md_path}")

    # Save Curve Figure
    curve_fig_path = figures_dir / "recalibration_curve.png"
    plot_recalibration_curves(aptos_results, curve_fig_path)
    print(f"✓ Saved recalibration curves figure: {curve_fig_path}")

    print("\n" + "=" * 75)
    print(f"Recalibration experiments completed successfully for '{run_name}'.")
    print("=" * 75)


if __name__ == "__main__":
    main()
