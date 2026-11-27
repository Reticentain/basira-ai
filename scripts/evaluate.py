"""
scripts/evaluate.py
===================
Evaluation pipeline for trained Diabetic Retinopathy screening models.

Key requirements:
  1. Load model checkpoint (best.pt) safely on CPU via load_checkpoint
     (map_location="cpu", weights_only=False), then move to target device.
  2. Run inference on eyepacs_val to obtain logits.
  3. Fit Platt scaling (logistic regression on logits: scale + bias) EXCLUSIVELY on VAL.
  4. Select operating threshold on VAL ONLY: the threshold achieving specificity >= 0.90,
     expressed on calibrated probabilities. Save scaler parameters + threshold to
     Drive reports/<run-name>/calibration.json.
  5. Evaluate using the same eval transforms (cached 256x256 PNGs, no augmentation) on:
       - eyepacs_test (53,576)
       - aptos_deduped (3,487; external unseen benchmark)
       - messidor2_test (1,744)
       - idrid_test (516)
  6. For every dataset report:
       - N, prevalence
       - ROC-AUC with 95% bootstrap CI (n=1000, seed from config)
       - Sensitivity and Specificity at FIXED val threshold with 95% CIs
       - Positive Predictive Value (PPV) and Negative Predictive Value (NPV)
       - Expected Calibration Error (ECE) before and after Platt scaling
       - Empirical fraction flagged referable per true DR grade (0 to 4)
       - Oracle metric (clearly labelled "oracle (tuned on this set, not deployable)"):
         sensitivity at 90% specificity using a threshold picked on that dataset.
       - Patient-level cluster bootstrap for EyePACS; image-level bootstrap for others
         with explicit limitation note ("no patient IDs available").
  7. Save all artifacts to Drive reports/<run-name>/:
       - Per-image prediction CSVs: <dataset>_predictions.csv
       - results.json
       - results_table.md (Markdown results table)
       - roc_curves.png (ROC curves for all datasets on one plot)
       - reliability_diagram.png (reliability curves before/after calibration)
       - training_curves.png (training curves from metrics.csv)
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
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Ensure src/ is on sys.path
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from basira.config import apply_overrides, load_config
from basira.data.cache import get_current_git_commit
from basira.data.dataset import CachedFundusDataset
from basira.data.loaders import read_split_csv
from basira.eval.bootstrap import bootstrap_ci
from basira.eval.calibration import PlattScaler, compute_reliability_curve
from basira.eval.metrics import (
    compute_auc,
    compute_ppv_npv,
    expected_calibration_error,
    fraction_referable_per_grade,
    sensitivity_at_specificity,
)


def find_split_file(filename: str, cfg: DictConfig) -> pathlib.Path:
    """Locate a split CSV in Drive output or local reports/splits directory."""
    candidates = [
        pathlib.Path(cfg.output.splits) / filename,
        _REPO_ROOT / "reports" / "splits" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Split file '{filename}' not found. Checked:\n"
        + "\n".join(f"  - {p}" for p in candidates)
        + "\nPlease ensure splits have been generated."
    )


def load_eval_checkpoint(
    filepath: pathlib.Path | str,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Load a checkpoint containing model weights, config, and metadata.

    Loads with map_location="cpu" and weights_only=False (for full dicts and RNG states).
    """
    return torch.load(filepath, map_location=map_location, weights_only=False)


def build_and_load_model(
    ckpt_path: pathlib.Path | str,
    device: torch.device,
    backbone_override: Optional[str] = None,
    cfg: Optional[DictConfig] = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Instantiate model backbone and load trained weights.

    Parameters
    ----------
    ckpt_path : Path or str
        Path to checkpoint (best.pt).
    device : torch.device
        Device to move model to.
    backbone_override : str, optional
        Explicit model architecture override.
    cfg : DictConfig, optional
        Loaded configuration.

    Returns
    -------
    model : nn.Module
    ckpt : dict[str, Any]
    """
    ckpt = load_eval_checkpoint(ckpt_path, map_location="cpu")

    # Determine backbone name
    backbone = None
    if backbone_override:
        backbone = backbone_override
    elif isinstance(ckpt, dict) and ckpt.get("config"):
        cfg_in_ckpt = ckpt["config"]
        if isinstance(cfg_in_ckpt, dict):
            backbone = cfg_in_ckpt.get("model", {}).get("backbone")
        elif hasattr(cfg_in_ckpt, "model"):
            backbone = getattr(cfg_in_ckpt.model, "backbone", None)
    if not backbone and cfg is not None and hasattr(cfg, "model") and hasattr(cfg.model, "backbone"):
        backbone = cfg.model.backbone
    if not backbone:
        backbone = "efficientnet_b0"

    timm_name = "mobilenetv3_large_100" if backbone in ("mobilenetv3", "mobilenetv3_large_100") else backbone

    # Task is binary referable DR: 1 logit output
    model = timm.create_model(timm_name, pretrained=False, num_classes=1)

    state_dict = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, ckpt if isinstance(ckpt, dict) else {}


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run forward pass across DataLoader and return raw logits and sigmoid probabilities.

    Returns
    -------
    logits : np.ndarray of shape (N,)
    probs : np.ndarray of shape (N,)
    """
    model.eval()
    all_logits: list[float] = []

    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = model(images).view(-1)
            else:
                out = model(images).view(-1)
            all_logits.extend(out.cpu().tolist())

    logits = np.array(all_logits, dtype=np.float64)
    # Numerically stable sigmoid for raw probabilities
    probs = np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-logits)),
        np.exp(logits) / (1.0 + np.exp(logits)),
    )
    return logits, probs


def evaluate_dataset(
    df: pd.DataFrame,
    logits: np.ndarray,
    raw_probs: np.ndarray,
    scaler: PlattScaler,
    fixed_threshold: float,
    dataset_name: str,
    n_bootstrap: int = 1000,
    seed: int = 0,
    target_specificity: float = 0.90,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Compute full evaluation metrics for a single dataset given fixed scaler and threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset split DataFrame containing 'image_id', 'grade', and optional 'patient_id'.
    logits : np.ndarray
        Model output logits.
    raw_probs : np.ndarray
        Sigmoided probabilities before calibration.
    scaler : PlattScaler
        Platt scaler fit strictly on validation set.
    fixed_threshold : float
        Decision threshold selected strictly on validation set.
    dataset_name : str
        Name of the dataset being evaluated.
    n_bootstrap : int
        Number of bootstrap iterations.
    seed : int
        RNG seed for bootstrap resampling.
    target_specificity : float
        Target operating specificity for oracle comparison.

    Returns
    -------
    metrics : dict[str, Any]
    pred_df : pd.DataFrame
    """
    cal_probs = scaler.predict_proba(logits)
    y_true = (df["grade"].values >= 2).astype(np.int64)
    n_samples = len(df)
    prevalence = float(np.mean(y_true))

    # Patient IDs for cluster bootstrap (EyePACS only; None for others)
    patient_ids = None
    if "patient_id" in df.columns:
        p_vals = df["patient_id"].values
        # If any non-null, non-empty patient IDs exist
        if any(p is not None and str(p).strip() not in ("", "nan", "None") for p in p_vals):
            patient_ids = p_vals

    bootstrap_limitation = (
        "patient-level cluster bootstrap (preserves inter-eye correlation)"
        if patient_ids is not None
        else "no patient IDs available; image-level bootstrap used"
    )

    # 1. Point metrics & 95% bootstrap CIs at FIXED validation threshold
    ci_results = bootstrap_ci(
        y_true=y_true,
        y_prob=cal_probs,
        threshold=fixed_threshold,
        patient_ids=patient_ids,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    # 2. Calibration ECE before and after Platt scaling
    ece_raw = expected_calibration_error(y_true, raw_probs, n_bins=10)
    ece_cal = expected_calibration_error(y_true, cal_probs, n_bins=10)

    # 3. Predictions at fixed threshold and fraction referable per true grade
    preds = (cal_probs >= fixed_threshold).astype(np.int64)
    frac_referable = fraction_referable_per_grade(df["grade"].values, preds, num_classes=5)

    # 4. Oracle metric (tuned on this set, not deployable)
    oracle_sens, oracle_thresh = sensitivity_at_specificity(
        y_true,
        cal_probs,
        target_specificity=target_specificity,
    )
    oracle_info = {
        "threshold": float(oracle_thresh),
        "sensitivity": float(oracle_sens),
        "target_specificity": float(target_specificity),
        "description": "oracle (tuned on this set, not deployable; upper-bound comparison only)",
    }

    # Consolidated metrics dictionary
    metrics = {
        "dataset": dataset_name,
        "N": int(n_samples),
        "prevalence": float(prevalence),
        "bootstrap_limitation": bootstrap_limitation,
        "fixed_val_threshold": float(fixed_threshold),
        "auc": ci_results["auc"],
        "sensitivity": ci_results["sensitivity"],
        "specificity": ci_results["specificity"],
        "ppv": ci_results["ppv"],
        "npv": ci_results["npv"],
        "ece_before_calibration": float(ece_raw),
        "ece_after_calibration": float(ece_cal),
        "fraction_referable_per_grade": {str(k): float(v) for k, v in frac_referable.items()},
        "oracle (tuned on this set, not deployable)": oracle_info,
    }

    # Per-image prediction DataFrame
    pred_df = pd.DataFrame({
        "image_id": df["image_id"].astype(str).values,
        "source": df["source"].astype(str).values if "source" in df.columns else np.array([dataset_name] * n_samples, dtype=object),
        "patient_id": df["patient_id"].values if "patient_id" in df.columns else [None] * n_samples,
        "grade": df["grade"].astype(int).values,
        "logit": logits,
        "prob": raw_probs,
        "calibrated_prob": cal_probs,
    })

    return metrics, pred_df


def generate_markdown_table(
    results: dict[str, dict[str, Any]],
    val_threshold: float,
    run_name: str,
) -> str:
    """Generate GitHub-flavored Markdown table summarizing evaluation results."""
    lines = [
        f"# Basira AI — Evaluation Report: `{run_name}`",
        "",
        f"**Operating Threshold (Fit on EyePACS Val):** `{val_threshold:.4f}` (target specificity ≥ 0.90)",
        "**Calibration:** Platt scaling (fit exclusively on EyePACS Val logits)",
        "",
        "| Dataset | N | Prev | AUC (95% CI) | Sens @ Fixed Val Thresh | Spec @ Fixed Val Thresh | PPV | NPV | ECE (Raw → Cal) | Oracle Sens @ 90% Spec |",
        "|:---|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for dname, m in results.items():
        n = m["N"]
        prev = f"{m['prevalence'] * 100:.1f}%"
        auc = f"{m['auc']['val']:.4f} ({m['auc']['ci_lower']:.4f}-{m['auc']['ci_upper']:.4f})"
        sens = f"{m['sensitivity']['val']:.4f} ({m['sensitivity']['ci_lower']:.4f}-{m['sensitivity']['ci_upper']:.4f})"
        spec = f"{m['specificity']['val']:.4f} ({m['specificity']['ci_lower']:.4f}-{m['specificity']['ci_upper']:.4f})"
        ppv = f"{m['ppv']['val']:.4f} ({m['ppv']['ci_lower']:.4f}-{m['ppv']['ci_upper']:.4f})" if not np.isnan(m['ppv']['val']) else "N/A"
        npv = f"{m['npv']['val']:.4f} ({m['npv']['ci_lower']:.4f}-{m['npv']['ci_upper']:.4f})" if not np.isnan(m['npv']['val']) else "N/A"
        ece = f"{m['ece_before_calibration']:.4f} → {m['ece_after_calibration']:.4f}"
        oracle = f"{m['oracle (tuned on this set, not deployable)']['sensitivity']:.4f}"

        lines.append(
            f"| `{dname}` | {n:,} | {prev} | {auc} | {sens} | {spec} | {ppv} | {npv} | {ece} | {oracle} |"
        )

    lines.extend([
        "",
        "### Notes & Methodology",
        "- **Validation-Only Calibration:** Platt scaling parameters and operating threshold were determined exclusively using `eyepacs_val`.",
        "- **Fixed Threshold Evaluation:** Test datasets (`eyepacs_test`, `aptos_deduped`, `messidor2_test`, `idrid_test`) were evaluated using the frozen validation threshold without tuning.",
        "- **Bootstrap Resampling:** 95% confidence intervals derived from 1,000 bootstrap iterations. Patient-level clustering used for EyePACS (accounting for inter-eye correlation); image-level bootstrap for datasets lacking patient identifiers.",
        "- **Oracle Field:** `oracle (tuned on this set, not deployable)` represents the theoretical upper bound when selecting a threshold at 90% specificity on that exact evaluation dataset.",
    ])
    return "\n".join(lines) + "\n"


def plot_roc_curves(
    eval_data: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    val_threshold: float,
    output_path: pathlib.Path,
) -> None:
    """
    Plot combined ROC curves for all evaluated datasets on a single figure.

    Parameters
    ----------
    eval_data : dict
        Mapping dataset_name -> (y_true, cal_probs, metrics_dict)
    val_threshold : float
        Operating threshold from validation set.
    output_path : Path
        PNG destination.
    """
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (dname, (y_true, cal_probs, m)) in enumerate(eval_data.items()):
        color = colors[idx % len(colors)]
        if len(np.unique(y_true)) >= 2:
            fpr, tpr, thresholds = roc_curve(y_true, cal_probs)
            auc_val = m["auc"]["val"]
            auc_lo = m["auc"]["ci_lower"]
            auc_hi = m["auc"]["ci_upper"]
            label = f"{dname} (AUC = {auc_val:.3f} [{auc_lo:.3f}, {auc_hi:.3f}])"
            ax.plot(fpr, tpr, label=label, color=color, lw=2.0)

            # Operating point marker at fixed validation threshold
            sens_val = m["sensitivity"]["val"]
            spec_val = m["specificity"]["val"]
            if not np.isnan(sens_val) and not np.isnan(spec_val):
                ax.scatter(
                    [1.0 - spec_val],
                    [sens_val],
                    color=color,
                    s=80,
                    edgecolor="black",
                    zorder=5,
                )

    # Reference diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.2, label="Random (AUC = 0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(f"ROC Curves — Operating Threshold = {val_threshold:.4f}", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".tmp_{os.getpid()}_{output_path.name}"
    fig.savefig(tmp_path, dpi=300)
    plt.close(fig)
    os.replace(tmp_path, output_path)


def plot_reliability_diagrams(
    eval_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]],
    output_path: pathlib.Path,
) -> None:
    """
    Plot reliability diagrams (confidence vs accuracy) before and after Platt calibration.

    Parameters
    ----------
    eval_data : dict
        Mapping dataset_name -> (y_true, raw_probs, cal_probs, metrics_dict)
    output_path : Path
        PNG destination.
    """
    n_datasets = len(eval_data)
    fig, axes = plt.subplots(n_datasets, 2, figsize=(10, 3.5 * n_datasets), squeeze=False, dpi=300)

    for i, (dname, (y_true, raw_probs, cal_probs, m)) in enumerate(eval_data.items()):
        # Raw / Uncalibrated
        ax_raw = axes[i, 0]
        confs_raw, accs_raw, counts_raw, ece_raw = compute_reliability_curve(y_true, raw_probs, n_bins=10)
        ax_raw.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect")
        if len(confs_raw) > 0:
            ax_raw.plot(confs_raw, accs_raw, "s-", color="#d62728", lw=2, markersize=6, label=f"ECE = {ece_raw:.4f}")
            ax_raw.bar(confs_raw, accs_raw, width=0.08, alpha=0.25, color="#d62728")
        ax_raw.set_title(f"{dname} — Before Calibration", fontsize=11, fontweight="bold")
        ax_raw.set_xlabel("Predicted Probability", fontsize=10)
        ax_raw.set_ylabel("Empirical Accuracy", fontsize=10)
        ax_raw.set_xlim([0, 1])
        ax_raw.set_ylim([0, 1])
        ax_raw.grid(True, linestyle=":", alpha=0.6)
        ax_raw.legend(loc="upper left", fontsize=9)

        # Calibrated
        ax_cal = axes[i, 1]
        confs_cal, accs_cal, counts_cal, ece_cal = compute_reliability_curve(y_true, cal_probs, n_bins=10)
        ax_cal.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect")
        if len(confs_cal) > 0:
            ax_cal.plot(confs_cal, accs_cal, "o-", color="#2ca02c", lw=2, markersize=6, label=f"ECE = {ece_cal:.4f}")
            ax_cal.bar(confs_cal, accs_cal, width=0.08, alpha=0.25, color="#2ca02c")
        ax_cal.set_title(f"{dname} — After Platt Calibration", fontsize=11, fontweight="bold")
        ax_cal.set_xlabel("Calibrated Probability", fontsize=10)
        ax_cal.set_ylabel("Empirical Accuracy", fontsize=10)
        ax_cal.set_xlim([0, 1])
        ax_cal.set_ylim([0, 1])
        ax_cal.grid(True, linestyle=":", alpha=0.6)
        ax_cal.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".tmp_{os.getpid()}_{output_path.name}"
    fig.savefig(tmp_path, dpi=300)
    plt.close(fig)
    os.replace(tmp_path, output_path)


def plot_training_curves(metrics_csv_path: pathlib.Path, output_path: pathlib.Path) -> bool:
    """
    Plot training curves (Loss, AUC, Sens@90%Spec, ECE, LR) from metrics.csv.

    Returns
    -------
    bool
        True if figure generated, False if metrics.csv not found or unparseable.
    """
    if not metrics_csv_path.exists():
        print(f"Metrics CSV '{metrics_csv_path}' not found. Skipping training curves figure.")
        return False

    try:
        df_m = pd.read_csv(metrics_csv_path)
    except Exception as exc:
        print(f"Failed to read '{metrics_csv_path}': {exc}. Skipping training curves.")
        return False

    if df_m.empty or "epoch" not in df_m.columns:
        print(f"Metrics CSV '{metrics_csv_path}' is empty or invalid. Skipping.")
        return False

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=300)
    epochs = df_m["epoch"].values

    # 1. Loss
    ax = axes[0, 0]
    if "train_loss" in df_m.columns:
        ax.plot(epochs, df_m["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    if "val_loss" in df_m.columns:
        ax.plot(epochs, df_m["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    ax.set_title("Training & Validation Loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # 2. Validation AUC & Sensitivity
    ax = axes[0, 1]
    if "val_auc" in df_m.columns:
        ax.plot(epochs, df_m["val_auc"], label="Val AUC", color="#2ca02c", lw=2)
    if "val_sens" in df_m.columns:
        ax.plot(epochs, df_m["val_sens"], label="Val Sens @ 90% Spec", color="#d62728", lw=2)
    ax.set_title("Validation AUC & Sensitivity", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # 3. Expected Calibration Error (ECE)
    ax = axes[1, 0]
    if "val_ece" in df_m.columns:
        ax.plot(epochs, df_m["val_ece"], label="Val ECE", color="#9467bd", lw=2)
    ax.set_title("Validation Expected Calibration Error", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ECE")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # 4. Learning Rate
    ax = axes[1, 1]
    if "lr" in df_m.columns:
        ax.plot(epochs, df_m["lr"], label="Learning Rate", color="#8c564b", lw=2)
        ax.set_yscale("log")
    ax.set_title("Learning Rate Schedule", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate (log)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".tmp_{os.getpid()}_{output_path.name}"
    fig.savefig(tmp_path, dpi=300)
    plt.close(fig)
    os.replace(tmp_path, output_path)
    return True


def save_atomic_json(data: dict[str, Any], filepath: pathlib.Path) -> None:
    """Save dictionary to JSON atomically."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.parent / f".tmp_{os.getpid()}_{filepath.name}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, filepath)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained Diabetic Retinopathy screening baseline.")
    parser.add_argument("--run-name", type=str, required=True, help="Run name (e.g. b0_referable_s42)")
    parser.add_argument("--ckpt", type=str, default="best.pt", help="Checkpoint filename or path (default: best.pt)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["eyepacs_test", "aptos_deduped", "messidor2_test", "idrid_test"],
        help="Datasets to evaluate (default: eyepacs_test aptos_deduped messidor2_test idrid_test)",
    )
    parser.add_argument("--model", type=str, default=None, help="Backbone model override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers override")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Config overrides as key=value pairs (e.g. evaluation.bootstrap_n=500)",
    )
    return parser.parse_args(argv)


def build_eval_config(args: argparse.Namespace) -> DictConfig:
    """Load base configuration and apply CLI key=value overrides."""
    cfg = load_config()
    if args.set:
        apply_overrides(cfg, args.set)
    return cfg


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    # 1. Configuration with CLI overrides
    cfg = build_eval_config(args)

    run_name = args.run_name.strip()
    target_spec = float(cfg.evaluation.target_specificity)
    n_bootstrap = int(cfg.evaluation.bootstrap_n)
    seed = int(cfg.seeds.bootstrap_seed)

    # 2. Checkpoint path resolution
    ckpt_arg = pathlib.Path(args.ckpt)
    if ckpt_arg.exists():
        ckpt_path = ckpt_arg
    else:
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
            raise FileNotFoundError(
                f"Checkpoint '{args.ckpt}' not found for run '{run_name}'. Checked:\n"
                + "\n".join(f"  - {p}" for p in candidates)
            )

    # 3. Output directory setup
    report_dir = pathlib.Path(cfg.output.reports) / run_name
    report_dir.mkdir(parents=True, exist_ok=True)

    # 4. Device and Model setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print(f"Basira AI — Model Evaluation Pipeline")
    print(f"Run Name        : {run_name}")
    print(f"Checkpoint      : {ckpt_path}")
    print(f"Device          : {device}")
    print(f"Report Dir      : {report_dir}")
    print(f"Target Spec     : {target_spec:.2f}")
    print(f"Bootstrap N     : {n_bootstrap} (seed={seed})")
    print("=" * 75)

    model, ckpt_meta = build_and_load_model(ckpt_path, device, backbone_override=args.model, cfg=cfg)

    batch_size = args.batch_size if args.batch_size is not None else int(cfg.training.batch_size)
    num_workers = args.num_workers if args.num_workers is not None else int(cfg.training.num_workers)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Validation Set Inference & Calibration (VAL ONLY)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/3] Running inference on EyePACS Validation Set for calibration...")
    val_split_file = find_split_file("eyepacs_val.csv", cfg)
    df_val = read_split_csv(val_split_file)
    val_dataset = CachedFundusDataset(df_val, cfg=cfg, train=False, task="referable", check_files=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_logits, val_raw_probs = run_inference(model, val_loader, device)
    val_labels = (df_val["grade"].values >= 2).astype(np.int64)

    # Fit Platt scaling strictly on validation set
    print("  Fitting Platt scaling strictly on validation logits...")
    scaler = PlattScaler()
    scaler.fit(val_logits, val_labels)
    val_cal_probs = scaler.predict_proba(val_logits)

    # Determine operating threshold achieving target specificity on VAL ONLY
    val_sens, val_thresh = sensitivity_at_specificity(val_labels, val_cal_probs, target_specificity=target_spec)
    val_auc = compute_auc(val_labels, val_cal_probs)
    val_ece_before = expected_calibration_error(val_labels, val_raw_probs, n_bins=10)
    val_ece_after = expected_calibration_error(val_labels, val_cal_probs, n_bins=10)

    print(f"  ✓ Calibration Fit Complete:")
    print(f"    - Platt parameters : slope = {scaler.slope:.4f}, intercept = {scaler.intercept:.4f}")
    print(f"    - Selected Thresh  : {val_thresh:.4f} (Validation Specificity ≥ {target_spec:.2f})")
    print(f"    - Val AUC          : {val_auc:.4f}")
    print(f"    - Val Sens@90%Spec : {val_sens:.4f}")
    print(f"    - Val ECE          : {val_ece_before:.4f} → {val_ece_after:.4f}")

    # Save calibration.json atomically to Drive reports/<run-name>/calibration.json
    calibration_info = {
        "method": "platt_scaling",
        "slope": float(scaler.slope),
        "intercept": float(scaler.intercept),
        "operating_threshold": float(val_thresh),
        "target_specificity": float(target_spec),
        "validation_metrics": {
            "auc": float(val_auc),
            "sensitivity": float(val_sens),
            "ece_before_calibration": float(val_ece_before),
            "ece_after_calibration": float(val_ece_after),
            "n_samples": len(val_labels),
        },
    }
    save_atomic_json(calibration_info, report_dir / "calibration.json")
    print(f"  ✓ Saved calibration details to: {report_dir / 'calibration.json'}")

    # Evaluate validation set with the fixed threshold
    val_metrics, df_val_preds = evaluate_dataset(
        df=df_val,
        logits=val_logits,
        raw_probs=val_raw_probs,
        scaler=scaler,
        fixed_threshold=val_thresh,
        dataset_name="eyepacs_val",
        n_bootstrap=n_bootstrap,
        seed=seed,
        target_specificity=target_spec,
    )
    # Save validation predictions CSV
    val_pred_csv = report_dir / "eyepacs_val_predictions.csv"
    df_val_preds.to_csv(val_pred_csv, index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Evaluate All Target Test Datasets (with frozen scaler and threshold)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/3] Evaluating target benchmark datasets with FROZEN threshold...")
    all_results: dict[str, dict[str, Any]] = {"eyepacs_val": val_metrics}
    roc_plot_data: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {
        "eyepacs_val": (val_labels, val_cal_probs, val_metrics)
    }
    rel_plot_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {
        "eyepacs_val": (val_labels, val_raw_probs, val_cal_probs, val_metrics)
    }

    for dname in args.datasets:
        dname_clean = dname.strip()
        print(f"\n  Evaluating dataset: '{dname_clean}'...")
        split_file = find_split_file(f"{dname_clean}.csv", cfg)
        df_test = read_split_csv(split_file)

        # Build CachedFundusDataset (fails clearly if cache files or manifest missing)
        test_dataset = CachedFundusDataset(df_test, cfg=cfg, train=False, task="referable", check_files=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        test_logits, test_raw_probs = run_inference(model, test_loader, device)

        # Evaluate using FIXED validation threshold and FROZEN Platt scaler
        test_metrics, df_test_preds = evaluate_dataset(
            df=df_test,
            logits=test_logits,
            raw_probs=test_raw_probs,
            scaler=scaler,
            fixed_threshold=val_thresh,
            dataset_name=dname_clean,
            n_bootstrap=n_bootstrap,
            seed=seed,
            target_specificity=target_spec,
        )

        all_results[dname_clean] = test_metrics

        # Save per-image predictions CSV
        pred_csv_path = report_dir / f"{dname_clean}_predictions.csv"
        df_test_preds.to_csv(pred_csv_path, index=False)
        print(f"    - Saved per-image predictions to: {pred_csv_path}")

        # Accumulate plotting data
        y_test_true = (df_test["grade"].values >= 2).astype(np.int64)
        test_cal_probs = scaler.predict_proba(test_logits)
        roc_plot_data[dname_clean] = (y_test_true, test_cal_probs, test_metrics)
        rel_plot_data[dname_clean] = (y_test_true, test_raw_probs, test_cal_probs, test_metrics)

        print(
            f"    - N={test_metrics['N']:,} | "
            f"AUC={test_metrics['auc']['val']:.4f} ({test_metrics['auc']['ci_lower']:.4f}-{test_metrics['auc']['ci_upper']:.4f}) | "
            f"Sens={test_metrics['sensitivity']['val']:.4f} | "
            f"Spec={test_metrics['specificity']['val']:.4f} | "
            f"ECE={test_metrics['ece_before_calibration']:.4f} → {test_metrics['ece_after_calibration']:.4f}"
        )
        oracle = test_metrics["oracle (tuned on this set, not deployable)"]
        print(f"    - Oracle Sens @ 90% Spec: {oracle['sensitivity']:.4f} (thresh={oracle['threshold']:.4f})")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Save Consolidated Results, Tables, and Figures
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/3] Saving summary reports and publication figures...")

    # Save results.json
    results_payload = {
        "run_name": run_name,
        "checkpoint": str(ckpt_path),
        "git_commit": get_current_git_commit(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "operating_threshold": float(val_thresh),
        "calibration": calibration_info,
        "datasets": all_results,
    }
    save_atomic_json(results_payload, report_dir / "results.json")
    print(f"  ✓ Saved results JSON: {report_dir / 'results.json'}")

    # Save Markdown results table
    md_table = generate_markdown_table(all_results, val_threshold=val_thresh, run_name=run_name)
    with open(report_dir / "results_table.md", "w", encoding="utf-8") as f:
        f.write(md_table)
    print(f"  ✓ Saved Markdown table: {report_dir / 'results_table.md'}")

    # Generate ROC curves figure
    roc_fig_path = report_dir / "roc_curves.png"
    plot_roc_curves(roc_plot_data, val_threshold=val_thresh, output_path=roc_fig_path)
    print(f"  ✓ Saved ROC curves plot: {roc_fig_path}")

    # Generate Reliability diagram figure
    rel_fig_path = report_dir / "reliability_diagram.png"
    plot_reliability_diagrams(rel_plot_data, output_path=rel_fig_path)
    print(f"  ✓ Saved Reliability diagram: {rel_fig_path}")

    # Generate Training curves figure from metrics.csv
    ckpt_dir = ckpt_path.parent
    metrics_csv = ckpt_dir / "metrics.csv"
    train_curves_path = report_dir / "training_curves.png"
    plot_training_curves(metrics_csv, output_path=train_curves_path)
    print(f"  ✓ Saved Training curves plot: {train_curves_path}")

    print("\n" + "=" * 75)
    print(f"Evaluation complete for run '{run_name}'. All artifacts saved to:\n  {report_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
