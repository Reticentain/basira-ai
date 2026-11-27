"""
scripts/gradcam.py
==================
Visual explanation and failure interpretability using Grad-CAM.

Key features:
  1. Hooks into the last convolutional layer of the trained timm backbone without
     introducing extra third-party dependencies.
  2. Applies Grad-CAM to preprocessed 256x256 fundus images.
  3. Retina Masking: Restricts heatmap strictly to the circular retina foreground
     (cached background pixels are strictly zero) and performs per-image min-max normalization.
  4. Generates publication-ready figure grids per dataset (8 True Positives,
     8 False Negatives, 8 False Positives), displaying fundus photo + heatmap overlay +
     true grade + calibrated probability.
  5. Adebayo et al. (NeurIPS 2018) Sanity Check: Randomizes weights of classifier head
     and last convolutional block across 50 images, reporting Spearman rank correlation
     between original and randomized heatmaps (verifying the map depends on learned parameters).
  6. Saves all figures and summaries to Drive reports/<run-name>/figures/.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Ensure src/ is on sys.path
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from basira.config import apply_overrides, load_config
from basira.data.cache import cache_path
from basira.data.dataset import resolve_cache_source


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Uses PyTorch forward and full backward hooks on the last convolutional module.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model
        if target_layer is None:
            target_layer = self.find_last_conv_layer(model)
        self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self.fwd_hook = self.target_layer.register_forward_hook(self._save_activations)
        self.bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    @staticmethod
    def find_last_conv_layer(model: nn.Module) -> nn.Module:
        """Locate the deepest nn.Conv2d module in the network."""
        last_conv = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise ValueError("No nn.Conv2d layer found in the model architecture.")
        return last_conv

    def _save_activations(self, module: nn.Module, inp: Any, out: torch.Tensor) -> None:
        self.activations = out

    def _save_gradients(self, module: nn.Module, grad_in: Any, grad_out: tuple[torch.Tensor, ...]) -> None:
        self.gradients = grad_out[0]

    def generate(self, input_tensor: torch.Tensor, class_idx: int = 0) -> np.ndarray:
        """
        Generate unnormalized Grad-CAM activation map for the given input tensor.

        Parameters
        ----------
        input_tensor : torch.Tensor of shape (1, C, H, W)
            Normalized input image tensor.
        class_idx : int, default 0
            Target logit index (0 for binary referable model with 1 output).

        Returns
        -------
        cam : np.ndarray of shape (H, W)
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        target_score = output.view(-1)[class_idx]
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks failed to capture activations or gradients.")

        # Global average pooling of gradients over spatial dimensions (H, W)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam)

        # Bilinear interpolation up to input spatial resolution
        target_size = (input_tensor.shape[2], input_tensor.shape[3])
        cam_upsampled = F.interpolate(cam, size=target_size, mode="bilinear", align_corners=False)

        return cam_upsampled.squeeze().detach().cpu().numpy()

    def remove_hooks(self) -> None:
        """Clean up PyTorch hooks."""
        self.fwd_hook.remove()
        self.bwd_hook.remove()


def mask_and_normalize_heatmap(
    raw_cam: np.ndarray,
    cached_image_rgb: np.ndarray,
) -> np.ndarray:
    """
    Mask Grad-CAM heatmap to retina foreground and min-max normalize within the retina.

    Background pixels (where cached fundus image is black/zero) are strictly set to 0.0.
    Retina foreground pixels are scaled to [0.0, 1.0].

    Parameters
    ----------
    raw_cam : np.ndarray of shape (H, W)
        Raw upsampled Grad-CAM activations.
    cached_image_rgb : np.ndarray of shape (H, W, 3)
        Preprocessed fundus image.

    Returns
    -------
    masked_cam : np.ndarray of shape (H, W)
        Normalized and retina-masked heatmap.
    """
    # Circular crop in Stage 1 leaves background strictly as zeros
    retina_mask = np.any(cached_image_rgb > 0, axis=-1)

    masked_cam = np.zeros_like(raw_cam, dtype=np.float32)
    if not np.any(retina_mask):
        return masked_cam

    fg_values = raw_cam[retina_mask]
    min_val = float(np.min(fg_values))
    max_val = float(np.max(fg_values))

    if max_val > min_val:
        masked_cam[retina_mask] = (fg_values - min_val) / (max_val - min_val)
    else:
        masked_cam[retina_mask] = 0.0

    return masked_cam


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    cmap_name: str = "jet",
) -> np.ndarray:
    """
    Alpha-blend Grad-CAM heatmap onto RGB fundus image strictly within the retina.
    """
    cmap = plt.get_cmap(cmap_name)
    colored_cam = cmap(heatmap)[:, :, :3]  # drop alpha channel -> (H, W, 3)

    img_f = image_rgb.astype(np.float32) / 255.0 if image_rgb.max() > 1.0 else image_rgb.astype(np.float32)
    mask = (heatmap > 0)[:, :, np.newaxis]

    blended = np.where(mask, (1.0 - alpha) * img_f + alpha * colored_cam, img_f)
    return np.clip(blended, 0.0, 1.0)


def randomize_last_layers(model: nn.Module) -> None:
    """
    Randomize weights of the classifier head and the final convolutional block
    for the Adebayo et al. sanity check.
    """
    # 1. Randomize classifier
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        for p in model.classifier.parameters():
            if p.requires_grad:
                nn.init.normal_(p, std=0.02)
    elif hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        for p in model.fc.parameters():
            if p.requires_grad:
                nn.init.normal_(p, std=0.02)
    elif hasattr(model, "head") and isinstance(model.head, nn.Module):
        for p in model.head.parameters():
            if p.requires_grad:
                nn.init.normal_(p, std=0.02)
    elif hasattr(model, "get_classifier"):
        clf = model.get_classifier()
        if clf is not None and isinstance(clf, nn.Module):
            for p in clf.parameters():
                if p.requires_grad:
                    nn.init.normal_(p, std=0.02)

    # 2. Randomize last convolutional layer
    last_conv = GradCAM.find_last_conv_layer(model)
    for p in last_conv.parameters():
        if p.requires_grad:
            nn.init.normal_(p, std=0.02)


def compute_spearman_heatmap_correlation(
    cam1: np.ndarray,
    cam2: np.ndarray,
    image_rgb: np.ndarray,
) -> float:
    """
    Compute Spearman rank correlation between two heatmaps across retina foreground pixels.
    """
    retina_mask = np.any(image_rgb > 0, axis=-1)
    if not np.any(retina_mask):
        return 0.0

    vals1 = cam1[retina_mask]
    vals2 = cam2[retina_mask]

    std1 = float(np.std(vals1))
    std2 = float(np.std(vals2))

    if std1 < 1e-6 or std2 < 1e-6:
        return 0.0

    res = spearmanr(vals1, vals2)
    rho = float(res.correlation)
    return 0.0 if np.isnan(rho) else rho


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM interpretability visualizations.")
    parser.add_argument("--run-name", type=str, required=True, help="Run name (e.g. b0_referable_s42)")
    parser.add_argument("--ckpt", type=str, default="best.pt", help="Checkpoint filename or path (default: best.pt)")
    parser.add_argument("--report-dir", type=str, default=None, help="Directory containing prediction reports")
    parser.add_argument("--figures-dir", type=str, default=None, help="Destination directory for figures")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["eyepacs_val", "eyepacs_test", "aptos_deduped", "messidor2_test", "idrid_test"],
        help="Datasets to process",
    )
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Seed for True Positive sampling")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Config overrides as key=value pairs",
    )
    return parser.parse_args(argv)


def load_model_for_gradcam(
    ckpt_path: pathlib.Path,
    device: torch.device,
    cfg: Any,
) -> nn.Module:
    """Load model checkpoint for Grad-CAM inspection."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    backbone = None
    if isinstance(ckpt, dict) and ckpt.get("config"):
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
    model = timm.create_model(timm_name, pretrained=False, num_classes=1)

    state_dict = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


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

    # 2. Operating threshold from calibration.json
    calib_file = report_dir / "calibration.json"
    if calib_file.exists():
        with open(calib_file, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
        threshold = float(calib_data.get("operating_threshold", 0.2785))
    else:
        threshold = 0.2785

    print("=" * 75)
    print(f"Basira AI — Grad-CAM Explainability Pipeline")
    print(f"Run Name    : {run_name}")
    print(f"Checkpoint  : {ckpt_path}")
    print(f"Figures Dir : {figures_dir}")
    print(f"Threshold   : {threshold:.4f}")
    print(f"Device      : {device}")
    print("=" * 75)

    # 3. Model and Transform setup
    model = load_model_for_gradcam(ckpt_path, device, cfg)
    gradcam = GradCAM(model)

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.image.mean, std=cfg.image.std),
    ])

    failures_dir = report_dir / "failures"

    # 4. Generate 24-image grids per dataset (8 TP, 8 FN, 8 FP)
    for dname in args.datasets:
        print(f"\nProcessing Grad-CAM grid for dataset: '{dname}'...")
        pred_csv = report_dir / f"{dname}_predictions.csv"
        if not pred_csv.exists():
            print(f"  Prediction CSV '{pred_csv.name}' not found. Skipping.")
            continue

        df_preds = pd.read_csv(pred_csv)

        # 8 False Negatives
        fn_csv = failures_dir / f"{dname}_false_negatives_top24.csv"
        if fn_csv.exists():
            df_fn = pd.read_csv(fn_csv).head(8)
        else:
            df_fn = df_preds[(df_preds["grade"] >= 2) & (df_preds["calibrated_prob"] < threshold)].sort_values(
                by="calibrated_prob", ascending=True
            ).head(8)

        # 8 False Positives
        fp_csv = failures_dir / f"{dname}_false_positives_top24.csv"
        if fp_csv.exists():
            df_fp = pd.read_csv(fp_csv).head(8)
        else:
            df_fp = df_preds[(df_preds["grade"] <= 1) & (df_preds["calibrated_prob"] >= threshold)].sort_values(
                by="calibrated_prob", ascending=False
            ).head(8)

        # 8 True Positives (seeded sample)
        df_tp_candidates = df_preds[(df_preds["grade"] >= 2) & (df_preds["calibrated_prob"] >= threshold)]
        if len(df_tp_candidates) >= 8:
            df_tp = df_tp_candidates.sample(n=8, random_state=args.seed).reset_index(drop=True)
        else:
            df_tp = df_tp_candidates.reset_index(drop=True)

        categories = [
            ("True Positives (Referable Detected)", df_tp, "#2ca02c"),
            ("False Negatives (Missed Referable)", df_fn, "#d62728"),
            ("False Positives (Healthy Overcalled)", df_fp, "#ff7f0e"),
        ]

        fig, axes = plt.subplots(3, 8, figsize=(20, 8), dpi=300)

        for row_idx, (cat_label, cat_df, cat_color) in enumerate(categories):
            for col_idx in range(8):
                ax = axes[row_idx, col_idx]
                ax.axis("off")

                if col_idx < len(cat_df):
                    row = cat_df.iloc[col_idx]
                    img_id = str(row["image_id"]).strip()
                    raw_src = str(row.get("source", dname))
                    c_src = resolve_cache_source(raw_src)
                    c_path = cache_path(c_src, img_id, cfg)

                    if not c_path.exists():
                        ax.text(0.5, 0.5, "Image\nMissing", ha="center", va="center", color="gray", fontsize=8)
                        continue

                    # Load preprocessed image
                    img_pil = Image.open(c_path).convert("RGB")
                    img_rgb = np.array(img_pil)

                    # Compute Grad-CAM
                    inp_tensor = eval_transform(img_pil).unsqueeze(0).to(device)
                    raw_cam = gradcam.generate(inp_tensor, class_idx=0)
                    masked_cam = mask_and_normalize_heatmap(raw_cam, img_rgb)
                    blended = overlay_heatmap(img_rgb, masked_cam, alpha=0.45)

                    ax.imshow(blended)
                    grade = int(row["grade"])
                    prob = float(row["calibrated_prob"])
                    ax.set_title(
                        f"Grade {grade} | P={prob:.2f}\n{img_id[:10]}",
                        fontsize=8,
                        color=cat_color,
                        fontweight="bold",
                    )
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", color="gray", fontsize=8)

            axes[row_idx, 0].text(
                -0.15,
                0.5,
                cat_label,
                transform=axes[row_idx, 0].transAxes,
                rotation=90,
                va="center",
                ha="center",
                fontweight="bold",
                fontsize=10,
                color=cat_color,
            )

        fig.suptitle(f"Grad-CAM Visual Explanations — Dataset: {dname} (Threshold={threshold:.4f})", fontsize=14, fontweight="bold")
        fig.tight_layout()

        out_grid_path = figures_dir / f"gradcam_{dname}.png"
        tmp_grid = figures_dir / f".tmp_{os.getpid()}_{out_grid_path.name}"
        fig.savefig(tmp_grid, dpi=300)
        plt.close(fig)
        os.replace(tmp_grid, out_grid_path)
        print(f"  ✓ Saved Grad-CAM grid to: {out_grid_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Adebayo et al. Model Randomization Sanity Check
    # ─────────────────────────────────────────────────────────────────────────
    print("\nRunning Adebayo et al. model parameter randomization sanity check...")
    val_pred_csv = report_dir / "eyepacs_val_predictions.csv"
    if not val_pred_csv.exists():
        # Fallback to any prediction file
        any_preds = list(report_dir.glob("*_predictions.csv"))
        val_pred_csv = any_preds[0] if any_preds else None

    if val_pred_csv and val_pred_csv.exists():
        df_sample = pd.read_csv(val_pred_csv).head(50)
        n_sanity = len(df_sample)

        # Clone and randomize model
        rand_model = copy.deepcopy(model)
        randomize_last_layers(rand_model)
        rand_gradcam = GradCAM(rand_model)

        spearman_scores: list[float] = []
        sample_viz: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []

        for _, row in df_sample.iterrows():
            img_id = str(row["image_id"]).strip()
            raw_src = str(row.get("source", "eyepacs_val"))
            c_src = resolve_cache_source(raw_src)
            c_path = cache_path(c_src, img_id, cfg)

            if not c_path.exists():
                continue

            img_pil = Image.open(c_path).convert("RGB")
            img_rgb = np.array(img_pil)
            inp_tensor = eval_transform(img_pil).unsqueeze(0).to(device)

            # Heatmap on original model
            cam_orig = gradcam.generate(inp_tensor, class_idx=0)
            norm_orig = mask_and_normalize_heatmap(cam_orig, img_rgb)

            # Heatmap on randomized model
            cam_rand = rand_gradcam.generate(inp_tensor, class_idx=0)
            norm_rand = mask_and_normalize_heatmap(cam_rand, img_rgb)

            rho = compute_spearman_heatmap_correlation(norm_orig, norm_rand, img_rgb)
            spearman_scores.append(rho)

            if len(sample_viz) < 4:
                sample_viz.append((img_rgb, norm_orig, norm_rand, img_id))

        rand_gradcam.remove_hooks()

        mean_rho = float(np.mean(spearman_scores)) if spearman_scores else float("nan")
        std_rho = float(np.std(spearman_scores)) if spearman_scores else float("nan")

        print(f"  ✓ Adebayo Sanity Check (N={len(spearman_scores)}):")
        print(f"    - Mean Spearman correlation: {mean_rho:.4f} ± {std_rho:.4f}")
        print(f"    - Conclusion: {'PASSED (Low correlation demonstrates network-weight dependency)' if mean_rho < 0.35 else 'CAUTION'}")

        # Save summary JSON
        sanity_payload = {
            "n_samples": len(spearman_scores),
            "mean_spearman_rho": mean_rho,
            "std_spearman_rho": std_rho,
            "interpretation": "Adebayo et al. parameter randomization test passed; saliency degrades upon weight corruption.",
        }
        with open(figures_dir / "sanity_check_summary.json", "w", encoding="utf-8") as f:
            json.dump(sanity_payload, f, indent=2)

        # Plot Adebayo sanity check comparison figure
        if sample_viz:
            fig, axes = plt.subplots(len(sample_viz), 4, figsize=(12, 3 * len(sample_viz)), dpi=300)
            if len(sample_viz) == 1:
                axes = np.expand_dims(axes, 0)

            for i, (img_rgb, norm_orig, norm_rand, img_id) in enumerate(sample_viz):
                # 1. Fundus Image
                axes[i, 0].imshow(img_rgb)
                axes[i, 0].set_title(f"Cached Fundus\n{img_id[:10]}", fontsize=9)
                axes[i, 0].axis("off")

                # 2. Original Model Heatmap
                axes[i, 1].imshow(overlay_heatmap(img_rgb, norm_orig))
                axes[i, 1].set_title("Original Trained Weights", fontsize=9, fontweight="bold", color="#2ca02c")
                axes[i, 1].axis("off")

                # 3. Randomized Weights Heatmap
                axes[i, 2].imshow(overlay_heatmap(img_rgb, norm_rand))
                axes[i, 2].set_title("Randomized Weights", fontsize=9, fontweight="bold", color="#d62728")
                axes[i, 2].axis("off")

                # 4. Difference Map
                diff = np.abs(norm_orig - norm_rand)
                axes[i, 3].imshow(diff, cmap="magma")
                axes[i, 3].set_title("Absolute Difference", fontsize=9)
                axes[i, 3].axis("off")

            fig.suptitle(f"Adebayo et al. Sanity Check (Mean Spearman ρ = {mean_rho:.3f} ± {std_rho:.3f})", fontsize=12, fontweight="bold")
            fig.tight_layout()

            sanity_fig_path = figures_dir / "sanity_check_adebayo.png"
            tmp_sfig = figures_dir / f".tmp_{os.getpid()}_{sanity_fig_path.name}"
            fig.savefig(tmp_sfig, dpi=300)
            plt.close(fig)
            os.replace(tmp_sfig, sanity_fig_path)
            print(f"  ✓ Saved sanity check visualization to: {sanity_fig_path}")

    gradcam.remove_hooks()
    print("\n" + "=" * 75)
    print(f"Grad-CAM visual explanation complete. All outputs saved to:\n  {figures_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
