"""
scripts/train.py
================
Baseline training pipeline for Diabetic Retinopathy screening models.

Key features:
  - Timm backbones (efficientnet_b0, mobilenetv3_large_100)
  - Tasks: 'referable' (binary DR>=2 with pos_weight on train) or 'grade' (0-4 multiclass)
  - Evaluates on EyePACS validation set every epoch:
      * Validation Loss
      * ROC-AUC
      * Sensitivity at 90% Specificity (threshold calibrated strictly on val)
      * Expected Calibration Error (ECE)
      * Throughput (images/sec) and epoch duration
  - Atomic checkpoints: last.pt every epoch, best.pt by val AUC
  - Automatic resume (--resume) restoring model, optimizer, scheduler, scaler, RNG states
  - Early stopping with configurable patience (default 5)
  - run_info.json recording best metrics, selected threshold, active config, and git commit
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Ensure repo root and src/ are on sys.path
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from basira.config import apply_overrides, load_config
from basira.data.cache import get_current_git_commit
from basira.data.dataset import CachedFundusDataset
from basira.data.loaders import read_split_csv
from basira.data.splits import assert_no_patient_leakage
from basira.eval.metrics import (
    compute_auc,
    expected_calibration_error,
    quadratic_weighted_kappa,
    sensitivity_at_specificity,
)


def seed_everything(seed: int) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader worker processes deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
        + "\nPlease run scripts/create_splits.py first."
    )


def save_checkpoint_atomic(state: dict[str, Any], filepath: pathlib.Path) -> None:
    """Atomically save checkpoint using a temporary file and atomic replace."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.parent / f".tmp_{os.getpid()}_{filepath.name}"
    torch.save(state, tmp_path)
    os.replace(tmp_path, filepath)


def load_checkpoint(filepath: pathlib.Path | str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """
    Load a checkpoint containing model weights, optimizer/scheduler states, and RNG states.

    Always loads with map_location="cpu" by default to ensure RNG state tensors remain on CPU
    (preventing TypeError: RNG state must be a torch.ByteTensor on CUDA runtimes).
    weights_only=False is explicitly specified because our checkpoints contain full
    Python dicts and NumPy RNG states (np.random.get_state()) required for deterministic
    resumption. These checkpoints are authored and saved exclusively by Basira AI on
    our own Google Drive / local storage.
    """
    return torch.load(filepath, map_location=map_location, weights_only=False)


def restore_rng_states(rng: dict[str, Any], device: torch.device) -> None:
    """
    Restore Python, NumPy, and PyTorch RNG states safely across CPU and CUDA.

    PyTorch requires torch.set_rng_state and torch.cuda.set_rng_state_all to receive
    CPU ByteTensors (torch.uint8). This converts torch_rng and each entry of cuda_rng
    with torch.as_tensor(x, dtype=torch.uint8).cpu() before restoring.
    """
    if not rng:
        return

    if "python_rng" in rng and rng["python_rng"] is not None:
        random.setstate(rng["python_rng"])

    if "numpy_rng" in rng and rng["numpy_rng"] is not None:
        np.random.set_state(rng["numpy_rng"])

    if "torch_rng" in rng and rng["torch_rng"] is not None:
        torch_state = torch.as_tensor(rng["torch_rng"], dtype=torch.uint8).cpu()
        torch.set_rng_state(torch_state)

    if device.type == "cuda" and torch.cuda.is_available() and rng.get("cuda_rng") is not None:
        cuda_states = [torch.as_tensor(x, dtype=torch.uint8).cpu() for x in rng["cuda_rng"]]
        torch.cuda.set_rng_state_all(cuda_states)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    task: str,
    max_batches: Optional[int] = None,
) -> tuple[float, float, float]:
    """
    Train for a single epoch.

    Returns
    -------
    avg_loss : float
    elapsed_time : float (seconds)
    images_per_sec : float
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            if task == "referable":
                loss = criterion(logits.view(-1), targets.view(-1))
            else:
                loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    elapsed = max(1e-5, time.time() - start_time)
    avg_loss = running_loss / max(1, total_samples)
    img_per_sec = total_samples / elapsed
    return avg_loss, elapsed, img_per_sec


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    task: str,
    target_specificity: float = 0.90,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """
    Evaluate model on validation set.

    Returns dictionary containing loss, AUC, sensitivity @ 90% spec, threshold, ECE, QWK.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_targets: list[float] = []
    all_probs: list[float] = []
    all_preds: list[int] = []

    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            if task == "referable":
                loss = criterion(logits.view(-1), targets.view(-1))
                probs = torch.sigmoid(logits.view(-1))
            else:
                loss = criterion(logits, targets)
                probs = torch.softmax(logits, dim=1)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

        if task == "referable":
            all_targets.extend(targets.view(-1).cpu().numpy().tolist())
            all_probs.extend(probs.view(-1).cpu().numpy().tolist())
        else:
            all_targets.extend(targets.view(-1).cpu().numpy().tolist())
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.view(-1).cpu().numpy().tolist())
            # For AUC on grade task, use prob of referable (grade >= 2)
            prob_ref = torch.sum(probs[:, 2:], dim=1)
            all_probs.extend(prob_ref.view(-1).cpu().numpy().tolist())

    val_loss = running_loss / max(1, total_samples)
    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)

    # Compute binary metrics on referable target (grade >= 2)
    y_true_binary = (y_true >= 2).astype(int) if task == "grade" else y_true.astype(int)
    auc = compute_auc(y_true_binary, y_prob)
    sens, thresh = sensitivity_at_specificity(y_true_binary, y_prob, target_specificity=target_specificity)
    ece = expected_calibration_error(y_true_binary, y_prob, n_bins=10)

    qwk = float("nan")
    if task == "grade" and all_preds:
        qwk = quadratic_weighted_kappa(y_true, np.array(all_preds), num_classes=5)

    return {
        "val_loss": float(val_loss),
        "val_auc": float(auc),
        "val_sens_at_90spec": float(sens),
        "val_threshold": float(thresh),
        "val_ece": float(ece),
        "val_qwk": float(qwk),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basira AI Baseline Training")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this run / checkpoint dir.")
    parser.add_argument("--task", type=str, choices=["referable", "grade"], default="referable")
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet_b0",
        help="Model backbone: 'efficientnet_b0', 'mobilenetv3', or timm identifier.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="Config overrides as key=value pairs (e.g. training.epochs=2).",
    )
    parser.add_argument("--subset", type=int, default=None, help="Train/val subset size for smoke tests.")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches per epoch for smoke tests.")
    parser.add_argument("--resume", action="store_true", help="Auto-resume from last.pt in run checkpoint dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()

    # Apply CLI key=value overrides to config
    if args.set:
        apply_overrides(cfg, args.set)

    # Resolve seed
    seed = args.seed if args.seed is not None else int(cfg.seeds.global_seed)
    seed_everything(seed)

    # Resolve run name
    model_short = "mobilenetv3" if "mobilenet" in args.model.lower() else "efficientnet_b0"
    if args.run_name is None:
        run_name = f"{model_short}_{args.task}_seed{seed}"
    else:
        run_name = args.run_name

    ckpt_dir = pathlib.Path(cfg.output.checkpoints) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BASIRA AI — Training Baseline")
    print(f"Run name        : {run_name}")
    print(f"Model backbone  : {args.model}")
    print(f"Task            : {args.task}")
    print(f"Random seed     : {seed}")
    print(f"Checkpoint dir  : {ckpt_dir}")
    print(f"Device available: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    # 1. Load splits (EyePACS train and EyePACS val)
    df_train = read_split_csv(find_split_file("eyepacs_train.csv", cfg))
    df_val = read_split_csv(find_split_file("eyepacs_val.csv", cfg))

    # Assert no patient leakage between train and val
    assert_no_patient_leakage({
        "train": df_train["patient_id"].tolist(),
        "val": df_val["patient_id"].tolist(),
    })

    if args.subset is not None:
        print(f"Subsetting data for smoke testing: {args.subset} train images...")
        df_train = df_train.sample(min(args.subset, len(df_train)), random_state=seed).reset_index(drop=True)
        val_sub = max(10, args.subset // 4)
        df_val = df_val.sample(min(val_sub, len(df_val)), random_state=seed).reset_index(drop=True)

    print(f"Dataset split sizes: Train={len(df_train)}, Val={len(df_val)}")

    # 2. Build Datasets
    train_dataset = CachedFundusDataset(df_train, cfg=cfg, train=True, task=args.task)
    val_dataset = CachedFundusDataset(df_val, cfg=cfg, train=False, task=args.task)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = int(cfg.training.num_workers)
    batch_size = int(cfg.training.batch_size)

    gen = torch.Generator()
    gen.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=gen,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # 3. Model setup via timm
    timm_name = "mobilenetv3_large_100" if args.model == "mobilenetv3" else args.model
    num_classes = 1 if args.task == "referable" else 5
    model = timm.create_model(timm_name, pretrained=bool(cfg.model.pretrained), num_classes=num_classes)
    model.to(device)

    # 4. Criterion with train-only pos_weight
    if args.task == "referable":
        train_grades = df_train["grade"].values
        n_pos = int(np.sum(train_grades >= 2))
        n_neg = int(np.sum(train_grades < 2))
        pos_weight_val = (n_neg / max(1, n_pos)) if n_pos > 0 else 1.0
        pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print(f"Referable class balance (train only): pos={n_pos}, neg={n_neg}, pos_weight={pos_weight_val:.4f}")
    else:
        criterion = nn.CrossEntropyLoss()

    # 5. Optimizer & Scheduler
    lr = float(cfg.training.learning_rate)
    wd = float(cfg.training.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    epochs = int(cfg.training.epochs)
    sched_type = str(cfg.training.get("scheduler", "cosine")).lower()
    if sched_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = bool(cfg.training.use_amp) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_val_auc = -1.0
    best_threshold = 0.5
    best_epoch = 0
    patience = int(cfg.training.get("early_stopping_patience", 5))
    epochs_no_improve = 0

    # 6. Auto-resume from last.pt if requested
    last_pt_path = ckpt_dir / "last.pt"
    if args.resume and last_pt_path.exists():
        print(f"Loading checkpoint for resume: {last_pt_path}")
        ckpt = load_checkpoint(last_pt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if use_amp and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_auc = float(ckpt.get("best_metric", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        best_threshold = float(ckpt.get("best_threshold", 0.5))
        epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))

        # Restore RNG states safely across CPU and CUDA
        if "rng_states" in ckpt and ckpt["rng_states"] is not None:
            restore_rng_states(ckpt["rng_states"], device=device)

        print(
            f"Resumed from epoch {start_epoch - 1}. Current best val AUC: {best_val_auc:.4f} "
            f"(epoch {best_epoch}), early stopping counter: {epochs_no_improve}/{patience}"
        )

    metrics_csv_path = ckpt_dir / "metrics.csv"
    if not metrics_csv_path.exists():
        with open(metrics_csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,val_auc,val_sens_at_90spec,val_threshold,val_ece,val_qwk,lr,epoch_time_sec,img_per_sec\n")

    # 7. Training Loop
    print("\nStarting training loop...")
    for epoch in range(start_epoch, epochs + 1):
        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"\n--- Epoch {epoch}/{epochs} (lr: {curr_lr:.2e}) ---")

        train_loss, epoch_time, img_per_sec = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            task=args.task,
            max_batches=args.max_batches,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            task=args.task,
            target_specificity=float(cfg.evaluation.target_specificity),
            max_batches=args.max_batches,
        )

        val_auc = val_metrics["val_auc"]
        val_loss = val_metrics["val_loss"]
        val_sens = val_metrics["val_sens_at_90spec"]
        val_thresh = val_metrics["val_threshold"]
        val_ece = val_metrics["val_ece"]
        val_qwk = val_metrics["val_qwk"]

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_auc if not np.isnan(val_auc) else -val_loss)
        else:
            scheduler.step()

        print(
            f"Epoch {epoch:02d} Summary: "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val AUC={val_auc:.4f} | "
            f"Sens@90%Spec={val_sens:.4f} (thresh={val_thresh:.4f}) | "
            f"ECE={val_ece:.4f} | "
            f"{img_per_sec:.1f} img/s ({epoch_time:.1f}s)"
        )

        # Append to metrics.csv
        with open(metrics_csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_auc:.6f},"
                f"{val_sens:.6f},{val_thresh:.6f},{val_ece:.6f},{val_qwk:.6f},"
                f"{curr_lr:.6e},{epoch_time:.2f},{img_per_sec:.2f}\n"
            )

        # Check if validation AUC improved
        improved = False
        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_threshold = val_thresh
            best_epoch = epoch
            epochs_no_improve = 0
            improved = True
        else:
            epochs_no_improve += 1

        # Checkpoint state containing all training, model, early-stopping, and RNG states
        rng_states = {
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if (device.type == "cuda" and torch.cuda.is_available()) else None,
        }

        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "best_metric": best_val_auc,
            "best_epoch": best_epoch,
            "best_threshold": best_threshold,
            "epochs_no_improve": epochs_no_improve,
            "rng_states": rng_states,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "task": args.task,
        }

        # Atomically save last.pt every epoch
        save_checkpoint_atomic(ckpt_state, last_pt_path)

        # Save best.pt and run_info.json if AUC improved
        if improved:
            save_checkpoint_atomic(ckpt_state, ckpt_dir / "best.pt")
            print(f"  ✓ New best model saved! (Val AUC: {best_val_auc:.4f} at epoch {epoch})")

            # Update run_info.json
            run_info = {
                "run_name": run_name,
                "git_commit": get_current_git_commit(),
                "task": args.task,
                "model": timm_name,
                "best_epoch": best_epoch,
                "best_val_auc": float(best_val_auc),
                "chosen_val_threshold": float(best_threshold),
                "epochs_no_improve": epochs_no_improve,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            tmp_info = ckpt_dir / f".tmp_{os.getpid()}_run_info.json"
            with open(tmp_info, "w", encoding="utf-8") as f:
                json.dump(run_info, f, indent=2)
            os.replace(tmp_info, ckpt_dir / "run_info.json")
        else:
            if epochs_no_improve >= patience:
                print(f"\n[EARLY STOPPING] Validation AUC did not improve for {patience} consecutive epochs. Stopping.")
                break

    print("\n" + "=" * 70)
    print(f"Training completed. Best epoch: {best_epoch} with Val AUC: {best_val_auc:.4f}")
    print(f"Checkpoints and metrics saved in: {ckpt_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
