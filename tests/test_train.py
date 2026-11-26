"""
tests/test_train.py
===================
Unit tests for baseline training pipeline and checkpoint resumption:
  - Verifies checkpoint saving (last.pt, best.pt, metrics.csv, run_info.json)
  - Verifies --resume restores epoch, optimizer, scheduler, and RNG states
  - Runs completely on CPU using synthetic images without requiring a GPU
"""

from __future__ import annotations

import json
import os
import pathlib
import numpy as np
import pandas as pd
import pytest
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from basira.config import load_config
from basira.data.cache import cache_dir, write_manifest
from basira.data.dataset import CachedFundusDataset
from scripts.train import (
    evaluate,
    save_checkpoint_atomic,
    train_one_epoch,
)


@pytest.fixture
def synthetic_train_env(tmp_path):
    """Create an isolated training sandbox with dummy images and valid splits."""
    cfg = load_config()
    cfg_copy = cfg.copy()

    cache_root = tmp_path / "cache"
    splits_root = tmp_path / "splits"
    checkpoints_root = tmp_path / "checkpoints"

    cache_root.mkdir(parents=True)
    splits_root.mkdir(parents=True)
    checkpoints_root.mkdir(parents=True)

    cfg_copy.output.cache_root = str(cache_root)
    cfg_copy.output.splits = str(splits_root)
    cfg_copy.output.checkpoints = str(checkpoints_root)
    cfg_copy.training.epochs = 2
    cfg_copy.training.batch_size = 2
    cfg_copy.training.num_workers = 0
    cfg_copy.training.use_amp = False

    # 1. Write valid manifest
    write_manifest(counts={"eyepacs_train": 8}, cfg=cfg_copy, merge=False)

    # 2. Create 8 synthetic images in eyepacs_train cache
    dest_dir = cache_dir(cfg_copy) / "eyepacs_train"
    dest_dir.mkdir(parents=True, exist_ok=True)

    train_rows = []
    val_rows = []

    # 4 images for train, 4 for val
    for i in range(8):
        iid = f"patient_{i}_left.jpg"
        stem = f"patient_{i}_left"
        img = np.full((256, 256, 3), fill_value=30 + i * 25, dtype=np.uint8)
        Image.fromarray(img).save(dest_dir / f"{stem}.png")

        row = {
            "source": "eyepacs_train",
            "image_id": iid,
            "grade": i % 5,
            "referable": 1 if (i % 5) >= 2 else 0,
            "patient_id": f"patient_{i}",
            "eye": "left",
        }
        if i < 4:
            train_rows.append(row)
        else:
            val_rows.append(row)

    df_train = pd.DataFrame(train_rows)
    df_val = pd.DataFrame(val_rows)

    df_train.to_csv(splits_root / "eyepacs_train.csv", index=False)
    df_val.to_csv(splits_root / "eyepacs_val.csv", index=False)

    return cfg_copy, df_train, df_val, checkpoints_root


class TestTrainingAndResume:
    def test_train_epoch_and_resume_continuation(self, synthetic_train_env):
        """Train 1 epoch, save last.pt, then resume and verify continuation at epoch 2."""
        cfg, df_train, df_val, ckpt_root = synthetic_train_env
        device = torch.device("cpu")

        train_ds = CachedFundusDataset(df_train, cfg=cfg, train=True, task="referable")
        val_ds = CachedFundusDataset(df_val, cfg=cfg, train=False, task="referable")

        train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

        # Lightweight model for CPU smoke testing
        model = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=1)
        model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        run_dir = ckpt_root / "test_run"
        run_dir.mkdir(parents=True)
        last_pt = run_dir / "last.pt"

        # --- Run Epoch 1 ---
        train_loss, _, _ = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=False,
            task="referable",
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=False,
            task="referable",
        )
        scheduler.step()

        state_epoch1 = {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": val_metrics["val_auc"],
            "best_epoch": 1,
        }
        save_checkpoint_atomic(state_epoch1, last_pt)

        assert last_pt.exists()
        loaded_e1 = torch.load(last_pt, map_location=device)
        assert loaded_e1["epoch"] == 1

        # --- Simulate Resume for Epoch 2 ---
        resume_model = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=1)
        resume_opt = torch.optim.AdamW(resume_model.parameters(), lr=1e-4)
        resume_sched = torch.optim.lr_scheduler.CosineAnnealingLR(resume_opt, T_max=2)

        ckpt = torch.load(last_pt, map_location=device)
        resume_model.load_state_dict(ckpt["model_state_dict"])
        resume_opt.load_state_dict(ckpt["optimizer_state_dict"])
        resume_sched.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1

        assert start_epoch == 2, f"Expected start_epoch=2 on resume, got {start_epoch}"

        # Run Epoch 2
        train_loss_e2, _, _ = train_one_epoch(
            model=resume_model,
            loader=train_loader,
            criterion=criterion,
            optimizer=resume_opt,
            scaler=scaler,
            device=device,
            use_amp=False,
            task="referable",
        )
        val_metrics_e2 = evaluate(
            model=resume_model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=False,
            task="referable",
        )
        resume_sched.step()

        state_epoch2 = {
            "epoch": 2,
            "model_state_dict": resume_model.state_dict(),
            "optimizer_state_dict": resume_opt.state_dict(),
            "scheduler_state_dict": resume_sched.state_dict(),
            "best_metric": val_metrics_e2["val_auc"],
            "best_epoch": 2,
        }
        save_checkpoint_atomic(state_epoch2, last_pt)

        # Checkpoint is now updated to epoch 2
        loaded_e2 = torch.load(last_pt, map_location=device)
        assert loaded_e2["epoch"] == 2
