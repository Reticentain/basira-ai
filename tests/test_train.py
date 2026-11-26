"""
tests/test_train.py
===================
Unit tests for baseline training pipeline and checkpoint resumption:
  - Verifies checkpoint saving and loading with full real Python/NumPy/Torch RNG states
  - Catches PyTorch 2.6+ weights_only=True UnpicklingError regression
  - Verifies early-stopping counter (epochs_no_improve) and best metrics are saved and restored on resume
  - Runs completely on CPU using synthetic images without requiring a GPU
"""

from __future__ import annotations

import json
import os
import pathlib
import random
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
    load_checkpoint,
    restore_rng_states,
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
    def test_train_epoch_and_resume_with_real_rng_states(self, synthetic_train_env):
        """
        Train 1 epoch, save last.pt with real Python/NumPy/Torch RNG states,
        and verify load_checkpoint successfully unpickles without PyTorch 2.6+ UnpicklingError.
        """
        cfg, df_train, df_val, ckpt_root = synthetic_train_env
        device = torch.device("cpu")

        train_ds = CachedFundusDataset(df_train, cfg=cfg, train=True, task="referable")
        val_ds = CachedFundusDataset(df_val, cfg=cfg, train=False, task="referable")

        train_loader = DataLoader(train_ds, batch_size=2, shuffle=False, drop_last=True)
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

        # Capture REAL RNG states including NumPy RNG state tuple (which triggers weights_only=True UnpicklingError)
        real_rng_states = {
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": None,
        }

        state_epoch1 = {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": None,
            "best_metric": val_metrics["val_auc"],
            "best_epoch": 1,
            "best_threshold": val_metrics["val_threshold"],
            "epochs_no_improve": 0,
            "rng_states": real_rng_states,
            "config": {},
            "task": "referable",
        }
        save_checkpoint_atomic(state_epoch1, last_pt)

        assert last_pt.exists()

        # Must load cleanly via load_checkpoint (weights_only=False, map_location='cpu')
        loaded_e1 = load_checkpoint(last_pt, map_location="cpu")
        assert loaded_e1["epoch"] == 1
        assert loaded_e1["epochs_no_improve"] == 0
        assert "rng_states" in loaded_e1
        # Restore RNG states through restore_rng_states
        restore_rng_states(loaded_e1["rng_states"], device=device)

        # --- Simulate Resume for Epoch 2 ---
        resume_model = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=1)
        resume_opt = torch.optim.AdamW(resume_model.parameters(), lr=1e-4)
        resume_sched = torch.optim.lr_scheduler.CosineAnnealingLR(resume_opt, T_max=2)

        ckpt = load_checkpoint(last_pt, map_location="cpu")
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
            "scaler_state_dict": None,
            "best_metric": val_metrics_e2["val_auc"],
            "best_epoch": 2,
            "best_threshold": val_metrics_e2["val_threshold"],
            "epochs_no_improve": 0,
            "rng_states": {
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": None,
            },
            "config": {},
            "task": "referable",
        }
        save_checkpoint_atomic(state_epoch2, last_pt)

        # Checkpoint is now updated to epoch 2
        loaded_e2 = load_checkpoint(last_pt, map_location="cpu")
        assert loaded_e2["epoch"] == 2

    def test_early_stopping_counter_and_best_metric_restored_on_resume(self, synthetic_train_env):
        """
        Verify that early stopping counter (epochs_no_improve), best_metric,
        and best_threshold are preserved in checkpoints and accurately restored on resume.
        """
        cfg, df_train, df_val, ckpt_root = synthetic_train_env
        device = torch.device("cpu")

        run_dir = ckpt_root / "early_stop_test_run"
        run_dir.mkdir(parents=True)
        last_pt = run_dir / "last.pt"

        model = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=1)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

        # Simulate checkpoint at epoch 4 where best was at epoch 2 (AUC=0.88), so epochs_no_improve = 2
        simulated_state = {
            "epoch": 4,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "scaler_state_dict": None,
            "best_metric": 0.88,
            "best_epoch": 2,
            "best_threshold": 0.42,
            "epochs_no_improve": 2,  # 2 epochs without improvement
            "rng_states": {
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": None,
            },
            "config": {},
            "task": "referable",
        }
        save_checkpoint_atomic(simulated_state, last_pt)

        # Restore on resume
        ckpt = load_checkpoint(last_pt, map_location="cpu")

        start_epoch = int(ckpt["epoch"]) + 1
        restored_best_auc = float(ckpt["best_metric"])
        restored_best_epoch = int(ckpt["best_epoch"])
        restored_best_thresh = float(ckpt["best_threshold"])
        restored_epochs_no_improve = int(ckpt["epochs_no_improve"])

        assert start_epoch == 5
        assert restored_best_auc == 0.88
        assert restored_best_epoch == 2
        assert restored_best_thresh == 0.42
        assert restored_epochs_no_improve == 2, (
            f"Expected restored epochs_no_improve=2, got {restored_epochs_no_improve}"
        )

        # Verify that if patience is 5, reaching 3 more non-improving epochs hits patience
        patience = 5
        for sim_epoch in range(start_epoch, start_epoch + 3):
            # Model does not improve
            restored_epochs_no_improve += 1

        assert restored_epochs_no_improve == patience

    def test_restore_rng_states_works_on_cpu_tensors(self):
        """restore_rng_states must properly restore Python, NumPy, and PyTorch CPU states."""
        device = torch.device("cpu")
        rng = {
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": None,
        }
        restore_rng_states(rng, device=device)
        assert torch.get_rng_state() is not None

    def test_checkpoint_loading_returns_no_cuda_tensors(self, synthetic_train_env):
        """load_checkpoint with default map_location='cpu' returns all tensors on CPU."""
        cfg, df_train, df_val, ckpt_root = synthetic_train_env
        run_dir = ckpt_root / "cpu_tensor_check"
        run_dir.mkdir(parents=True)
        ckpt_path = run_dir / "last.pt"

        state = {
            "epoch": 1,
            "model_state_dict": {"dummy_weight": torch.randn(3, 3)},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "scaler_state_dict": None,
            "best_metric": 0.5,
            "best_epoch": 1,
            "best_threshold": 0.5,
            "epochs_no_improve": 0,
            "rng_states": {
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": None,
            },
            "config": {},
            "task": "referable",
        }
        save_checkpoint_atomic(state, ckpt_path)

        loaded = load_checkpoint(ckpt_path)

        # Check model state dict tensors are on CPU
        for k, v in loaded["model_state_dict"].items():
            if isinstance(v, torch.Tensor):
                assert v.device.type == "cpu", f"Tensor {k} is on {v.device}, expected CPU"

        # Check torch_rng tensor is on CPU
        torch_rng = loaded["rng_states"]["torch_rng"]
        if isinstance(torch_rng, torch.Tensor):
            assert torch_rng.device.type == "cpu", f"torch_rng is on {torch_rng.device}, expected CPU"

    def test_restore_rng_states_with_cuda_tensors(self):
        """
        If CUDA is available, verify that RNG states stored as CUDA tensors
        are safely converted to CPU ByteTensor without raising TypeError.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA runtime not available on this machine.")

        device = torch.device("cuda")

        # Simulate checkpoint where torch_rng was moved to GPU by map_location=device
        torch_rng_cuda = torch.get_rng_state().cuda()
        cuda_rng_states = [x.cuda() for x in torch.cuda.get_rng_state_all()]

        rng_with_cuda_tensors = {
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch_rng_cuda,
            "cuda_rng": cuda_rng_states,
        }

        # Must NOT raise: "TypeError: RNG state must be a torch.ByteTensor"
        restore_rng_states(rng_with_cuda_tensors, device=device)
