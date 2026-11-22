"""
scripts/create_splits.py
========================
Generates and exports standardized train/val/test split CSV files for:
  - EyePACS: official train (35,126) -> train (85%) / val (15%) by patient (GroupShuffleSplit).
             official test (53,576) -> test.
  - APTOS 2019: deduplicated -> train (70%) / val (15%) / test (15%) by image (stratified).
                full deduplicated set exported as aptos_deduped.csv (for EyePACS external test).
  - Messidor-2: external test only.
  - IDRiD: external test only (516 images).

CSVs store: [source, image_id, grade, referable, patient_id, eye]
Paths are NOT stored in CSVs; they are rebuilt dynamically at runtime using `rebuild_path()`.

Outputs are saved to:
  1. cfg.output.splits (Google Drive: /content/drive/MyDrive/basira/splits/)
  2. reports/splits/ (mirrored locally for git tracking)
"""

from __future__ import annotations

import pathlib
import sys

# Ensure src/ is on sys.path if run directly
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import pandas as pd
from omegaconf import DictConfig

from basira.config import load_config
from basira.data.dedup import deduplicate_aptos
from basira.data.loaders import load_aptos, load_eyepacs, load_idrid, load_messidor2
from basira.data.splits import assert_no_patient_leakage, patient_split, stratified_image_split

SPLIT_COLUMNS = ["source", "image_id", "grade", "referable", "patient_id", "eye"]


def save_split_csv(df: pd.DataFrame, filename: str, drive_dir: pathlib.Path, local_dir: pathlib.Path) -> None:
    """Save clean split DataFrame to both Drive and local reports/splits."""
    clean_df = df[SPLIT_COLUMNS].copy()

    # Save to Drive
    drive_dir.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(drive_dir / filename, index=False)

    # Mirror to local reports/splits
    local_dir.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(local_dir / filename, index=False)

    print(f"  [SAVED] {filename:<25} ({len(clean_df):>6} rows)")


def create_all_splits(cfg: DictConfig | None = None) -> None:
    """Run full splitting workflow using configurations and fixed seeds."""
    if cfg is None:
        cfg = load_config()

    seed = int(cfg.seeds.split_seed)
    drive_splits_dir = pathlib.Path(cfg.output.splits)
    local_splits_dir = _REPO_ROOT / "reports" / "splits"

    print("=" * 70)
    print("BASIRA AI — Generating Dataset Splits")
    print(f"Random seed: {seed}")
    print(f"Drive destination: {drive_splits_dir}")
    print(f"Local destination: {local_splits_dir}")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. EyePACS Splits
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/4] Processing EyePACS...")
    df_ep_train_raw = load_eyepacs(cfg, split="train")
    df_ep_test_raw = load_eyepacs(cfg, split="test")

    print(f"  EyePACS raw train loaded: {len(df_ep_train_raw)} images (expected 35126)")
    print(f"  EyePACS raw test loaded : {len(df_ep_test_raw)} images (expected 53576)")

    # Split train into train / val by patient (val_frac=0.15, test_frac=0.0)
    val_frac = float(cfg.splits.eyepacs_val_fraction)
    ep_train, ep_val, _ = patient_split(
        df_ep_train_raw,
        patient_col="patient_id",
        val_frac=val_frac,
        test_frac=0.0,
        seed=seed,
    )

    # Verify no leakage between train and val
    assert_no_patient_leakage({
        "train": ep_train["patient_id"].tolist(),
        "val": ep_val["patient_id"].tolist(),
    })

    save_split_csv(ep_train, "eyepacs_train.csv", drive_splits_dir, local_splits_dir)
    save_split_csv(ep_val, "eyepacs_val.csv", drive_splits_dir, local_splits_dir)
    save_split_csv(df_ep_test_raw, "eyepacs_test.csv", drive_splits_dir, local_splits_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. APTOS Splits (with Deduplication)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/4] Processing APTOS 2019...")
    df_aptos_raw = load_aptos(cfg)
    print(f"  APTOS raw loaded: {len(df_aptos_raw)} images")

    # Deduplicate APTOS using 256-bit dHash
    df_aptos_dedup, dedup_summary = deduplicate_aptos(df_aptos_raw, cfg=cfg)

    # Save full deduplicated set (external test benchmark for EyePACS models)
    save_split_csv(df_aptos_dedup, "aptos_deduped.csv", drive_splits_dir, local_splits_dir)

    # Split 70% train / 15% val / 15% test using stratified_image_split
    aptos_val_frac = float(cfg.splits.aptos_val_fraction)
    aptos_test_frac = float(cfg.splits.aptos_test_fraction)

    apt_train, apt_val, apt_test = stratified_image_split(
        df_aptos_dedup,
        label_col="grade",
        val_frac=aptos_val_frac,
        test_frac=aptos_test_frac,
        seed=seed,
        id_col="image_id",
    )

    save_split_csv(apt_train, "aptos_train.csv", drive_splits_dir, local_splits_dir)
    save_split_csv(apt_val, "aptos_val.csv", drive_splits_dir, local_splits_dir)
    save_split_csv(apt_test, "aptos_test.csv", drive_splits_dir, local_splits_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Messidor-2 (External Test Only)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/4] Processing Messidor-2...")
    df_messidor = load_messidor2(cfg)
    save_split_csv(df_messidor, "messidor2_test.csv", drive_splits_dir, local_splits_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. IDRiD (External Test Only)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4/4] Processing IDRiD...")
    df_idrid = load_idrid(cfg)
    save_split_csv(df_idrid, "idrid_test.csv", drive_splits_dir, local_splits_dir)

    print("\n" + "=" * 70)
    print("All splits generated and validated successfully.")
    print("=" * 70)


if __name__ == "__main__":
    create_all_splits()
