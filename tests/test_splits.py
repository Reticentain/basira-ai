"""
tests/test_splits.py
====================
Validation suite for dataset split CSV files:
  1. Safe execution: tests that read real split files automatically pytest.skip()
     if the CSV files have not been generated yet.
  2. EyePACS split integrity:
     - Total train + val equals exactly 35,126 rows.
     - Test equals exactly 53,576 rows.
     - Zero patient_id leakage across train and val splits.
     - Both eyes of each patient remain in the same split (never divided).
  3. APTOS split integrity:
     - Train + val + test equals the deduped total (no hardcoded post-dedup count).
     - Zero image_id overlap across train, val, and test.
  4. Messidor-2 & IDRiD integrity:
     - Messidor-2 has 1,744 images.
     - IDRiD has 516 images.
"""

import pathlib
import pytest
import pandas as pd

from basira.config import load_config
from basira.data.loaders import read_split_csv
from basira.data.splits import assert_no_patient_leakage

# Search paths for split CSVs: local reports/splits or Drive splits
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LOCAL_SPLITS_DIR = _REPO_ROOT / "reports" / "splits"


def get_split_path(filename: str) -> pathlib.Path | None:
    """Find split CSV in local reports/splits or configured Drive path."""
    # Check local mirror first
    p_local = LOCAL_SPLITS_DIR / filename
    if p_local.exists():
        return p_local

    # Fallback to configured output directory
    try:
        cfg = load_config()
        p_drive = pathlib.Path(cfg.output.splits) / filename
        if p_drive.exists():
            return p_drive
    except Exception:
        pass

    return None


class TestRealDatasetSplits:
    """Tests executing against generated split CSVs (skipped if files not yet generated)."""

    @pytest.fixture(autouse=True)
    def check_splits_exist(self):
        """Skip test suite if split CSVs are not present."""
        ep_train = get_split_path("eyepacs_train.csv")
        ep_val = get_split_path("eyepacs_val.csv")
        if ep_train is None or ep_val is None:
            pytest.skip("Dataset split CSV files not generated yet. Run scripts/create_splits.py in Colab first.")

    def test_eyepacs_split_counts_and_sizes(self):
        """Hardcode only EyePACS: exactly 35,126 (train+val) and 53,576 (test)."""
        ep_train_path = get_split_path("eyepacs_train.csv")
        ep_val_path = get_split_path("eyepacs_val.csv")
        ep_test_path = get_split_path("eyepacs_test.csv")

        df_train = read_split_csv(ep_train_path)
        df_val = read_split_csv(ep_val_path)

        # 1. Train + Val row count must equal 35,126
        train_val_total = len(df_train) + len(df_val)
        assert train_val_total == 35126, (
            f"EyePACS train + val count mismatch: expected 35126, got {train_val_total}"
        )

        # 2. Test row count must equal 53,576 (if test split CSV is present)
        if ep_test_path is not None and ep_test_path.exists():
            df_test = read_split_csv(ep_test_path)
            assert len(df_test) == 53576, (
                f"EyePACS test count mismatch: expected 53576, got {len(df_test)}"
            )

    def test_eyepacs_patient_leakage_and_both_eyes_together(self):
        """
        Verify:
          (a) Zero patient_id leakage across EyePACS train, val, and test.
          (b) Every patient has all of its images in exactly one split.
        """
        ep_train_path = get_split_path("eyepacs_train.csv")
        ep_val_path = get_split_path("eyepacs_val.csv")
        ep_test_path = get_split_path("eyepacs_test.csv")

        df_train = read_split_csv(ep_train_path)
        df_val = read_split_csv(ep_val_path)

        splits_df = {"train": df_train, "val": df_val}
        if ep_test_path is not None and ep_test_path.exists():
            splits_df["test"] = read_split_csv(ep_test_path)

        splits_patients = {
            name: set(df["patient_id"].dropna().tolist())
            for name, df in splits_df.items()
        }

        # (a) No patient is in more than one of train/val/test
        split_names = list(splits_patients.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                s1, s2 = split_names[i], split_names[j]
                overlap = splits_patients[s1].intersection(splits_patients[s2])
                assert len(overlap) == 0, (
                    f"EyePACS patient leakage between '{s1}' and '{s2}': "
                    f"{len(overlap)} patients overlap: {list(overlap)[:5]}"
                )

        assert_no_patient_leakage({
            name: df["patient_id"].dropna().tolist()
            for name, df in splits_df.items()
        })

        # (b) Every patient has all of its images in exactly one split
        all_ep = pd.concat(list(splits_df.values()), ignore_index=True)
        for pid, group in all_ep.groupby("patient_id"):
            pid_str = str(pid)
            in_splits = [name for name, p_set in splits_patients.items() if pid_str in p_set]
            assert len(in_splits) == 1, (
                f"Patient {pid_str} images appear in multiple splits: {in_splits}"
            )
            # Ensure within group, all rows actually belong to that single split
            unique_sources = set(group["source"])
            assert len(unique_sources) == 1, (
                f"Patient {pid_str} images have mixed sources: {unique_sources}"
            )

    def test_aptos_split_sums_to_deduped_total(self):
        """Do not hardcode post-dedup count: test that train + val + test == deduped_total."""
        apt_train_path = get_split_path("aptos_train.csv")
        apt_val_path = get_split_path("aptos_val.csv")
        apt_test_path = get_split_path("aptos_test.csv")
        apt_dedup_path = get_split_path("aptos_deduped.csv")

        if any(p is None for p in [apt_train_path, apt_val_path, apt_test_path, apt_dedup_path]):
            pytest.skip("APTOS split CSV files missing.")

        df_train = read_split_csv(apt_train_path)
        df_val = read_split_csv(apt_val_path)
        df_test = read_split_csv(apt_test_path)
        df_dedup = read_split_csv(apt_dedup_path)

        total_split_rows = len(df_train) + len(df_val) + len(df_test)
        total_dedup_rows = len(df_dedup)

        assert total_split_rows == total_dedup_rows, (
            f"APTOS split row sum mismatch: train({len(df_train)}) + val({len(df_val)}) + "
            f"test({len(df_test)}) = {total_split_rows} != deduped total ({total_dedup_rows})"
        )

        # Zero image_id overlap across the three partitions
        train_ids = set(df_train["image_id"])
        val_ids = set(df_val["image_id"])
        test_ids = set(df_test["image_id"])

        assert len(train_ids.intersection(val_ids)) == 0, "Overlap between APTOS train and val"
        assert len(train_ids.intersection(test_ids)) == 0, "Overlap between APTOS train and test"
        assert len(val_ids.intersection(test_ids)) == 0, "Overlap between APTOS val and test"

    def test_messidor_and_idrid_external_counts(self):
        """Verify external test set sizes: Messidor-2 has 1,744 images, IDRiD has 516."""
        mess_path = get_split_path("messidor2_test.csv")
        idrid_path = get_split_path("idrid_test.csv")

        if mess_path is not None and mess_path.exists():
            df_mess = read_split_csv(mess_path)
            assert len(df_mess) == 1744, f"Messidor-2 count expected 1744, got {len(df_mess)}"

        if idrid_path is not None and idrid_path.exists():
            df_idrid = read_split_csv(idrid_path)
            assert len(df_idrid) == 516, f"IDRiD count expected 516, got {len(df_idrid)}"
