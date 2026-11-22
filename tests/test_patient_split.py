"""
tests/test_patient_split.py
===========================
Tests that the patient-level and image-level splitting utilities enforce
data integrity and split requirements:

  (a) Both eyes of one patient in the same split does NOT raise.
  (b) A patient in two splits DOES raise.
  (c) patient_split with test_frac=0 returns an empty test set and no overlap.
  (d) stratified_image_split keeps grade proportions close across splits and is
      deterministic for a fixed seed.

Runs on synthetic data only — no real images needed.
"""

import numpy as np
import pandas as pd
import pytest

from basira.data.splits import (
    aptos_patient_id,
    assert_no_patient_leakage,
    eyepacs_patient_id,
    patient_split,
    stratified_image_split,
)


# ---------------------------------------------------------------------------
# Unit tests for patient-ID extractors
# ---------------------------------------------------------------------------

class TestPatientIdExtractors:
    def test_eyepacs_left_eye(self):
        assert eyepacs_patient_id("1234_left.jpeg") == "1234"

    def test_eyepacs_right_eye(self):
        assert eyepacs_patient_id("5678_right.jpeg") == "5678"

    def test_eyepacs_no_extension(self):
        assert eyepacs_patient_id("42_left") == "42"

    def test_eyepacs_large_id(self):
        assert eyepacs_patient_id("99999_right.jpg") == "99999"

    def test_aptos_returns_code_unchanged(self):
        assert aptos_patient_id("0a4e1a29fc2d") == "0a4e1a29fc2d"


# ---------------------------------------------------------------------------
# Tests for leakage detection (Requirements a & b)
# ---------------------------------------------------------------------------

class TestAssertNoPatientLeakage:
    def test_clean_splits_pass(self):
        """No overlap between distinct patients across splits."""
        splits = {
            "train": ["p1", "p2", "p3"],
            "val":   ["p4", "p5"],
            "test":  ["p6"],
        }
        assert_no_patient_leakage(splits)

    def test_both_eyes_in_same_split_does_not_raise(self):
        """(a) Both eyes of one patient in the same split does NOT raise."""
        splits = {
            "train": ["p1", "p1", "p2", "p2", "p3"],
            "val":   ["p4", "p4", "p5"],
            "test":  ["p6", "p6"],
        }
        # Repeating the same patient ID within one partition is expected (left + right eye)
        assert_no_patient_leakage(splits)

    def test_patient_in_two_splits_raises(self):
        """(b) A patient in two splits DOES raise."""
        splits = {
            "train": ["p1", "p2", "p3"],
            "val":   ["p2", "p4"],  # p2 is in both train and val
            "test":  ["p5"],
        }
        with pytest.raises(AssertionError, match="Patient-level split leakage"):
            assert_no_patient_leakage(splits)

    def test_patient_in_train_and_test_raises(self):
        """(b) A patient in train and test splits DOES raise."""
        splits = {
            "train": ["p1", "p2"],
            "val":   ["p3"],
            "test":  ["p1"],  # p1 is in both train and test
        }
        with pytest.raises(AssertionError, match="Patient-level split leakage"):
            assert_no_patient_leakage(splits)


# ---------------------------------------------------------------------------
# Tests for patient_split (Requirement c)
# ---------------------------------------------------------------------------

class TestPatientSplit:
    @pytest.fixture()
    def synthetic_df(self):
        """50 synthetic patients, 2 eyes each (left/right), 5 DR grades."""
        rng = np.random.default_rng(42)
        records = []
        for i in range(50):
            pid = f"p{i}"
            for side in ("left", "right"):
                records.append({
                    "filename":   f"{i}_{side}.jpg",
                    "patient_id": pid,
                    "grade":      int(rng.integers(0, 5)),
                })
        return pd.DataFrame(records)

    def test_no_leakage_standard_split(self, synthetic_df):
        train, val, test = patient_split(
            synthetic_df,
            patient_col="patient_id",
            val_frac=0.15,
            test_frac=0.15,
            seed=42,
        )
        assert_no_patient_leakage({
            "train": train["patient_id"].tolist(),
            "val":   val["patient_id"].tolist(),
            "test":  test["patient_id"].tolist(),
        })

    def test_patient_split_test_frac_zero(self, synthetic_df):
        """(c) patient_split with test_frac=0 returns an empty test set and no overlap."""
        train, val, test = patient_split(
            synthetic_df,
            patient_col="patient_id",
            val_frac=0.20,
            test_frac=0.0,
            seed=42,
        )

        # Test set must be empty DataFrame
        assert len(test) == 0, "Expected test set to be empty when test_frac=0"
        assert isinstance(test, pd.DataFrame)

        # All rows must be accounted for between train and val
        assert len(train) + len(val) == len(synthetic_df)

        # No patient ID leakage between train and val
        assert_no_patient_leakage({
            "train": train["patient_id"].tolist(),
            "val":   val["patient_id"].tolist(),
        })

        # Ensure train and val have expected relative patient counts
        train_patients = train["patient_id"].nunique()
        val_patients = val["patient_id"].nunique()
        assert train_patients + val_patients == synthetic_df["patient_id"].nunique()
        assert val_patients == 10  # 20% of 50 patients


# ---------------------------------------------------------------------------
# Tests for stratified_image_split (Requirement d)
# ---------------------------------------------------------------------------

class TestStratifiedImageSplit:
    @pytest.fixture()
    def imbalanced_df(self):
        """1000 images with imbalanced DR grades (grades 0-4)."""
        rng = np.random.default_rng(42)
        grades = [0] * 500 + [1] * 200 + [2] * 150 + [3] * 100 + [4] * 50
        rng.shuffle(grades)
        return pd.DataFrame({
            "image_id": [f"img_{i:04d}" for i in range(len(grades))],
            "grade":    grades,
        })

    def test_stratified_image_split_determinism(self, imbalanced_df):
        """(d) stratified_image_split is deterministic for a fixed seed."""
        train1, val1, test1 = stratified_image_split(
            imbalanced_df, label_col="grade", val_frac=0.15, test_frac=0.15, seed=42, id_col="image_id"
        )
        train2, val2, test2 = stratified_image_split(
            imbalanced_df, label_col="grade", val_frac=0.15, test_frac=0.15, seed=42, id_col="image_id"
        )

        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_stratified_image_split_proportions(self, imbalanced_df):
        """(d) stratified_image_split keeps grade proportions close across splits."""
        train, val, test = stratified_image_split(
            imbalanced_df, label_col="grade", val_frac=0.15, test_frac=0.15, seed=42, id_col="image_id"
        )

        overall_props = imbalanced_df["grade"].value_counts(normalize=True).sort_index()

        for split_df, split_name in [(train, "train"), (val, "val"), (test, "test")]:
            split_props = split_df["grade"].value_counts(normalize=True).sort_index()
            # Proportions across each grade should match overall within 2% margin
            diff = (overall_props - split_props).abs().max()
            assert diff < 0.02, f"Proportion mismatch in {split_name}: max diff {diff:.4f} >= 0.02"

        # Verify total rows
        assert len(train) + len(val) + len(test) == len(imbalanced_df)
