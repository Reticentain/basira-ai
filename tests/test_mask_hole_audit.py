"""tests/test_mask_hole_audit.py – unit tests for scripts/mask_hole_audit.py.

No GPU, no real cache, no network required.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap (ensure repo root and src/ are importable)
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from scripts.mask_hole_audit import (  # noqa: E402
    HOLE_THRESHOLDS,
    _atomic_write_csv,
    _atomic_write_json,
    _atomic_write_text,
    _fn_fp_rates,
    build_markdown,
    compute_hole_fraction,
    parse_args,
)


# ---------------------------------------------------------------------------
# Helpers to synthesise images
# ---------------------------------------------------------------------------


def _make_solid_disc(size: int = 64, radius: int | None = None) -> np.ndarray:
    """Return a uint8 HxWx3 image with a solid white disc, black background."""
    if radius is None:
        radius = size // 2 - 2
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cy, cx = size // 2, size // 2
    yy, xx = np.ogrid[:size, :size]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    img[mask] = 200
    return img


def _punch_hole(img: np.ndarray, hole_radius: int) -> np.ndarray:
    """Return a copy of *img* with an interior circular hole zeroed out."""
    img = img.copy()
    size = img.shape[0]
    cy, cx = size // 2, size // 2
    yy, xx = np.ogrid[:size, :size]
    hole_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= hole_radius ** 2
    img[hole_mask] = 0
    return img


# ---------------------------------------------------------------------------
# Tests for compute_hole_fraction
# ---------------------------------------------------------------------------


class TestComputeHoleFraction:
    def test_solid_disc_no_holes(self):
        img = _make_solid_disc(64)
        frac, hole_px, filled_px = compute_hole_fraction(img)
        assert frac == pytest.approx(0.0, abs=1e-9), "Solid disc should have zero holes"
        assert hole_px == 0

    def test_all_black_returns_zero(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        frac, hole_px, filled_px = compute_hole_fraction(img)
        assert frac == 0.0
        assert hole_px == 0
        assert filled_px == 0

    def test_disc_with_punched_hole(self):
        """A punched interior hole should give a non-zero fraction."""
        disc_radius = 28
        hole_radius = 8
        img = _make_solid_disc(64, radius=disc_radius)
        img_holed = _punch_hole(img, hole_radius=hole_radius)

        frac, hole_px, filled_px = compute_hole_fraction(img_holed)

        # The hole area ≈ π * hole_radius²
        expected_hole_area = np.pi * hole_radius ** 2
        assert hole_px == pytest.approx(expected_hole_area, rel=0.05), (
            f"Hole pixels {hole_px} should ≈ π*{hole_radius}²={expected_hole_area:.0f}"
        )
        assert frac > 0.0, "Hole fraction must be positive"
        # filled_disc area ≈ π * disc_radius²
        expected_disc = np.pi * disc_radius ** 2
        assert filled_px == pytest.approx(expected_disc, rel=0.05)

    def test_hole_fraction_monotone_with_hole_size(self):
        """Larger interior holes → monotonically increasing hole fraction."""
        disc = _make_solid_disc(128, radius=56)
        fracs = []
        for hr in [0, 4, 8, 12, 16, 20]:
            if hr == 0:
                img = disc
            else:
                img = _punch_hole(disc, hole_radius=hr)
            frac, _, _ = compute_hole_fraction(img)
            fracs.append(frac)

        for i in range(1, len(fracs)):
            assert fracs[i] >= fracs[i - 1], (
                f"Hole fraction not monotone: fracs={fracs}"
            )

    def test_all_white_image_no_holes(self):
        """Image with no background at all — the filled disc IS the whole image."""
        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        frac, hole_px, _ = compute_hole_fraction(img)
        assert frac == pytest.approx(0.0, abs=1e-9)
        assert hole_px == 0

    def test_fractional_value_range(self):
        """Hole fraction must always be in [0, 1]."""
        img = _make_solid_disc(64, radius=28)
        for hr in [0, 5, 10, 15, 20, 25]:
            holed = _punch_hole(img, hole_radius=hr)
            frac, _, _ = compute_hole_fraction(holed)
            assert 0.0 <= frac <= 1.0, f"Hole fraction {frac} out of [0,1] for hr={hr}"


# ---------------------------------------------------------------------------
# Tests for _fn_fp_rates
# ---------------------------------------------------------------------------


class TestFnFpRates:
    def _make_df(
        self,
        grades: list[int],
        probs: list[float],
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "grade": grades,
                "calibrated_prob": probs,
            }
        )
        df["is_referable"] = (df["grade"] >= 2).astype(int)
        df["predicted_referable"] = (df["calibrated_prob"] >= threshold).astype(int)
        return df

    def test_perfect_classifier(self):
        grades = [0, 0, 2, 3]
        probs = [0.1, 0.1, 0.9, 0.9]
        df = self._make_df(grades, probs)
        result = _fn_fp_rates(df)
        assert result["fn_rate"] == pytest.approx(0.0)
        assert result["fp_rate"] == pytest.approx(0.0)

    def test_all_fn(self):
        """All referables missed → fn_rate == 1.0."""
        grades = [2, 3]
        probs = [0.1, 0.1]
        df = self._make_df(grades, probs)
        result = _fn_fp_rates(df)
        assert result["fn_rate"] == pytest.approx(1.0)
        assert result["fp_rate"] is None  # no non-referables

    def test_empty_subset(self):
        df = pd.DataFrame(
            columns=["grade", "calibrated_prob", "is_referable", "predicted_referable"]
        ).astype(
            {
                "grade": int,
                "calibrated_prob": float,
                "is_referable": int,
                "predicted_referable": int,
            }
        )
        result = _fn_fp_rates(df)
        assert result["n"] == 0
        assert result["fn_rate"] is None
        assert result["fp_rate"] is None

    def test_mixed_errors(self):
        # 4 referables: 2 TP, 2 FN; 4 non-ref: 1 FP, 3 TN
        grades = [2, 2, 2, 2, 0, 0, 0, 0]
        probs = [0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]
        df = self._make_df(grades, probs)
        r = _fn_fp_rates(df)
        assert r["fn_rate"] == pytest.approx(0.5)
        assert r["fp_rate"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Tests for build_markdown
# ---------------------------------------------------------------------------


class TestBuildMarkdown:
    def _make_result(self, dname: str = "eyepacs_test") -> dict:
        return {
            "dataset": dname,
            "n_total": 100,
            "n_skipped": 2,
            "mean_hole_fraction": 0.03,
            "median_hole_fraction": 0.01,
            "n_hole_gt1pct": 20,
            "n_hole_gt5pct": 5,
            "n_hole_gt10pct": 1,
            "fn_fp_by_hole_threshold": {
                "gt1pct": {
                    "has_holes": {"n": 20, "fn_rate": 0.3, "fp_rate": 0.1},
                    "no_holes": {"n": 80, "fn_rate": 0.1, "fp_rate": 0.05},
                },
                "gt5pct": {
                    "has_holes": {"n": 5, "fn_rate": 0.4, "fp_rate": 0.2},
                    "no_holes": {"n": 95, "fn_rate": 0.1, "fp_rate": 0.05},
                },
                "gt10pct": {
                    "has_holes": {"n": 1, "fn_rate": None, "fp_rate": None},
                    "no_holes": {"n": 99, "fn_rate": 0.1, "fp_rate": 0.05},
                },
            },
        }

    def test_markdown_contains_dataset_name(self):
        md = build_markdown([self._make_result("eyepacs_test")], 0.2785)
        assert "eyepacs_test" in md

    def test_markdown_contains_header(self):
        md = build_markdown([self._make_result()], 0.2785)
        assert "# Mask Interior-Hole Audit" in md

    def test_markdown_contains_threshold(self):
        md = build_markdown([self._make_result()], 0.2785)
        assert "0.2785" in md

    def test_markdown_tables_present(self):
        md = build_markdown([self._make_result()], 0.5)
        assert "| Dataset |" in md
        assert "FN rate" in md

    def test_markdown_read_only_note(self):
        md = build_markdown([self._make_result()], 0.5)
        assert "Read-only" in md or "read-only" in md

    def test_none_fractions_shown_as_dash(self):
        result = self._make_result()
        result["mean_hole_fraction"] = None
        result["median_hole_fraction"] = None
        md = build_markdown([result], 0.5)
        assert "—" in md


# ---------------------------------------------------------------------------
# Tests for parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_required_run_name(self):
        args = parse_args(["--run-name", "b0_referable_s42"])
        assert args.run_name == "b0_referable_s42"

    def test_default_datasets(self):
        args = parse_args(["--run-name", "myrun"])
        assert "eyepacs_val" in args.datasets
        assert "eyepacs_test" in args.datasets
        assert len(args.datasets) == 5

    def test_custom_datasets(self):
        args = parse_args(["--run-name", "myrun", "--datasets", "aptos_deduped", "idrid_test"])
        assert args.datasets == ["aptos_deduped", "idrid_test"]

    def test_set_overrides(self):
        args = parse_args(["--run-name", "myrun", "--set", "paths.cache_dir=/tmp/cache"])
        assert args.overrides == ["paths.cache_dir=/tmp/cache"]

    def test_help_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Tests for atomic write helpers
# ---------------------------------------------------------------------------


class TestAtomicWriteHelpers:
    def test_atomic_write_text(self, tmp_path):
        p = tmp_path / "test.md"
        _atomic_write_text(p, "hello world")
        assert p.read_text(encoding="utf-8") == "hello world"

    def test_atomic_write_text_overwrites(self, tmp_path):
        p = tmp_path / "test.md"
        _atomic_write_text(p, "first")
        _atomic_write_text(p, "second")
        assert p.read_text(encoding="utf-8") == "second"

    def test_atomic_write_json(self, tmp_path):
        p = tmp_path / "data.json"
        data = {"key": "value", "n": 42}
        _atomic_write_json(p, data)
        loaded = json.loads(p.read_text(encoding="utf-8"))
        assert loaded == data

    def test_atomic_write_csv(self, tmp_path):
        p = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        _atomic_write_csv(df, p)
        loaded = pd.read_csv(p)
        pd.testing.assert_frame_equal(df, loaded)

    def test_no_tmp_file_left(self, tmp_path):
        p = tmp_path / "clean.md"
        _atomic_write_text(p, "content")
        tmp_files = list(tmp_path.glob(".tmp_*"))
        assert len(tmp_files) == 0, f"Leftover tmp files: {tmp_files}"


# ---------------------------------------------------------------------------
# Integration-style test: synthetic audit of a small fake dataset
# ---------------------------------------------------------------------------


class TestSyntheticAudit:
    """End-to-end test of the hole audit pipeline using tiny fake images."""

    def _write_fake_png(self, path: pathlib.Path, img: np.ndarray) -> None:
        from PIL import Image as PILImage

        path.parent.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(img).save(path)

    def test_synthetic_audit_produces_correct_summary(self, tmp_path, monkeypatch):
        """Create 4 fake images (2 referables, 2 non-referables), 2 with holes,
        verify the audit produces sensible FN/FP counts."""
        from unittest.mock import MagicMock

        from scripts.mask_hole_audit import audit_dataset

        cache_dir = tmp_path / "cache"
        report_dir = tmp_path / "reports" / "testrun"
        report_dir.mkdir(parents=True)

        # Build 4 fake images and write them
        disc_no_hole = _make_solid_disc(64, radius=28)
        disc_with_hole = _punch_hole(disc_no_hole, hole_radius=10)

        images = {
            "img1": disc_no_hole,  # grade 0, no hole, prob 0.1 (TN)
            "img2": disc_with_hole,  # grade 0, hole,    prob 0.9 (FP)
            "img3": disc_no_hole,  # grade 2, no hole, prob 0.9 (TP)
            "img4": disc_with_hole,  # grade 2, hole,    prob 0.1 (FN)
        }
        for img_id, arr in images.items():
            self._write_fake_png(cache_dir / "fake_ds" / f"{img_id}.png", arr)

        # Predictions CSV
        preds = pd.DataFrame(
            {
                "image_id": ["img1", "img2", "img3", "img4"],
                "source": ["fake_ds"] * 4,
                "patient_id": ["p1", "p2", "p3", "p4"],
                "grade": [0, 0, 2, 2],
                "logit": [0.0, 2.0, 2.0, -2.0],
                "prob": [0.1, 0.9, 0.9, 0.1],
                "calibrated_prob": [0.1, 0.9, 0.9, 0.1],
            }
        )
        pred_csv = report_dir / "fake_ds_predictions.csv"
        preds.to_csv(pred_csv, index=False)

        # Mock cache_path and resolve_cache_source to return deterministic paths
        def _mock_cache_path(source, img_id, cfg):
            return str(cache_dir / source / f"{img_id}.png")

        def _mock_resolve(raw_src):
            return raw_src

        monkeypatch.setattr("scripts.mask_hole_audit.cache_path", _mock_cache_path)
        monkeypatch.setattr("scripts.mask_hole_audit.resolve_cache_source", _mock_resolve)

        cfg = MagicMock()

        summary = audit_dataset(
            dataset_name="fake_ds",
            predictions_csv=pred_csv,
            cfg=cfg,
            operating_threshold=0.5,
            out_dir=report_dir / "mask_hole_audit",
        )

        assert summary["n_total"] == 4
        assert summary["n_skipped"] == 0

        # img2 and img4 have holes → n_hole_gt1pct should be 2
        assert summary["n_hole_gt1pct"] == 2
        assert summary["n_hole_gt5pct"] == 2  # holes are ~10% disc area

        # Verify FN/FP at the gt1pct threshold:
        # has_holes: img2 (FP), img4 (FN)
        # no_holes: img1 (TN), img3 (TP)
        fn_fp = summary["fn_fp_by_hole_threshold"]["gt1pct"]
        has = fn_fp["has_holes"]
        no = fn_fp["no_holes"]

        assert has["fn"] == 1   # img4 missed
        assert has["fp"] == 1   # img2 false alarm
        assert no["fn"] == 0    # img3 caught
        assert no["fp"] == 0    # img1 correctly negative

        # Check CSV was written
        audit_csv = report_dir / "mask_hole_audit" / "fake_ds_hole_audit.csv"
        assert audit_csv.exists()
        audit_df = pd.read_csv(audit_csv)
        assert len(audit_df) == 4
        assert "hole_fraction" in audit_df.columns
