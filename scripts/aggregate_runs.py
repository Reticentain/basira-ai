"""
scripts/aggregate_runs.py
=========================
Aggregate evaluation results across multiple training runs (e.g. seeds).

Reads each run's `results.json` from Drive reports/<run-name>/ and computes
mean +/- SD across runs for:
  - AUC
  - Sensitivity at the frozen operating threshold
  - Specificity at the frozen operating threshold
  - Oracle Sensitivity (tuned directly on test set at target specificity)

Outputs:
  - Drive reports/aggregate/aggregate_results.json
  - Drive reports/aggregate/aggregate_table.md

Usage:
    python scripts/aggregate_runs.py --runs b0_referable_s42 b0_referable_s1 b0_referable_s2
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, Optional

import numpy as np

# Repository path bootstrap
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from basira.config import apply_overrides, load_config


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation metrics across multiple runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="List of run names to aggregate (e.g. b0_referable_s42 b0_referable_s1).",
    )
    parser.add_argument(
        "--reports-root",
        type=str,
        default=None,
        help="Root directory containing run report folders (default: from config output.reports).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for aggregate reports (default: <reports-root>/aggregate).",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        dest="set_overrides",
        default=[],
        metavar="KEY=VALUE",
        help="Override configuration keys (e.g. --set output.reports=/path/to/reports).",
    )
    return parser.parse_args(argv)


def locate_run_results(run_name: str, reports_root: pathlib.Path) -> pathlib.Path:
    """Find results.json for a given run name across standard search paths."""
    candidates = [
        reports_root / run_name / "results.json",
        reports_root / "reports" / run_name / "results.json",
        _REPO_ROOT / "reports" / run_name / "results.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find 'results.json' for run '{run_name}'. Looked in:\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )


def load_run_results(run_name: str, reports_root: pathlib.Path) -> dict[str, Any]:
    """Load and parse results.json for a given run."""
    path = locate_run_results(run_name, reports_root)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_metric_value(metric_obj: Any) -> float:
    """Extract float value from scalar or metric dictionary (e.g. {'val': 0.95})."""
    if isinstance(metric_obj, dict):
        if "val" in metric_obj:
            return float(metric_obj["val"])
        if "value" in metric_obj:
            return float(metric_obj["value"])
    return float(metric_obj)


def extract_oracle_sensitivity(dataset_data: dict[str, Any]) -> float:
    """Extract oracle sensitivity across possible field names."""
    # Standard field from evaluate.py
    for k in (
        "oracle (tuned on this set, not deployable)",
        "oracle",
        "oracle_upper_bound",
    ):
        if k in dataset_data and isinstance(dataset_data[k], dict):
            if "sensitivity" in dataset_data[k]:
                return float(dataset_data[k]["sensitivity"])

    if "oracle_sensitivity" in dataset_data:
        return float(dataset_data["oracle_sensitivity"])

    return float("nan")


def aggregate_runs_data(
    runs: list[str],
    runs_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute mean and sample standard deviation across runs per dataset.

    Parameters
    ----------
    runs : list of str
        List of run names.
    runs_results : list of dict
        Parsed results.json contents for each run in matching order.

    Returns
    -------
    dict
        Consolidated aggregate statistics dictionary.
    """
    if len(runs) != len(runs_results):
        raise ValueError("Length mismatch between runs list and runs_results.")

    # Collect all dataset names in order of appearance
    canonical_order = ["eyepacs_test", "aptos_deduped", "messidor2_test", "idrid_test"]
    discovered_datasets: list[str] = []

    for res in runs_results:
        dsets = res.get("datasets", {})
        for dname in dsets:
            if dname not in discovered_datasets:
                discovered_datasets.append(dname)

    # Sort so canonical datasets appear first in defined order
    all_datasets = [d for d in canonical_order if d in discovered_datasets]
    all_datasets += [d for d in discovered_datasets if d not in all_datasets]

    agg_datasets: dict[str, Any] = {}

    for dname in all_datasets:
        auc_vals: list[float] = []
        sens_vals: list[float] = []
        spec_vals: list[float] = []
        oracle_vals: list[float] = []

        for res in runs_results:
            d_dict = res.get("datasets", {}).get(dname)
            if not d_dict:
                continue

            if "auc" in d_dict:
                auc_vals.append(extract_metric_value(d_dict["auc"]))
            if "sensitivity" in d_dict:
                sens_vals.append(extract_metric_value(d_dict["sensitivity"]))
            if "specificity" in d_dict:
                spec_vals.append(extract_metric_value(d_dict["specificity"]))

            orc_sens = extract_oracle_sensitivity(d_dict)
            if not np.isnan(orc_sens):
                oracle_vals.append(orc_sens)

        def calc_stats(values: list[float]) -> dict[str, Any]:
            if not values:
                return {"mean": float("nan"), "sd": float("nan"), "n": 0, "values": []}
            arr = np.array(values, dtype=np.float64)
            mean_val = float(np.mean(arr))
            sd_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            return {
                "mean": mean_val,
                "sd": sd_val,
                "n": len(values),
                "values": [float(v) for v in values],
            }

        agg_datasets[dname] = {
            "n_runs_present": len(auc_vals),
            "auc": calc_stats(auc_vals),
            "sensitivity": calc_stats(sens_vals),
            "specificity": calc_stats(spec_vals),
            "oracle_sensitivity": calc_stats(oracle_vals),
        }

    return {
        "runs": runs,
        "n_runs": len(runs),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": agg_datasets,
    }


def generate_aggregate_markdown(agg_data: dict[str, Any]) -> str:
    """Generate publication-ready Markdown table summarizing aggregate performance."""
    runs = agg_data.get("runs", [])
    n_runs = agg_data.get("n_runs", len(runs))
    datasets = agg_data.get("datasets", {})

    lines = [
        "# Multi-Run Aggregate Evaluation Summary",
        "",
        f"**Runs Evaluated ($N={n_runs}$):**",
    ]
    for r in runs:
        lines.append(f"- `{r}`")
    lines.extend([
        "",
        "| Dataset | AUC (Mean ± SD) | Sensitivity @ Frozen Thresh (Mean ± SD) | Specificity @ Frozen Thresh (Mean ± SD) | Oracle Sensitivity (Mean ± SD) |",
        "|:---|:---:|:---:|:---:|:---:|",
    ])

    for dname, metrics in datasets.items():
        def fmt_stat(stat: dict[str, Any]) -> str:
            m = stat.get("mean", float("nan"))
            s = stat.get("sd", float("nan"))
            if np.isnan(m):
                return "N/A"
            return f"{m:.4f} ± {s:.4f}"

        auc_str = fmt_stat(metrics["auc"])
        sens_str = fmt_stat(metrics["sensitivity"])
        spec_str = fmt_stat(metrics["specificity"])
        oracle_str = fmt_stat(metrics["oracle_sensitivity"])

        lines.append(f"| `{dname}` | {auc_str} | {sens_str} | {spec_str} | {oracle_str} |")

    lines.extend([
        "",
        "### Notes & Methodology",
        f"- **Aggregation Statistics:** Metrics reported as Sample Mean ± Sample Standard Deviation ($ddof=1$) across $N={n_runs}$ independent runs.",
        "- **Validation-Only Calibration:** Sensitivity and Specificity are evaluated at the frozen operating threshold determined solely on `eyepacs_val`.",
        "- **External Benchmark Independence:** No operating threshold or Platt calibration parameters were tuned on test datasets.",
        "- **Oracle Benchmark:** Oracle Sensitivity represents the non-deployable theoretical upper bound obtained by tuning directly at 90% specificity on that exact evaluation dataset.",
    ])
    return "\n".join(lines) + "\n"


def save_atomic(content: str, filepath: pathlib.Path) -> None:
    """Atomically write text content to file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp = filepath.parent / f".tmp_{os.getpid()}_{filepath.name}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, filepath)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_config()
    if args.set_overrides:
        apply_overrides(cfg, args.set_overrides)

    reports_root = (
        pathlib.Path(args.reports_root)
        if args.reports_root
        else pathlib.Path(cfg.output.reports)
    )
    output_dir = (
        pathlib.Path(args.output_dir)
        if args.output_dir
        else reports_root / "aggregate"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f"Basira AI — Multi-Run Aggregation ({len(args.runs)} runs)")
    print(f"Reports Root : {reports_root}")
    print(f"Output Dir   : {output_dir}")
    print("=" * 75)

    # 1. Load results.json for each run
    runs_results: list[dict[str, Any]] = []
    for r in args.runs:
        r_clean = r.strip()
        print(f"Loading results for run: '{r_clean}'...")
        res = load_run_results(r_clean, reports_root)
        runs_results.append(res)
        print(f"  ✓ Loaded: {r_clean}")

    # 2. Aggregate metrics across runs
    print("\nComputing cross-run aggregate statistics (Mean ± SD)...")
    agg_data = aggregate_runs_data(args.runs, runs_results)

    # 3. Save JSON and Markdown artifacts atomically
    json_path = output_dir / "aggregate_results.json"
    json_content = json.dumps(agg_data, indent=2)
    save_atomic(json_content, json_path)
    print(f"  ✓ Saved aggregate JSON to: {json_path}")

    md_path = output_dir / "aggregate_table.md"
    md_content = generate_aggregate_markdown(agg_data)
    save_atomic(md_content, md_path)
    print(f"  ✓ Saved aggregate Markdown table to: {md_path}")

    # 4. Print summary table to console
    print("\n" + "=" * 75)
    print(md_content)
    print("=" * 75)
    print("Aggregation completed successfully.")


if __name__ == "__main__":
    main()
