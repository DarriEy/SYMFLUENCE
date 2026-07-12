#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""
Analyze parallel scaling study results for Section 4.13.

Computes speedup, efficiency, and scaling metrics from timing CSV files.
Generates summary tables and JSON for downstream visualization.

Usage:
    python analyze_scaling.py                    # Analyze all available results
    python analyze_scaling.py --source local     # Local results only
    python analyze_scaling.py --source hpc       # HPC results only
    python analyze_scaling.py --experiment 1     # TauDEM only
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# timing_*_latest.csv come from the scaling-run harness (see the
# 08_parallel_scaling configs/README); drop them into results/ next to this
# script before running.
_HERE = Path(__file__).resolve()
RESULTS_DIR = _HERE.parent / "results"
ANALYSIS_DIR = _HERE.parent / "analysis"


def load_csv(filepath: Path) -> List[Dict[str, Any]]:
    """Load a CSV file into list of dicts, coercing numeric types."""
    if not filepath.exists():
        return []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v is None or v == "":
                    parsed[k] = None
                elif v.lower() in ("true", "false"):
                    parsed[k] = v.lower() == "true"
                else:
                    try:
                        parsed[k] = int(v)
                    except ValueError:
                        try:
                            parsed[k] = float(v)
                        except ValueError:
                            parsed[k] = v
                rows.append(parsed)
        return rows


def compute_strong_scaling(
    records: List[Dict], time_key: str = "wall_clock_seconds",
    np_key: str = "num_processes", group_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Compute strong scaling metrics: speedup, efficiency, overhead.

    Groups by group_key (e.g., resolution or strategy) if provided.
    """
    results = []

    # Group records
    groups = {}
    for r in records:
        if not r.get("success", True):
            continue
        g = r.get(group_key, "all") if group_key else "all"
        groups.setdefault(g, []).append(r)

    for group_name, group_records in groups.items():
        # Average times per NP
        np_times = {}
        for r in group_records:
            np_val = r[np_key]
            np_times.setdefault(np_val, []).append(r[time_key])

        np_avg = {np: sum(t) / len(t) for np, t in np_times.items()}

        # Baseline: single processor
        baseline_np = min(np_avg.keys())
        t1 = np_avg[baseline_np]

        for np_val in sorted(np_avg.keys()):
            t_n = np_avg[np_val]
            speedup = t1 / t_n if t_n > 0 else 0
            efficiency = speedup / (np_val / baseline_np) if np_val > 0 else 0
            overhead = (t_n * np_val) - (t1 * baseline_np)

            results.append({
                "group": group_name,
                "num_processes": np_val,
                "wall_clock_seconds": round(t_n, 2),
                "speedup": round(speedup, 3),
                "efficiency": round(efficiency, 3),
                "overhead_seconds": round(overhead, 2),
                "baseline_seconds": round(t1, 2),
                "baseline_np": baseline_np,
                "n_runs": len(np_times[np_val]),
            })

    return results


def analyze_taudem(source: str = "all") -> Dict[str, Any]:
    """Analyze TauDEM delineation scaling results."""
    results = {"local": [], "hpc": []}

    # Local results
    if source in ("all", "local"):
        local_data = load_csv(RESULTS_DIR / "timing_taudem_latest.csv")
        if local_data:
            results["local"] = compute_strong_scaling(
                local_data, group_key="resolution"
            )

    return results


def analyze_calibration(source: str = "all") -> Dict[str, Any]:
    """Analyze SUMMA calibration scaling results."""
    results = {"local": [], "hpc": []}

    # Local results
    if source in ("all", "local"):
        local_data = load_csv(RESULTS_DIR / "timing_calibration_latest.csv")
        if local_data:
            results["local"] = compute_strong_scaling(
                local_data, group_key="strategy"
            )

    # HPC results - loaded from pre-computed JSON (Fir data entered manually)
    if source in ("all", "hpc"):
        json_path = ANALYSIS_DIR / "scaling_calibration.json"
        if json_path.exists():
            with open(json_path) as f:
                existing = json.load(f)
            if existing.get("hpc"):
                results["hpc"] = existing["hpc"]

    return results


def save_analysis(data: Dict, name: str) -> None:
    """Save analysis results to JSON and CSV."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = ANALYSIS_DIR / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {json_path.name}")

    # Flatten to CSV for each sub-key
    for key, records in data.items():
        if records and isinstance(records, list):
            csv_path = ANALYSIS_DIR / f"{name}_{key}.csv"
            fieldnames = sorted(set().union(*(r.keys() for r in records)))
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in records:
                    writer.writerow(r)
            print(f"  Saved: {csv_path.name}")


def print_scaling_table(records: List[Dict], title: str) -> None:
    """Print a formatted scaling metrics table."""
    if not records:
        print(f"\n  {title}: No data available")
        return

    print(f"\n  {title}")
    print(f"  {'Group':<15} {'NP':>5} {'Time(s)':>10} {'Speedup':>8} {'Efficiency':>10}")
    print(f"  {'-'*52}")
    for r in records:
        print(
            f"  {r['group']:<15} {r['num_processes']:>5} "
            f"{r['wall_clock_seconds']:>10.1f} {r['speedup']:>8.2f} "
            f"{r['efficiency']:>10.1%}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Section 4.13 parallel scaling results"
    )
    parser.add_argument(
        "--source", choices=["all", "local", "hpc"], default="all",
        help="Data source to analyze"
    )
    parser.add_argument(
        "--experiment", type=int, nargs="*", default=[1, 2],
        help="Experiments to analyze: 1=TauDEM, 2=Calibration"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Section 4.13 Parallel Scaling Analysis")
    print("=" * 60)

    if 1 in args.experiment:
        print("\n--- Experiment 1: TauDEM Delineation ---")
        taudem = analyze_taudem(args.source)
        save_analysis(taudem, "scaling_taudem")
        print_scaling_table(taudem.get("local", []), "Iceland (Local)")

    if 2 in args.experiment:
        print("\n--- Experiment 2: SUMMA Calibration ---")
        calib = analyze_calibration(args.source)
        save_analysis(calib, "scaling_calibration")
        print_scaling_table(calib.get("local", []), "Calibration Local (PP + MPI)")
        print_scaling_table(calib.get("hpc", []), "Calibration HPC (DDS + DE)")

    print(f"\n{'=' * 60}")
    print(f"  Analysis outputs in: {ANALYSIS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
