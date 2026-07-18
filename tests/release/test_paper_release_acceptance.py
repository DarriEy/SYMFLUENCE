# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline acceptance gate for a paper release artifact.

The release workflow installs the built wheel before running this file. Live
providers and model binaries remain covered by the full reproduction recipe
and platform installation workflows.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from symfluence.core.config.models import SymfluenceConfig

pytestmark = pytest.mark.paper_release

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "examples" / "paper_case_studies"
CONFIG_ROOT = PAPER_ROOT / "configs"


def test_installed_artifact_validates_workflow_and_figure(tmp_path: Path) -> None:
    """Validate configs, run an offline step, and render a deterministic figure."""
    config_paths = sorted(CONFIG_ROOT.rglob("*.yaml"))
    assert len(config_paths) == 185, "Update the acceptance inventory intentionally"
    for paper_config in config_paths:
        with paper_config.open() as stream:
            SymfluenceConfig.model_validate(yaml.safe_load(stream))

    # Exercise the CLI and workflow dispatcher without network or binaries.
    source = CONFIG_ROOT / "05_benchmarking" / "config_bow_benchmark.yaml"
    with source.open() as stream:
        workflow_config = yaml.safe_load(stream)
    workflow_config["system"]["workflow_steps"] = ["setup_project"]
    workflow_config["system"]["data_dir"] = str(tmp_path / "data")
    workflow_config["system"]["code_dir"] = str(REPO_ROOT)
    workflow_config["domain"]["name"] = "paper_release_acceptance"
    workflow_config["domain"]["experiment_id"] = "offline_smoke"
    config_path = tmp_path / "paper_acceptance.yaml"
    config_path.write_text(yaml.safe_dump(workflow_config, sort_keys=False))

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    for command in (
        ["config", "validate", "--config", str(config_path)],
        ["workflow", "step", "setup_project", "--config", str(config_path)],
    ):
        subprocess.run(
            [sys.executable, "-m", "symfluence.main_cli", *command],
            cwd=tmp_path,
            env=env,
            check=True,
            timeout=60,
        )
    assert (tmp_path / "data" / "domain_paper_release_acceptance").is_dir()

    # Render Fig. 8 from deterministic, schema-correct fixture inputs.
    evaluation = (
        tmp_path
        / "figure_data"
        / "domain_Bow_at_Banff_lumped_era5"
        / "evaluation"
    )
    evaluation.mkdir(parents=True)
    benchmark_names = [
        "mean_flow",
        "median_flow",
        "monthly_mean_flow",
        "monthly_median_flow",
        "daily_mean_flow",
        "daily_median_flow",
        "rainfall_runoff_ratio_to_all",
        "rainfall_runoff_ratio_to_annual",
        "rainfall_runoff_ratio_to_monthly",
        "rainfall_runoff_ratio_to_daily",
        "scaled_precipitation_benchmark",
        "adjusted_smoothed_precipitation_benchmark",
    ]
    scores = pd.DataFrame(
        {
            "kge_cal": [0.10 + i * 0.02 for i in range(len(benchmark_names))],
            "kge_val": [0.08 + i * 0.02 for i in range(len(benchmark_names))],
        },
        index=benchmark_names,
    )
    scores.to_csv(evaluation / "benchmark_scores.csv")
    output_dir = tmp_path / "figures"
    plotting_script = (
        PAPER_ROOT
        / "plotting"
        / "05_benchmarking"
        / "create_publication_figures.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(plotting_script),
            "--data-dir",
            str(tmp_path / "figure_data"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        timeout=60,
    )

    outputs = [
        output_dir / "figure_08_benchmarking.png",
        output_dir / "figure_08_benchmarking.pdf",
    ]
    assert all(path.stat().st_size > 1_000 for path in outputs)
    checksums = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs
    }
    (tmp_path / "paper_release_acceptance.json").write_text(
        json.dumps(
            {"paper_configs": len(config_paths), "figure_sha256": checksums},
            indent=2,
        )
    )
