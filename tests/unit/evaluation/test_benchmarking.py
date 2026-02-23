"""Tests for benchmarking streamflow path resolution."""

import logging
import tempfile
from pathlib import Path

import pandas as pd

from symfluence.core.config.models import SymfluenceConfig
from symfluence.evaluation.benchmarking import BenchmarkPreprocessor


def _make_config(temp_root: Path) -> SymfluenceConfig:
    """Create a minimal typed config for benchmarking preprocessor tests."""
    return SymfluenceConfig.from_minimal(
        domain_name="test_domain",
        model="SUMMA",
        SYMFLUENCE_DATA_DIR=str(temp_root),
        EXPERIMENT_ID="test_exp",
        EXPERIMENT_TIME_START="2010-01-01 00:00",
        EXPERIMENT_TIME_END="2010-01-10 23:00",
    )


def _write_discharge_csv(path: Path) -> None:
    """Write unsorted hourly discharge data expected by _load_streamflow_data."""
    df = pd.DataFrame(
        {
            "datetime": [
                "2010-01-01 02:00:00",
                "2010-01-01 00:00:00",
                "2010-01-01 01:00:00",
            ],
            "discharge_cms": [2.0, 0.5, 1.0],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_load_streamflow_data_uses_processed_standard_fallback() -> None:
    """Loads from observations/streamflow/processed when preprocessed is missing."""
    with tempfile.TemporaryDirectory(prefix="sf_benchmark_tests_") as tmp_dir:
        config = _make_config(Path(tmp_dir))
        preprocessor = BenchmarkPreprocessor(config, logging.getLogger("test_benchmarking"))
        project_dir = preprocessor.project_dir

        _write_discharge_csv(
            project_dir
            / "observations"
            / "streamflow"
            / "processed"
            / "test_domain_streamflow_processed.csv"
        )

        data = preprocessor._load_streamflow_data()

        assert "streamflow" in data.columns
        assert data.index.is_monotonic_increasing
        assert data["streamflow"].tolist() == [0.5, 1.0, 2.0]


def test_load_streamflow_data_uses_grdc_fallback_when_standard_missing() -> None:
    """Falls back to GRDC naming when standard streamflow files are absent."""
    with tempfile.TemporaryDirectory(prefix="sf_benchmark_tests_") as tmp_dir:
        config = _make_config(Path(tmp_dir))
        preprocessor = BenchmarkPreprocessor(config, logging.getLogger("test_benchmarking"))
        project_dir = preprocessor.project_dir

        _write_discharge_csv(
            project_dir
            / "observations"
            / "streamflow"
            / "processed"
            / "test_domain_grdc_streamflow_processed.csv"
        )

        data = preprocessor._load_streamflow_data()

        assert "streamflow" in data.columns
        assert len(data) == 3
        assert data.index[0] == pd.Timestamp("2010-01-01 00:00:00")
