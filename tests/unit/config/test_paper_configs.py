# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Unit tests for paper case study configuration files.

Pins reproducibility-critical properties in
examples/paper_case_studies/configs/configs_nested/* so that accidental
removal of required fields (e.g. random_seed on stochastic experiments)
is caught in CI rather than in downstream reproducibility runs.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.config.models.optimization import MultiGaugeConfig

pytestmark = [pytest.mark.unit, pytest.mark.quick]

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_NESTED = REPO_ROOT / "examples" / "paper_case_studies" / "configs" / "configs_nested"


def _load(path: Path) -> SymfluenceConfig:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    return SymfluenceConfig.model_validate(data)


def test_decision_ensemble_sets_random_seed():
    """Decision ensemble must pin a random_seed for paper reproducibility.

    PW reported best-run performance diverging from the paper (0.89 vs 0.86)
    with three different decision choices. Root cause: DDS initialized with
    unseeded RNG. This test prevents the seed from being silently dropped.
    """
    cfg = _load(CONFIGS_NESTED / "06_decision_ensemble" / "config_fuse_decisions.yaml")
    assert cfg.system.random_seed is not None, (
        "06_decision_ensemble config must set system.random_seed for "
        "reproducibility of DDS-driven decision enumeration."
    )
    assert isinstance(cfg.system.random_seed, int)


# Benchmark configs — one per model family. PW reported that 05_benchmarking
# only shipped a SUMMA config, making it impossible to reproduce Figure 8's
# other model points without copy-paste. The variants below close that gap.
BENCHMARK_CONFIGS = [
    ("config_bow_benchmark.yaml",       "SUMMA"),
    ("config_bow_benchmark_fuse.yaml",  "FUSE"),
    ("config_bow_benchmark_hbv.yaml",   "HBV"),
    ("config_bow_benchmark_gr4j.yaml",  "GR4J"),
    ("config_bow_benchmark_hype.yaml",  "HYPE"),
]


def test_multi_gauge_accepts_int_ids_and_mapping_path():
    """The large_domain (Iceland) configs supply integer LaMAH-Ice gauge IDs in
    MULTI_GAUGE_EXCLUDE_IDS and a CSV *path string* in GAUGE_SEGMENT_MAPPING —
    exactly what the calibration workers read back via flat ``config.get(...)``.

    A prior change typed exclude_ids as ``List[str]`` and gauge_segment_mapping
    as ``Dict``, which made every large_domain paper config fail typed loading
    (raising before the workers' flat access ever ran). This pins the contract:
    integer IDs and a mapping path must both validate and round-trip unchanged.
    """
    cfg = MultiGaugeConfig.model_validate({
        "MULTI_GAUGE_CALIBRATION": True,
        "MULTI_GAUGE_EXCLUDE_IDS": [13, 39, 104, 1010],  # ints, not strings
        "GAUGE_SEGMENT_MAPPING": "/data/domain_X/settings/mizuRoute/gauge_segment_mapping.csv",
    })
    assert cfg.exclude_ids == [13, 39, 104, 1010]
    assert all(isinstance(i, int) for i in cfg.exclude_ids)
    # The mapping must stay a path string (what Path(config.get(...)) consumes),
    # not be coerced into a dict.
    assert isinstance(cfg.gauge_segment_mapping, str)
    assert cfg.gauge_segment_mapping.endswith("gauge_segment_mapping.csv")
    # The explicit-dict form must still validate for callers that pass one.
    cfg_dict = MultiGaugeConfig.model_validate({"GAUGE_SEGMENT_MAPPING": {"1": 5979}})
    assert cfg_dict.gauge_segment_mapping == {"1": 5979}


@pytest.mark.parametrize("filename,expected_model", BENCHMARK_CONFIGS)
def test_benchmark_config_loads(filename, expected_model):
    """Each 05_benchmarking variant must parse, target the right model, and
    enable the benchmarking analysis. Removing the analysis or swapping the
    model field would silently invalidate the experiment."""
    cfg = _load(CONFIGS_NESTED / "05_benchmarking" / filename)
    assert str(cfg.model.hydrological_model).upper() == expected_model.upper(), (
        f"{filename} must target {expected_model}; got {cfg.model.hydrological_model}"
    )
    analyses = cfg.evaluation.analyses or []
    analyses_lower = [str(a).lower() for a in analyses]
    assert "benchmarking" in analyses_lower, (
        f"{filename} must include 'benchmarking' in evaluation.analyses; "
        f"got {analyses}"
    )
