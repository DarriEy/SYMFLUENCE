# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Property-based tests over the configuration model (review item 17).

Hypothesis exercises invariants of the flat<->nested transform and the
stage-marker hash that are easy to regress when the config schema changes.
"""

from __future__ import annotations

import string

from hypothesis import given, settings
from hypothesis import strategies as st

from symfluence.core.config.flattening import flatten_nested_config
from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.stage_marker import compute_config_hash

from .conftest import build_minimal_config

_NAME = st.text(alphabet=string.ascii_letters + string.digits + "_-", min_size=1, max_size=24)
_HASH_SECTIONS = ["domain", "forcing", "model", "system"]


@settings(max_examples=50, deadline=None)
@given(domain_name=_NAME, num_processes=st.integers(min_value=1, max_value=64))
def test_flatten_round_trip_preserves_key_fields(domain_name, num_processes):
    cfg = build_minimal_config(domain_name=domain_name, NUM_PROCESSES=num_processes)
    flat = flatten_nested_config(cfg)
    cfg2 = SymfluenceConfig(**flat)
    assert cfg2.domain.name == cfg.domain.name
    assert cfg2.model.hydrological_model == cfg.model.hydrological_model
    assert cfg2.system.num_processes == cfg.system.num_processes


@settings(max_examples=50, deadline=None)
@given(domain_name=_NAME)
def test_hash_is_deterministic_and_order_independent(domain_name):
    cfg = build_minimal_config(domain_name=domain_name)
    h1 = compute_config_hash(cfg, _HASH_SECTIONS)
    h2 = compute_config_hash(cfg, list(reversed(_HASH_SECTIONS)))
    assert h1 == h2
    assert len(h1) == 64 and all(c in string.hexdigits for c in h1)


@settings(max_examples=50, deadline=None)
@given(name_a=_NAME, name_b=_NAME)
def test_hash_is_sensitive_to_changes(name_a, name_b):
    cfg_a = build_minimal_config(domain_name=name_a)
    cfg_b = build_minimal_config(domain_name=name_b)
    same = compute_config_hash(cfg_a, ["domain"]) == compute_config_hash(cfg_b, ["domain"])
    assert same == (name_a == name_b)


def test_flatten_prefers_canonical_key_over_legacy_alias():
    """NUM_PROCESSES is the canonical flat key; the legacy MPI_PROCESSES is not emitted."""
    cfg = build_minimal_config(NUM_PROCESSES=8)
    flat = flatten_nested_config(cfg)
    assert flat.get("NUM_PROCESSES") == 8
    assert "MPI_PROCESSES" not in flat
