# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unit mirror of the strict-config dogfood guard (RTI Q3 / item 21).

`scripts/check_shipped_configs_strict.py` enforces that the curated
``STRICT_CLEAN`` shipped configs validate with zero unknown keys under strict
mode, and that every other shipped config is consciously classified as exempt.
This test runs that guard inside the unit suite so a regression fails fast.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GUARD_PATH = REPO_ROOT / "scripts" / "check_shipped_configs_strict.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("_check_shipped_configs_strict", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_strict_clean_configs_have_no_unknown_keys():
    guard = _load_guard()
    failures, _reported, unclassified = guard.evaluate()
    assert failures == [], "strict-clean configs regressed:\n" + "\n".join(
        f"{p}: {keys}" for p, keys in failures
    )
    assert unclassified == [], (
        "shipped config(s) neither STRICT_CLEAN nor EXEMPT — classify in "
        "scripts/check_shipped_configs_strict.py:\n" + "\n".join(unclassified)
    )


@pytest.mark.unit
def test_strict_clean_list_is_non_trivial():
    """Guard against the clean list silently shrinking to nothing."""
    guard = _load_guard()
    assert len(guard.STRICT_CLEAN) >= 2
