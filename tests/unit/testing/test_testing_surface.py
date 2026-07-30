# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the public ``symfluence.testing`` surface.

This package exists so the ``symfluence-models`` extraction has a supported way
to set up a test session and isolate registries. The properties worth pinning are
therefore the *contract* ones: it imports without pytest, it does not name a
model, and the registry snapshot actually restores rather than reinstating
emptiness.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from symfluence.core.registries import R, Registries
from symfluence.testing import (
    configure_test_environment,
    make_config,
    registry_snapshot,
    scaffold_domain,
)

pytestmark = pytest.mark.unit


class _FakeRunner:
    pass


# ======================================================================
# registry_snapshot — the subtle one
# ======================================================================


def test_snapshot_restores_real_entries_rather_than_emptiness():
    """The failure mode this helper exists to prevent.

    Snapshotting before the lazy seeder is spent captures nothing, so the restore
    reinstates an empty registry — permanently, because the ``add`` decorators
    only fire on a module's first import. Assert the real registrations are back
    after the block, not that the block merely ran.
    """
    before = set(Registries.registered_models())
    assert before, 'precondition: some models must be registered to detect a loss'

    with registry_snapshot(clear=True):
        assert Registries.registered_models() == []

    assert set(Registries.registered_models()) == before


def test_snapshot_rolls_back_additions():
    with registry_snapshot(clear=True):
        R.runners.add('FAKE_SNAPSHOT_MODEL')(_FakeRunner)
        assert R.runners.get('FAKE_SNAPSHOT_MODEL') is _FakeRunner

    assert R.runners.get('FAKE_SNAPSHOT_MODEL') is None


def test_snapshot_without_clear_keeps_real_entries_visible():
    """``clear=False`` is for tests that need the real components present."""
    with registry_snapshot(clear=False):
        assert Registries.registered_models() != []
        R.runners.add('FAKE_PRESERVED_MODEL')(_FakeRunner)

    assert R.runners.get('FAKE_PRESERVED_MODEL') is None


def test_snapshot_restores_after_an_exception():
    before = set(Registries.registered_models())

    with pytest.raises(RuntimeError, match='boom'):
        with registry_snapshot(clear=True):
            R.runners.add('FAKE_RAISER')(_FakeRunner)
            raise RuntimeError('boom')

    assert set(Registries.registered_models()) == before
    assert R.runners.get('FAKE_RAISER') is None


# ======================================================================
# make_config / scaffold_domain
# ======================================================================


def test_make_config_builds_a_validated_config(tmp_path: Path):
    config = make_config(tmp_path, model='SUMMA')

    assert config.domain.name == 'test_domain'
    assert Path(config.system.data_dir) == tmp_path / 'data'
    assert Path(config.system.code_dir) == tmp_path / 'code'


def test_make_config_accepts_model_specific_keys(tmp_path: Path):
    """A model's own settings arrive through overrides, not from core.

    This is the phase-0 constraint: core must not carry per-model knowledge, so
    the helper has to take these from the caller.
    """
    config = make_config(
        tmp_path,
        model='FUSE',
        SETTINGS_FUSE_FILEMANAGER='fm_catch.txt',
        FUSE_SPATIAL_MODE='lumped',
    )
    assert config.model.hydrological_model == 'FUSE'


def test_make_config_overrides_win_over_defaults(tmp_path: Path):
    config = make_config(tmp_path, DOMAIN_NAME='override_domain')
    assert config.domain.name == 'override_domain'


def test_scaffold_domain_creates_the_tree(tmp_path: Path):
    config = make_config(tmp_path)
    paths = scaffold_domain(config, base_settings_for=[('SUMMA', 'summa')])

    assert paths['data_dir'].is_dir()
    assert paths['code_dir'].is_dir()
    assert paths['domain_dir'].is_dir()
    assert paths['domain_dir'].name == 'domain_test_domain'

    settings = (
        paths['code_dir']
        / 'src' / 'symfluence' / 'models' / 'summa' / 'base_settings'
        / 'SUMMA_settings.txt'
    )
    assert settings.is_file()


def test_scaffold_domain_is_idempotent(tmp_path: Path):
    config = make_config(tmp_path)
    first = scaffold_domain(config, base_settings_for=[('SUMMA', 'summa')])
    second = scaffold_domain(config, base_settings_for=[('SUMMA', 'summa')])
    assert first == second


# ======================================================================
# Contract properties
# ======================================================================


def test_configure_test_environment_is_idempotent():
    import os

    configure_test_environment()
    configure_test_environment()
    assert os.environ['MPLBACKEND'] == 'Agg'


def test_hdf5_setup_belongs_to_the_framework_not_this_helper():
    """Pin the division of labour, so the duplication does not creep back.

    ``symfluence/__init__.py`` calls ``configure_hdf5_safety()`` during its own
    import, before its heavy imports. Reaching ``symfluence.testing`` means
    ``symfluence`` is already imported, so those variables are set no matter what
    this helper does — and re-setting them here would fork
    ``symfluence.core.hdf5_safety``, the single authoritative definition.
    """
    import os

    from symfluence.core.hdf5_safety import HDF5_ENV_VARS

    # Set by importing symfluence, not by configure_test_environment().
    for key, expected in HDF5_ENV_VARS.items():
        assert os.environ.get(key) == expected

    environment_source = Path(
        sys.modules['symfluence.testing.environment'].__file__
    ).read_text(encoding='utf-8')
    for key in HDF5_ENV_VARS:
        assert f'"{key}"' not in environment_source, (
            f'{key} is owned by symfluence.core.hdf5_safety; '
            'setting it here would fork that definition'
        )


def test_importable_without_pytest_installed():
    """The wheel ships this package, so importing it must not require pytest.

    ``plugin`` imports pytest, which is why ``__init__`` does not import
    ``plugin``. Block pytest in a subprocess and prove the public surface still
    imports — otherwise the package would break every non-test install.
    """
    script = r'''
import sys
import importlib.abc


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pytest' or fullname.startswith('pytest.'):
            raise ImportError('blocked: pytest')
        return None


sys.meta_path.insert(0, _Blocker())

import symfluence.testing
assert symfluence.testing.registry_snapshot is not None
assert symfluence.testing.make_config is not None
assert 'pytest' not in sys.modules
print('NO-PYTEST-OK')
'''
    proc = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, (
        f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}'
    )
    assert 'NO-PYTEST-OK' in proc.stdout


def test_defaults_carry_no_per_model_settings():
    """The config defaults must not bake in any model's ``SETTINGS_*`` keys.

    A helper in core that knew ``SETTINGS_SUMMA_FILEMANAGER`` would put per-model
    knowledge straight back into core, which is exactly what phase 0 removed, and
    would mean the extracted models repo could not add a model without editing
    the framework.

    Asserted structurally against ``_DEFAULTS`` rather than by scanning the source
    text — the prose in these modules legitimately names models to explain *why*
    they are excluded, and a text scan flags its own documentation.
    """
    from symfluence.testing.config import _DEFAULTS

    settings_keys = [key for key in _DEFAULTS if key.startswith('SETTINGS_')]
    assert settings_keys == []

    # The only model-shaped value is the selector itself, supplied by the caller.
    assert 'HYDROLOGICAL_MODEL' not in _DEFAULTS


def test_snapshot_reaches_a_fixed_point_before_clearing():
    """Cross-registry declarations must not import entries in after the clear.

    Spending the lazy population one registry at a time is not enough: importing
    one registry's declared modules can append modules to a registry already
    visited, which then load inside the block. GNN and LSTM reappeared in a
    supposedly-empty registry exactly this way.
    """
    with registry_snapshot(clear=True):
        assert Registries.registered_models() == []
        # Reading a model registry must not repopulate it either.
        assert R.runners.get('SUMMA') is None
        assert Registries.registered_models() == []
