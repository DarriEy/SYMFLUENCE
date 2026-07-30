# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Pytest fixtures for repositories that test against SYMFLUENCE.

Opt in from a ``conftest.py``::

    pytest_plugins = ["symfluence.testing.plugin"]

Deliberately NOT registered as a ``pytest11`` entry point. An auto-loading plugin
would activate its autouse fixtures in every environment that merely has
SYMFLUENCE installed, including unrelated test suites — the consuming repository
should decide.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

from symfluence.testing.config import make_config, scaffold_domain
from symfluence.testing.environment import configure_test_environment
from symfluence.testing.registries import registry_snapshot

__all__ = [
    "symfluence_environment",
    "isolate_npm_bundle",
    "isolated_registries",
    "preserved_registries",
    "symfluence_config",
    "symfluence_domain",
]


@pytest.fixture(scope="session", autouse=True)
def symfluence_environment() -> None:
    """Safety net for the process-level native-library setup.

    ``configure_test_environment()`` should be called at the top of the root
    conftest, because by the time fixtures run, numpy may already be imported and
    the HDF5 locking variables will have been read. This fixture cannot fix that
    ordering — it exists so a suite that forgot the call still gets the settings
    that are order-independent, and it is idempotent when the call was made.
    """
    configure_test_environment()


@pytest.fixture(autouse=True)
def isolate_npm_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep executable resolution away from a real ``npm install -g symfluence``.

    ``get_model_executable`` falls back to the npm bundle when a source install is
    absent, and ``npm_bundle_bin()`` locates it via ``npm root -g`` unless
    ``SYMFLUENCE_NPM_DIST_BIN`` is set. On a developer machine that has the npm
    package, a test asserting "this executable is missing" resolves a real binary
    instead — passing in CI and failing locally.

    Pointing the override at a non-existent path makes ``npm_bundle_bin()`` return
    None (its documented contract for a non-directory override) and spares every
    test the ``npm root -g`` subprocess. A test that wants to exercise the
    fallback sets the variable in its own body, which runs after this fixture and
    therefore wins.
    """
    monkeypatch.setenv("SYMFLUENCE_NPM_DIST_BIN", str(tmp_path / "no-npm-bundle"))


@pytest.fixture
def isolated_registries() -> Iterator[None]:
    """Empty registries for the test, restored exactly afterwards.

    Use when registering fakes: the test sees only what it registers. See
    :func:`symfluence.testing.registries.registry_snapshot` for why the lazy
    population has to be spent before snapshotting.
    """
    with registry_snapshot(clear=True):
        yield


@pytest.fixture
def preserved_registries() -> Iterator[None]:
    """Keep the real registrations, but roll back anything the test adds."""
    with registry_snapshot(clear=False):
        yield


@pytest.fixture
def symfluence_config(tmp_path: Path) -> Callable[..., Any]:
    """Factory for a validated config rooted in this test's ``tmp_path``.

    Example:
        >>> def test_thing(symfluence_config):
        ...     cfg = symfluence_config(model='FUSE', FUSE_SPATIAL_MODE='lumped')
    """

    def _build(**kwargs: Any) -> Any:
        return make_config(tmp_path, **kwargs)

    return _build


@pytest.fixture
def symfluence_domain(symfluence_config: Callable[..., Any]) -> Callable[..., Any]:
    """Factory returning ``(config, paths)`` with the domain tree created on disk.

    ``base_settings_for`` is forwarded to
    :func:`symfluence.testing.config.scaffold_domain`; everything else builds the
    config.
    """

    def _build(*, base_settings_for: Any = (), **kwargs: Any) -> Any:
        config = symfluence_config(**kwargs)
        paths = scaffold_domain(config, base_settings_for=base_settings_for)
        return config, paths

    return _build
