# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Config and on-disk domain builders for tests.

Deliberately model-agnostic. ``make_config`` knows the framework-level keys every
domain needs (data/code dirs, domain name, experiment window, forcing cadence)
and nothing about any particular model — a model's own test suite passes its
``SETTINGS_<MODEL>_*`` keys through ``**overrides``.

That restriction is the point. Phase 0 moved per-model knowledge out of core; a
test helper in core that hardcoded ``SETTINGS_SUMMA_FILEMANAGER`` would put it
straight back, and would mean the extracted ``symfluence-models`` repo could not
add a model without editing the framework.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from symfluence.core.config.models import SymfluenceConfig

__all__ = ["make_config", "scaffold_domain"]

#: Framework-level defaults. Nothing here names a model beyond the required
#: ``HYDROLOGICAL_MODEL`` selector, whose value the caller chooses.
_DEFAULTS: Dict[str, Any] = {
    "DOMAIN_NAME": "test_domain",
    "EXPERIMENT_ID": "test_run",
    "EXPERIMENT_TIME_START": "2020-01-01 00:00",
    "EXPERIMENT_TIME_END": "2020-01-02 00:00",
    "DOMAIN_DEFINITION_METHOD": "lumped",
    "SUB_GRID_DISCRETIZATION": "GRUs",
    "FORCING_DATASET": "ERA5",
    "FORCING_TIME_STEP_SIZE": 3600,
}


def make_config(
    root: Path,
    *,
    model: str = "SUMMA",
    **overrides: Any,
) -> SymfluenceConfig:
    """Build a validated :class:`SymfluenceConfig` rooted under ``root``.

    Args:
        root: Directory to place ``data/`` and ``code/`` under — normally
            pytest's ``tmp_path``, so each test gets its own tree.
        model: Value for ``HYDROLOGICAL_MODEL``.
        **overrides: Any flat config key, including a model's own
            ``SETTINGS_<MODEL>_*`` keys. Overrides win over the defaults.

    Returns:
        A ``SymfluenceConfig`` — validated, so an unknown or malformed key fails
        here rather than deep inside the code under test.

    Example:
        >>> cfg = make_config(
        ...     tmp_path,
        ...     model='FUSE',
        ...     SETTINGS_FUSE_FILEMANAGER='fm_catch.txt',
        ...     FUSE_SPATIAL_MODE='lumped',
        ... )
    """
    flat: Dict[str, Any] = {
        "SYMFLUENCE_DATA_DIR": str(root / "data"),
        "SYMFLUENCE_CODE_DIR": str(root / "code"),
        "HYDROLOGICAL_MODEL": model,
        **_DEFAULTS,
        **overrides,
    }
    return SymfluenceConfig(**flat)


def scaffold_domain(
    config: SymfluenceConfig,
    *,
    base_settings_for: Iterable[Tuple[str, str]] = (),
) -> Dict[str, Path]:
    """Create the directory tree ``config`` points at, and return the paths.

    Args:
        config: A config built by :func:`make_config` (or any equivalent).
        base_settings_for: ``(model_name, package_name)`` pairs to create a
            ``base_settings/`` directory for, each seeded with a placeholder
            ``<MODEL>_settings.txt``. Per-package by design: each model package
            ships its own ``base_settings/``, resolved through
            ``R.base_settings`` rather than a path core knows.

    Returns:
        ``{'data_dir', 'code_dir', 'domain_dir'}`` as ``Path`` objects.

    The caller supplies the model/package pairs for the same reason
    :func:`make_config` takes ``**overrides``: core must not carry a list of
    which models exist.
    """
    data_dir = Path(config.system.data_dir)
    code_dir = Path(config.system.code_dir)
    domain_dir = data_dir / f"domain_{config.domain.name}"

    for directory in (data_dir, code_dir, domain_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for model_name, package_name in base_settings_for:
        base_settings = (
            code_dir / "src" / "symfluence" / "models" / package_name / "base_settings"
        )
        base_settings.mkdir(parents=True, exist_ok=True)
        (base_settings / f"{model_name}_settings.txt").write_text(
            "# placeholder base settings written by symfluence.testing\n",
            encoding="utf-8",
        )

    return {"data_dir": data_dir, "code_dir": code_dir, "domain_dir": domain_dir}
