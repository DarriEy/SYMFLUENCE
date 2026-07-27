# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""mizuRoute resolves its source runoff file from the registered declarations.

``MizuRouteRunner.fix_time_precision`` used to carry a 65-line if/elif chain
duplicating every model's runoff directory and filename -- the *fourth* copy of
that table, after ``runoff_loader``, the mizuRoute control writer, and the
schema each model registers. It now delegates to ``resolve_runoff_file``.

These tests pin the two things the collapse had to preserve: the source-model
precedence, and the filename each source resolves to.

CHANGED: HYPE left both. It is not a routable source — its runoff declaration
named ``{experiment_id}_timestep.nc``, which nothing in the HYPE adapter
writes, and its ``cout`` is already routed discharge at subbasin outlets, so
converting it would route it twice. With the declaration gone,
``resolve_runoff_file('HYPE')`` raises instead of resolving a path, and HYPE had
to leave the precedence too: otherwise a ``SUMMA,HYPE`` run would select HYPE
and fail where it previously routed SUMMA.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from symfluence.core.exceptions import ConfigurationError
from symfluence.core.modeling.utilities.runoff_loader import resolve_runoff_file

pytestmark = [pytest.mark.unit]

EXPERIMENT_ID = 'mizu_src_001'
DOMAIN_NAME = 'testdom'

# Precedence the if/elif chain applied, in order; SUMMA is the fallback.
# HYPE used to sit between GR and NGEN.
_PRECEDENCE = ('FUSE', 'GR', 'NGEN')


def _resolve_source(active_models) -> str:
    """The selection expression used by fix_time_precision."""
    return next((m for m in _PRECEDENCE if m in active_models), 'SUMMA')


@pytest.mark.parametrize(
    "active_models,expected",
    [
        (['SUMMA'], 'SUMMA'),
        ([], 'SUMMA'),
        (['MIZUROUTE'], 'SUMMA'),
        (['FUSE'], 'FUSE'),
        (['GR'], 'GR'),
        (['NGEN'], 'NGEN'),
        # FUSE outranks everything, as in the original chain.
        (['FUSE', 'GR', 'HYPE', 'NGEN', 'SUMMA'], 'FUSE'),
        (['GR', 'HYPE', 'NGEN'], 'GR'),
        (['HYPE', 'NGEN'], 'NGEN'),
        # A SUMMA+FUSE run routes FUSE, not SUMMA.
        (['FUSE', 'SUMMA'], 'FUSE'),
        # HYPE is not a routable source, so it never wins the selection —
        # a SUMMA+HYPE run routes SUMMA, and a HYPE-only run falls back.
        (['HYPE'], 'SUMMA'),
        (['HYPE', 'SUMMA'], 'SUMMA'),
    ],
)
def test_source_model_precedence_is_preserved(active_models, expected):
    assert _resolve_source(active_models) == expected


def test_hype_is_not_a_resolvable_source():
    """Asking for HYPE runoff fails immediately and says why."""
    with pytest.raises(ConfigurationError, match='HYPE'):
        resolve_runoff_file(
            source_model='HYPE',
            project_dir=Path('.'),
            experiment_id=EXPERIMENT_ID,
            domain_name=DOMAIN_NAME,
        )


@pytest.mark.parametrize(
    "source_model,filename",
    [
        ('SUMMA', f"{EXPERIMENT_ID}_timestep.nc"),
        ('GR', f"{DOMAIN_NAME}_{EXPERIMENT_ID}_runs_def.nc"),
        ('NGEN', f"{EXPERIMENT_ID}_runoff.nc"),
    ],
)
def test_each_source_resolves_the_filename_the_chain_hardcoded(
    tmp_path: Path, source_model, filename
):
    """The names the deleted if/elif chain built, now served by the schema."""
    out_dir = tmp_path / 'simulations' / EXPERIMENT_ID / source_model
    out_dir.mkdir(parents=True)
    (out_dir / filename).write_bytes(b'')

    resolved = resolve_runoff_file(
        source_model=source_model,
        project_dir=tmp_path,
        experiment_id=EXPERIMENT_ID,
        domain_name=DOMAIN_NAME,
    )

    assert resolved is not None
    assert resolved.name == filename


def test_fuse_file_id_is_hashed_past_six_chars(tmp_path: Path):
    """FUSE's Fortran-driven 6-char truncation survives the collapse."""
    import hashlib

    file_id = hashlib.md5(
        EXPERIMENT_ID.encode(), usedforsecurity=False
    ).hexdigest()[:6]
    expected = f"{DOMAIN_NAME}_{file_id}_runs_def.nc"

    out_dir = tmp_path / 'simulations' / EXPERIMENT_ID / 'FUSE'
    out_dir.mkdir(parents=True)
    (out_dir / expected).write_bytes(b'')

    resolved = resolve_runoff_file(
        source_model='FUSE',
        project_dir=tmp_path,
        experiment_id=EXPERIMENT_ID,
        domain_name=DOMAIN_NAME,
    )

    assert resolved is not None
    assert resolved.name == expected


def test_missing_output_resolves_to_none_not_a_guess(tmp_path: Path, caplog):
    """Nothing to route resolves to None so the runner can report it."""
    with caplog.at_level(logging.WARNING):
        resolved = resolve_runoff_file(
            source_model='NGEN',
            project_dir=tmp_path,
            experiment_id=EXPERIMENT_ID,
            domain_name=DOMAIN_NAME,
        )

    assert resolved is None
