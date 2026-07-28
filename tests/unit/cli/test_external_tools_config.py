# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Regression tests for external build-tool registrations.

Pins reproducibility-critical properties (e.g. TauDEM pinned to a tagged
release) so that accidental reversions to floating HEAD are caught in CI.
"""
from __future__ import annotations

import types

import pytest

from symfluence.cli.external_tools_config import get_external_tools_definitions

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _compiled_string_constants(source: str, filename: str) -> set[str]:
    """Every string constant in *source*, recursively through code objects.

    A source-text grep for a dotted path is defeated by writing it as
    ``'symfluence.' + 'models'``. Compiling first defeats that back: CPython's
    peephole optimiser folds adjacent literal concatenation into a single
    constant, so the reassembled path shows up here.
    """
    def walk(code: types.CodeType) -> set[str]:
        found: set[str] = set()
        for constant in code.co_consts:
            if isinstance(constant, str):
                found.add(constant)
            elif isinstance(constant, types.CodeType):
                found |= walk(constant)
        return found

    return walk(compile(source, filename, 'exec'))


def test_no_model_packages_are_hardcoded():
    """Model build instructions are declared by the model, never enumerated here.

    Supersedes the narrower cfuse/droute pin (issue #150): those two were
    extracted to standalone pip packages and left behind in a hardcoded
    ``model_modules`` list, which logged a spurious "Failed to load build
    instructions" warning on every ``symfluence binary`` invocation.  The list
    is gone — a model package registers ``build_instructions_module`` through
    ``model_manifest`` and the CLI reads ``R.build_instructions`` — so the
    failure mode cannot recur, for in-tree or external packages alike.

    Scanned twice: the source text (so prose in that module says "the models
    package" rather than spelling the dotted name) and the compiled string
    constants (so a path split across literals cannot slip past the grep).
    """
    import inspect

    from symfluence.cli import external_tools_config

    src = inspect.getsource(external_tools_config)
    assert 'symfluence.models' not in src, (
        "external_tools_config must not name model packages; build instructions "
        "are declared by the model package via model_manifest("
        "build_instructions_module=...) and read from R.build_instructions."
    )
    offenders = sorted(
        constant
        for constant in _compiled_string_constants(
            src, external_tools_config.__file__)
        if 'symfluence.models' in constant
    )
    assert not offenders, (
        f"external_tools_config assembles model-package paths from string "
        f"fragments: {offenders}"
    )


def test_model_build_instructions_are_discovered_from_the_registry():
    """In-tree models still surface in `symfluence binary`, via the registry."""
    tools = get_external_tools_definitions()
    for tool in ('summa', 'fuse', 'mizuroute', 'wmfire', 'rhessys'):
        assert tool in tools, f"{tool} disappeared from the build-tool catalogue"
        assert tools[tool].get('description')


def test_taudem_pinned_to_release_tag():
    """TauDEM must be pinned to a tagged release, not HEAD.

    AI/IA/AP reported TauDEM build failing intermittently with 'pitremove'
    missing. Root cause: upstream HEAD drift. Pinning to a tag (git clone -b
    works for both branches and tags) freezes the install against a known-good
    version. Dropping the pin would regress to floating HEAD.
    """
    tools = get_external_tools_definitions()
    assert 'taudem' in tools, "TauDEM build instructions must be registered"
    spec = tools['taudem']
    assert spec.get('branch'), (
        "TauDEM must pin 'branch' to a tagged release (e.g. 'v5.4.0'); "
        "a None branch lets upstream HEAD drift break installs."
    )
    # Must look like a release tag, not a moving branch name
    branch = str(spec['branch'])
    assert branch.startswith('v') and any(c.isdigit() for c in branch), (
        f"TauDEM branch '{branch}' does not look like a release tag. "
        "Pin to something like 'v5.4.0'."
    )
