# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""CI must actually run the conformance suite.

``tests/conformance/`` holds the proofs the service decomposition rests on:
that the framework imports and runs with the models layer physically absent,
and that the acquisition contract is extractable with no ``symfluence`` import.

Every workflow invoked pytest with explicit paths and none of them listed this
directory, so those guards existed and passed locally while CI never executed
them — a test nothing runs is indistinguishable from a test that does not exist,
and it is worse, because it reads as coverage. This pins the wiring so the
directory cannot fall out again.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_WORKFLOWS = Path(__file__).resolve().parents[3] / ".github" / "workflows"

#: A pytest invocation that runs the unit suite must also run conformance.
#: Matched on the command text rather than parsed YAML: the invocations are
#: multi-line shell inside ``run:`` blocks, and the question here is only which
#: paths reach pytest.
_PYTEST_CALL = re.compile(r"pytest[^\n]*(?:\\\s*\n[^\n]*)*")

#: Workflows that run part of the unit suite as a SMOKE CHECK of an installed
#: package, not as the authoritative suite. The conformance proofs are
#: architectural invariants of the source tree — that the framework works with
#: the models layer absent, that the contract imports standalone — so they are
#: verified once in the main suite rather than repeated across every entry of a
#: multi-OS, multi-install-method matrix. Exempting them is a decision; adding a
#: NEW workflow here should be one too.
_SMOKE_ONLY_WORKFLOWS = {
    "install-validate.yml",
    "install-validate-arm.yml",
}


def _pytest_invocations(text: str) -> list[str]:
    return [m.group(0).replace("\\\n", " ") for m in _PYTEST_CALL.finditer(text)]


def _workflow_files() -> list[Path]:
    files = sorted(_WORKFLOWS.glob("*.yml"))
    assert files, f"no workflow files found under {_WORKFLOWS}"
    return files


def test_every_unit_suite_invocation_also_runs_conformance():
    """No workflow may run tests/unit/ without tests/conformance/."""
    offenders: list[str] = []
    checked = 0

    for workflow in _workflow_files():
        if workflow.name in _SMOKE_ONLY_WORKFLOWS:
            continue
        text = workflow.read_text(encoding="utf-8")
        for call in _pytest_invocations(text):
            if "tests/unit/" not in call:
                continue
            checked += 1
            if "tests/conformance/" not in call:
                offenders.append(f"{workflow.name}: {call.strip()[:120]}")

    assert checked, (
        "no non-exempt workflow invokes pytest on tests/unit/ — the CI layout "
        "changed and this guard is now vacuous"
    )
    assert not offenders, (
        "these CI invocations run the unit suite but not the conformance "
        "suite, so the models-absent and extraction-readiness proofs would not "
        "be executed:\n  " + "\n  ".join(offenders)
    )


def test_the_decomposition_proofs_are_where_ci_looks_for_them():
    """The files the wiring exists to run must be in the directory CI runs.

    Guards the other half: pointing CI at the directory achieves nothing if the
    proofs move out of it.
    """
    conformance = Path(__file__).resolve().parents[2] / "conformance"
    required = {"test_models_absent.py", "test_extraction_readiness.py"}
    present = {p.name for p in conformance.glob("test_*.py")}

    missing = required - present
    assert not missing, (
        f"{sorted(missing)} are not in tests/conformance/, which is the "
        "directory CI is wired to run. Either move them back or update the "
        "workflow paths and this test together."
    )


def test_conformance_tests_are_selected_by_the_unit_marker():
    """CI selects with ``-m "unit"``; an unmarked conformance test is skipped.

    Listing the directory is necessary but not sufficient — a conformance file
    that carries only, say, ``@pytest.mark.integration`` would be deselected by
    the marker expression and silently not run.
    """
    conformance = Path(__file__).resolve().parents[2] / "conformance"
    unmarked = [
        path.name
        for path in sorted(conformance.glob("test_*.py"))
        if "pytest.mark.unit" not in path.read_text(encoding="utf-8")
    ]
    assert not unmarked, (
        f"{unmarked} carry no 'unit' marker, so CI's -m \"unit\" selection "
        "would deselect them even though the directory is listed."
    )


def test_the_smoke_only_exemptions_still_exist():
    """An exemption for a workflow that is gone hides the next one that is not."""
    present = {path.name for path in _workflow_files()}
    stale = sorted(_SMOKE_ONLY_WORKFLOWS - present)
    assert not stale, (
        f"{stale} are exempted from the conformance requirement but no longer "
        "exist — drop them from _SMOKE_ONLY_WORKFLOWS so the guard keeps its "
        "teeth."
    )
