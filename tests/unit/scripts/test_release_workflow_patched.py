# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The release workflow must build RHESSys the way the paper does.

`--paper-repro` forces `--patched` on purpose. argument_parser.py says why:

    RHESSys is always built --patched here: the paper's runs use the
    SYMFLUENCE subsurface-GW physics, and an unpatched binary silently caps
    its calibration (KGE ~0.15 vs 0.85).

The release job builds the same tool through a separate RELEASE_EXTRAS loop,
which did not pass the flag. Released and npm-installed RHESSys was therefore
not the paper's binary, and nothing failed to say so — a reproduction run would
simply score ~0.15 and look like a bad calibration.

These tests tie the two paths together so they cannot drift apart again.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOW = (Path(__file__).resolve().parents[3]
            / ".github" / "workflows" / "release-binaries.yml")

#: Tools whose released build must carry SYMFLUENCE patches. Keep in step with
#: the --paper-repro behaviour in cli/commands/binary_commands.py.
MUST_BE_PATCHED = {"rhessys"}


@pytest.fixture(scope="module")
def workflow_text() -> str:
    if not WORKFLOW.exists():
        pytest.skip(f"workflow not found: {WORKFLOW}")
    return WORKFLOW.read_text(encoding="utf-8")


def test_rhessys_is_in_the_paper_tool_set():
    """If this ever stops being true, the guard below is arguing about nothing."""
    from symfluence.cli.argument_parser import PAPER_REPRO_TOOLS

    assert "rhessys" in PAPER_REPRO_TOOLS


def test_paper_repro_still_forces_patched():
    """The contract this whole guard is derived from."""
    src = (Path(__file__).resolve().parents[3] / "src" / "symfluence" / "cli"
           / "commands" / "binary_commands.py").read_text(encoding="utf-8")
    # --paper-repro must set patched, not merely default it
    assert re.search(r"paper_repro.*\n(?:.*\n){0,12}?\s*patched = True", src), \
        "--paper-repro no longer forces patched; update MUST_BE_PATCHED"


def test_release_workflow_builds_patched_tools_with_the_flag(workflow_text):
    """The regression: RELEASE_EXTRAS built RHESSys without --patched."""
    assert "RELEASE_EXTRAS=" in workflow_text, "RELEASE_EXTRAS loop not found"

    for tool in MUST_BE_PATCHED:
        assert re.search(rf"{tool}\s*\)\s*EXTRA_INSTALL_FLAGS=\"--patched\"",
                         workflow_text), (
            f"the release workflow must build {tool} with --patched; an "
            f"unpatched binary silently caps calibration"
        )


def test_the_install_invocation_actually_passes_the_flags(workflow_text):
    """A case statement that nothing interpolates would be decoration."""
    assert re.search(r"binary install \"\$tool\" \$EXTRA_INSTALL_FLAGS",
                     workflow_text), (
        "EXTRA_INSTALL_FLAGS is set but never passed to `binary install`"
    )


def test_patched_tools_are_actually_released(workflow_text):
    """A patched build nobody ships helps no one."""
    extras = re.search(r'RELEASE_EXTRAS="([^"]*)"', workflow_text)
    assert extras, "RELEASE_EXTRAS assignment not found"
    released = set(extras.group(1).split())
    missing = MUST_BE_PATCHED - released
    assert not missing, f"must-be-patched tools absent from RELEASE_EXTRAS: {missing}"
