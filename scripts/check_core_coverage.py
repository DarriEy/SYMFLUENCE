#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Core coverage gates — the single definition shared by every workflow.

The global floor (25%) blends in model-binary/live-API code that can't run in
CI. The core spine (config parsing, registry, path resolution, mixins) IS fully
testable, so it gets its own stricter ratchet, reusing the ``.coverage`` data
from the unit-test step. Instrumentation-only modules (profiling, npm packaging)
are excluded outright.

``core/calibration`` and ``core/modeling`` were promoted into core by the
service-decomposition campaign and were left to the global floor alone. That is
too weak for what they now are: the primary contract surface the
``symfluence-models`` extraction rests on. Blending them into the 80% spine gate
is not the fix either — they measure 45% and 49%, so folding them in would drag
core to roughly 56% and force the spine floor DOWN, which ADR-0008 forbids.

Instead each gets its own floor at today's reality, which is the extension
ADR-0008 explicitly contemplates ("extending per-package gates beyond core/ ...
using the same pattern"). Weak coverage in the contract surface can no longer be
masked by the strong spine, nor drag the spine's floor down.

One-way ratchet, per ADR-0008: raise a floor as coverage improves, NEVER lower
one. Lowering requires revisiting that ADR.

``core/metrics`` and ``core/build`` remain governed by the global floor only —
they are not part of the extraction contract, and gating them is available
follow-on work using this same pattern.

This script exists because the gate was previously duplicated across ``ci.yml``
and ``cross-platform.yml`` with a "mirrors the other file" comment — and the
copies drifted, turning every develop push red in one workflow while the other
stayed green. Workflows must invoke this script instead of inlining the command.
"""
from __future__ import annotations

import subprocess
import sys
from typing import List, NamedTuple, Sequence


class Gate(NamedTuple):
    """One coverage floor over one slice of the tree."""

    name: str
    include: str
    fail_under: int
    omit: Sequence[str] = ()


#: Excluded from the spine gate outright — instrumentation, not logic.
_INSTRUMENTATION_OMIT = (
    "src/symfluence/core/profiling/*",
    "src/symfluence/core/npm_bundle.py",
)

#: Excluded from the spine gate because they carry their OWN floor below, or
#: (metrics, build) because they remain on the global floor for now. Keeping them
#: out of the spine number is what lets the spine floor stay at 80.
_SEPARATELY_GOVERNED_OMIT = (
    "src/symfluence/core/calibration/*",
    "src/symfluence/core/metrics/*",
    "src/symfluence/core/build/*",
    "src/symfluence/core/modeling/*",
)

GATES: List[Gate] = [
    Gate(
        name="core spine",
        include="src/symfluence/core/*",
        omit=_INSTRUMENTATION_OMIT + _SEPARATELY_GOVERNED_OMIT,
        fail_under=80,
    ),
    # Measured 45% on 2026-07-29; floored 4 points below, matching ADR-0008's
    # margin for parallel-run variance so the gate binds without flaking.
    Gate(
        name="core/calibration (extraction contract surface)",
        include="src/symfluence/core/calibration/*",
        fail_under=41,
    ),
    # Measured 49% on 2026-07-29; same 4-point margin.
    Gate(
        name="core/modeling (extraction contract surface)",
        include="src/symfluence/core/modeling/*",
        fail_under=45,
    ),
]


def _run(gate: Gate) -> int:
    cmd = [
        sys.executable,
        "-m",
        "coverage",
        "report",
        f"--include={gate.include}",
        f"--fail-under={gate.fail_under}",
    ]
    if gate.omit:
        cmd.append(f"--omit={','.join(gate.omit)}")
    print(f"\n=== coverage gate: {gate.name} (fail-under={gate.fail_under}) ===")
    print("  " + " ".join(cmd[3:]))
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    # Run every gate before failing, so one breach does not hide another.
    failed = [gate.name for gate in GATES if _run(gate) != 0]

    print()
    if failed:
        print("coverage gates FAILED:", file=sys.stderr)
        for name in failed:
            print(f"  - {name}", file=sys.stderr)
        print(
            "\nFloors are a one-way ratchet (ADR-0008): raise them as coverage "
            "improves, never lower them to make a build pass.",
            file=sys.stderr,
        )
        return 1

    print(f"all {len(GATES)} coverage gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
