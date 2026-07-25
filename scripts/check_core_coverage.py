#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Core-spine coverage gate — the single definition shared by every workflow.

The global floor (25%) blends in model-binary/live-API code that can't run in
CI. The core spine (config parsing, registry, path resolution, mixins) IS
fully testable, so it gets its own stricter ratchet, reusing the ``.coverage``
data from the unit-test step. Instrumentation-only modules (profiling, npm
packaging) are excluded, and the packages promoted into core by the
service-decomposition campaign (calibration engine, metrics stack, build
helpers, the model-adapter tier) keep their previous governance — the global
floor — because they exercise model binaries, SLURM, and observation files
that can't run in CI.

One-way ratchet: raise FAIL_UNDER as coverage improves, never lower.

This script exists because the gate was previously duplicated across
``ci.yml`` and ``cross-platform.yml`` with a "mirrors the other file" comment
— and the copies drifted, turning every develop push red in one workflow
while the other stayed green. Workflows must invoke this script instead of
inlining the command.
"""
from __future__ import annotations

import subprocess
import sys

FAIL_UNDER = 80

INCLUDE = "src/symfluence/core/*"

OMIT = ",".join(
    [
        "src/symfluence/core/profiling/*",
        "src/symfluence/core/npm_bundle.py",
        "src/symfluence/core/calibration/*",
        "src/symfluence/core/metrics/*",
        "src/symfluence/core/build/*",
        "src/symfluence/core/modeling/*",
    ]
)


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "coverage",
        "report",
        f"--include={INCLUDE}",
        f"--omit={OMIT}",
        f"--fail-under={FAIL_UNDER}",
    ]
    print("core coverage gate:", " ".join(cmd[3:]))
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
