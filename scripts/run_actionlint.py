#!/usr/bin/env python3
"""Run actionlint with the project's shellcheck policy.

Single source of truth for the workflow-lint gate (CI lint job and the
pre-commit hook both call this). The excluded shellcheck codes are
pre-existing style debt in the workflow run-scripts (quoting, ls-vs-find,
redirect grouping) plus SC2193, which false-positives on comparisons
against ``${{ }}`` expressions actionlint substitutes with placeholders.
Error-level shellcheck findings and all native actionlint checks still
fail the gate. Shrink this list as the scripts are cleaned up; never grow
it without recording why.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

SHELLCHECK_EXCLUSIONS = (
    "SC2012",  # ls-vs-find in artifact listings
    "SC2086",  # unquoted expansions in run scripts
    "SC2129",  # individual >> redirects vs grouped block
    "SC2155",  # declare-and-assign masking return values
    "SC2193",  # false positive: ${{ }} placeholder comparisons
    "SC2231",  # unquoted expansions in for-loop globs
)


def main() -> int:
    actionlint = shutil.which("actionlint")
    if actionlint is None:
        print("actionlint not found on PATH (pip install actionlint-py)", file=sys.stderr)
        return 1
    env = dict(os.environ)
    env["SHELLCHECK_OPTS"] = " ".join(
        f"-e {code}" for code in SHELLCHECK_EXCLUSIONS
    )
    return subprocess.run([actionlint, *sys.argv[1:]], env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
