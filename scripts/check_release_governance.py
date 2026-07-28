#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Enforce repository-level release workflow security invariants."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
PUBLISH_WORKFLOWS = ("publish-pypi.yml", "release-binaries.yml", "auto-release.yml")
PINNED_ACTION = re.compile(r"^\s*uses:\s*[^#\s]+@([0-9a-f]{40})(?:\s|#|$)", re.MULTILINE)
ANY_ACTION = re.compile(r"^\s*uses:\s*[^#\s]+@([^\s#]+)", re.MULTILINE)


def main() -> int:
    errors: list[str] = []
    if not (ROOT / ".github" / "CODEOWNERS").is_file():
        errors.append("missing .github/CODEOWNERS")

    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        if "pull_request_target:" in text and workflow.name != "cla.yml":
            errors.append(f"{workflow.name}: pull_request_target is forbidden")
        all_refs = ANY_ACTION.findall(text)
        pinned_refs = PINNED_ACTION.findall(text)
        if len(all_refs) != len(pinned_refs):
            errors.append(f"{workflow.name}: every third-party action must use a full commit SHA")

    for name in PUBLISH_WORKFLOWS:
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        if "environment:" not in text:
            errors.append(f"{name}: publishing jobs require a protected GitHub environment")
        if name != "auto-release.yml" and "id-token: write" not in text:
            errors.append(f"{name}: publishing requires scoped OIDC id-token permission")
        if name == "publish-pypi.yml" and "attestations: true" not in text:
            errors.append(f"{name}: PyPI attestations must remain enabled")

    if errors:
        print("Release governance check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Release governance check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
