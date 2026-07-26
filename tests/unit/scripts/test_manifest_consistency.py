# SPDX-License-Identifier: GPL-3.0-or-later
"""Focused tests for dependency-manifest grouping rules."""

from __future__ import annotations

from pathlib import Path

from scripts.check_manifest_consistency import load_pixi, load_pyproject


def test_pyproject_loader_keeps_optional_dependencies_out_of_core(tmp_path: Path) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        """
[project]
dependencies = ["numpy>=2"]

[project.optional-dependencies]
ml = ["torch>=2"]
""".strip(),
        encoding="utf-8",
    )

    core, groups = load_pyproject(manifest)

    assert set(core) == {"numpy"}
    assert set(groups["ml"]) == {"torch"}


def test_pixi_loader_keeps_feature_dependencies_out_of_default(tmp_path: Path) -> None:
    manifest = tmp_path / "pixi.toml"
    manifest.write_text(
        """
[dependencies]
numpy = ">=2"

[feature.ml.dependencies]
pytorch = { version = ">=2", channel = "pytorch" }
""".strip(),
        encoding="utf-8",
    )

    default, features = load_pixi(manifest)

    assert set(default) == {"numpy"}
    assert features["ml"] == {"pytorch": ">=2"}
