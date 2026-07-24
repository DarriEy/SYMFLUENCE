# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import pytest

from symfluence.core.build.build_snippet_catalog import BuildSnippetCatalog
from symfluence.core.build.build_snippets import get_all_snippets


def test_catalog_renders_factories_lazily() -> None:
    calls: list[str] = []
    catalog = BuildSnippetCatalog({"example": lambda: calls.append("called") or "echo ready"})

    assert calls == []
    assert catalog.render() == {"example": "echo ready"}
    assert calls == ["called"]


def test_catalog_rejects_empty_snippet() -> None:
    with pytest.raises(ValueError, match="empty"):
        BuildSnippetCatalog({"broken": lambda: "  "}).render()


def test_public_catalog_contains_all_supported_snippets() -> None:
    snippets = get_all_snippets()

    assert set(snippets) == {
        "common_env",
        "netcdf_detect",
        "hdf5_detect",
        "netcdf_lib_detect",
        "safe_build_path",
        "geos_proj_detect",
        "udunits2_detect_build",
        "bison_detect_build",
        "flex_detect_build",
    }
    assert all(snippet.strip() for snippet in snippets.values())
