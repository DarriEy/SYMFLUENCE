# SPDX-License-Identifier: GPL-3.0-or-later
"""Typed catalog for lazily materializing external-build shell snippets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class BuildSnippetCatalog:
    """Named snippet factories with output validation."""

    factories: Mapping[str, Callable[[], str]]

    def render(self) -> dict[str, str]:
        rendered: dict[str, str] = {}
        for name, factory in self.factories.items():
            snippet = factory()
            if not snippet.strip():
                raise ValueError(f"Build snippet {name!r} is empty")
            rendered[name] = snippet
        return rendered
