# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Terminal presentation for the SYMFLUENCE agent handoff."""
from __future__ import annotations

from pathlib import Path

from rich.console import Group
from rich.panel import Panel
from rich.text import Text


def launch_card(
    *,
    launcher_name: str,
    workdir: Path,
    interactive: bool,
    primed: bool,
) -> Panel:
    """Build the compact launch card shown before the host CLI takes over."""
    title = Text("SYMFLUENCE", style="bold bright_cyan")
    title.append("  Agent", style="bold white")

    route = Text()
    route.append("Runtime     ", style="dim")
    route.append(launcher_name, style="bold white")
    route.append("  ·  ", style="dim")
    route.append("interactive session" if interactive else "one-shot task", style="#b9dce5")

    context = Text()
    context.append("Project     ", style="dim")
    context.append(workdir.name or str(workdir), style="white")

    capabilities = Text()
    capabilities.append("Context     ", style="dim")
    if primed:
        capabilities.append("Skills  ·  Project  ·  MCP", style="#43d6b5")
    else:
        capabilities.append("Disabled", style="yellow")

    body = Group(title, Text(""), route, context, capabilities)
    return Panel(
        body,
        border_style="#287f95",
        padding=(1, 2),
    )
