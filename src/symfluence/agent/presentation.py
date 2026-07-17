# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Terminal presentation for the SYMFLUENCE agent handoff."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from rich.console import Group
from rich.panel import Panel
from rich.text import Text

# Display labels for the priming layers, in card order.
_LAYER_LABELS = {
    'skills': 'Skills',
    'identity': 'Project',
    'mcp': 'MCP',
    'subagents': 'Subagents',
}


def launch_card(
    *,
    launcher_name: str,
    workdir: Path,
    interactive: bool,
    layers: Sequence[str] = (),
) -> Panel:
    """Build the compact launch card shown before the host CLI takes over.

    ``layers`` names the priming layers that are genuinely active for this
    launch (see :class:`~symfluence.agent.priming.PrimingReport`); the card
    renders exactly those, so it never claims context the CLI didn't get.
    """
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

    active = [_LAYER_LABELS[name] for name in _LAYER_LABELS if name in layers]
    capabilities = Text()
    capabilities.append("Context     ", style="dim")
    if len(active) == len(_LAYER_LABELS):
        capabilities.append("  ·  ".join(active), style="#43d6b5")
    elif active:
        capabilities.append("  ·  ".join(active), style="#43d6b5")
        capabilities.append("  (partial)", style="yellow")
    else:
        capabilities.append("Disabled", style="yellow")

    body = Group(title, Text(""), route, context, capabilities)
    return Panel(
        body,
        border_style="#287f95",
        padding=(1, 2),
    )
