# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

from io import StringIO
from pathlib import Path

from rich.console import Console

from symfluence.agent.presentation import launch_card


def _render(**overrides) -> str:
    output = StringIO()
    console = Console(file=output, width=78, color_system=None)
    values = {
        'launcher_name': 'Codex',
        'workdir': Path('/work/Bow_at_Banff'),
        'interactive': True,
        'primed': True,
    }
    values.update(overrides)
    console.print(launch_card(**values))
    return output.getvalue()


def test_launch_card_identifies_route_and_context():
    rendered = _render()
    assert 'SYMFLUENCE  Agent' in rendered
    assert 'Codex' in rendered
    assert 'interactive session' in rendered
    assert 'Bow_at_Banff' in rendered
    assert 'Skills  ·  Project  ·  MCP' in rendered


def test_launch_card_distinguishes_bare_one_shot_mode():
    rendered = _render(interactive=False, primed=False)
    assert 'one-shot task' in rendered
    assert 'Disabled' in rendered
