# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for launch-time priming (skills + identity + MCP + subagents)."""
from __future__ import annotations

import json

import pytest

from symfluence.agent import priming, registry


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Keep priming off the real temp cache and env."""
    monkeypatch.delenv('SYMFLUENCE_NO_SKILLS', raising=False)
    monkeypatch.setattr('tempfile.gettempdir', lambda: str(tmp_path / 'cache'))


CLAUDE = registry.get('claude')
CODEX = registry.get('codex')
GEMINI = registry.get('gemini')


def test_no_skills_flag_skips_all_priming(tmp_path):
    argv, messages = priming.prime_launch(CLAUDE, tmp_path, no_skills=True)
    assert argv == []
    assert any('disabled' in m for m in messages)


def test_no_skills_env_skips_all_priming(monkeypatch, tmp_path):
    monkeypatch.setenv('SYMFLUENCE_NO_SKILLS', '1')
    argv, _ = priming.prime_launch(CLAUDE, tmp_path)
    assert argv == []


def test_claude_priming_wires_all_four_layers(tmp_path):
    argv, messages = priming.prime_launch(CLAUDE, tmp_path)

    assert '--add-dir' in argv                  # skills
    assert '--append-system-prompt' in argv     # identity
    assert '--mcp-config' in argv               # MCP server
    assert '--agents' in argv                   # subagents

    identity = argv[argv.index('--append-system-prompt') + 1]
    assert 'SYMFLUENCE' in identity
    assert str(tmp_path) in identity  # live project context mentions workdir

    mcp_path = argv[argv.index('--mcp-config') + 1]
    config = json.loads(open(mcp_path, encoding='utf-8').read())
    assert config['mcpServers']['symfluence']['args'][-2:] == ['agent', 'mcp']

    agents = json.loads(argv[argv.index('--agents') + 1])
    assert 'calibration-debugger' in agents
    assert agents['calibration-debugger']['prompt']
    assert 'platform-scout' in agents


def test_agents_md_cli_gets_identity_preamble(tmp_path):
    argv, _ = priming.prime_launch(CODEX, tmp_path)

    # Identity rides in AGENTS.md, not in flags this CLI doesn't have.
    assert '--append-system-prompt' not in argv
    text = (tmp_path / 'AGENTS.md').read_text(encoding='utf-8')
    assert text.startswith('This session was started by `symfluence agent`')
    assert '# SYMFLUENCE agent skills' in text


def test_codex_mcp_wired_via_config_overrides(tmp_path):
    argv, _ = priming.prime_launch(CODEX, tmp_path)

    overrides = [argv[i + 1] for i, part in enumerate(argv) if part == '-c']
    assert any(o.startswith('mcp_servers.symfluence.command=') for o in overrides)
    args_override = next(
        o for o in overrides if o.startswith('mcp_servers.symfluence.args=')
    )
    assert json.loads(args_override.split('=', 1)[1])[-2:] == ['agent', 'mcp']


def test_cli_without_mcp_flags_gets_manual_hint(tmp_path):
    argv, messages = priming.prime_launch(GEMINI, tmp_path)

    assert argv == []  # nothing this CLI can't consume is forced on it
    assert any('agent mcp' in m for m in messages)


def test_agents_md_left_alone_if_present(tmp_path):
    (tmp_path / 'AGENTS.md').write_text('mine', encoding='utf-8')
    priming.prime_launch(CODEX, tmp_path)
    assert (tmp_path / 'AGENTS.md').read_text(encoding='utf-8') == 'mine'


def test_build_agents_json_parses_packaged_definitions():
    agents = json.loads(priming.build_agents_json())
    for spec in agents.values():
        assert set(spec) == {'description', 'prompt'}
        assert spec['description'] and spec['prompt']
