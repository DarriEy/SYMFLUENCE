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
    report = priming.prime_launch(CLAUDE, tmp_path, no_skills=True)
    assert report.argv == []
    assert not any(report.layers.values())
    assert any('disabled' in m for m in report.notes)


def test_no_skills_env_skips_all_priming(monkeypatch, tmp_path):
    monkeypatch.setenv('SYMFLUENCE_NO_SKILLS', '1')
    assert priming.prime_launch(CLAUDE, tmp_path).argv == []


def test_claude_priming_wires_all_four_layers(tmp_path):
    report = priming.prime_launch(CLAUDE, tmp_path)
    argv = report.argv
    assert all(report.layers.values())

    assert '--add-dir' in argv                  # skills
    assert '--append-system-prompt' in argv     # identity
    assert '--mcp-config' in argv               # MCP server
    assert '--agents' in argv                   # subagents

    identity = argv[argv.index('--append-system-prompt') + 1]
    assert 'SYMFLUENCE' in identity
    assert str(tmp_path) in identity  # live project context mentions workdir

    mcp_path = argv[argv.index('--mcp-config') + 1]
    config = json.loads(open(mcp_path, encoding='utf-8').read())
    server_args = config['mcpServers']['symfluence']['args']
    assert server_args[-4:] == ['agent', 'mcp', '--mode', 'code']

    agents = json.loads(argv[argv.index('--agents') + 1])
    assert 'calibration-debugger' in agents
    assert agents['calibration-debugger']['prompt']
    assert 'platform-scout' in agents


def test_agents_md_cli_gets_identity_preamble(tmp_path):
    report = priming.prime_launch(CODEX, tmp_path)
    argv = report.argv

    # Identity rides in AGENTS.md, not in flags this CLI doesn't have.
    assert '--append-system-prompt' not in argv
    assert report.layers['skills'] and report.layers['identity']
    text = (tmp_path / 'AGENTS.md').read_text(encoding='utf-8')
    assert text.startswith('This session was started by `symfluence agent`')
    assert '# SYMFLUENCE agent skills' in text


def test_codex_mcp_wired_via_config_overrides(tmp_path):
    argv = priming.prime_launch(CODEX, tmp_path).argv

    overrides = [argv[i + 1] for i, part in enumerate(argv) if part == '-c']
    assert any(o.startswith('mcp_servers.symfluence.command=') for o in overrides)
    args_override = next(
        o for o in overrides if o.startswith('mcp_servers.symfluence.args=')
    )
    server_args = json.loads(args_override.split('=', 1)[1])
    assert server_args[-4:] == ['agent', 'mcp', '--mode', 'code']


def test_cli_without_mcp_flags_gets_manual_hint(tmp_path):
    report = priming.prime_launch(GEMINI, tmp_path)

    assert report.argv == []  # nothing this CLI can't consume is forced on it
    assert report.layers['mcp'] is False
    assert any('agent mcp' in m for m in report.notes)


def test_agents_md_left_alone_if_present(tmp_path):
    (tmp_path / 'AGENTS.md').write_text('mine', encoding='utf-8')
    report = priming.prime_launch(CODEX, tmp_path)
    assert (tmp_path / 'AGENTS.md').read_text(encoding='utf-8') == 'mine'
    # ...and the gap is reported honestly, not silently swallowed.
    assert report.layers['skills'] is False
    assert report.layers['identity'] is False
    assert any('NOT injected' in w for w in report.warnings)


def test_priming_filesystem_failure_degrades_not_aborts(tmp_path, monkeypatch):
    """A PermissionError during skill materialization must not abort the launch."""
    def boom(*args, **kwargs):
        raise PermissionError('shared temp dir owned by another user')

    import symfluence.resources
    monkeypatch.setattr(symfluence.resources, 'prepare_agent_context', boom)

    report = priming.prime_launch(CLAUDE, tmp_path)
    assert report.layers['skills'] is False
    assert report.layers['identity'] is True   # system-prompt flag still works
    assert any('without SYMFLUENCE skills' in w for w in report.warnings)


def test_build_agents_json_parses_packaged_definitions():
    agents = json.loads(priming.build_agents_json())
    for spec in agents.values():
        assert set(spec) == {'description', 'prompt'}
        assert spec['description'] and spec['prompt']


def test_build_agents_json_filters_to_named_subagents():
    agents = json.loads(priming.build_agents_json(('calibration-debugger',)))
    assert set(agents) == {'calibration-debugger'}
    assert priming.build_agents_json(('no-such-subagent',)) is None


def test_modelling_mode_primes_operational_subset(tmp_path):
    from pathlib import Path

    from symfluence.agent.modes import AgentMode

    report = priming.prime_launch(CLAUDE, tmp_path, mode=AgentMode.MODELLING)
    argv = report.argv

    # Identity carries the modelling rules.
    identity = argv[argv.index('--append-system-prompt') + 1]
    assert 'Never modify SYMFLUENCE platform source' in identity

    # Skills cache holds only the modelling subset, in a mode-scoped dir.
    cache_root = Path(argv[argv.index('--add-dir') + 1])
    assert cache_root.name == 'model'
    materialized = sorted(p.name for p in (cache_root / '.claude' / 'skills').iterdir())
    assert materialized == [
        'debug-calibration', 'explore-platform', 'run-workflow-locally',
    ]

    # The MCP server is registered under the modelling profile.
    config = json.loads(
        Path(argv[argv.index('--mcp-config') + 1]).read_text(encoding='utf-8'))
    server_args = config['mcpServers']['symfluence']['args']
    assert server_args[-2:] == ['--mode', 'model']


def test_modes_use_separate_cache_dirs(tmp_path):
    from symfluence.agent.modes import AgentMode

    model_argv = priming.prime_launch(CLAUDE, tmp_path, mode=AgentMode.MODELLING).argv
    code_argv = priming.prime_launch(CLAUDE, tmp_path, mode=AgentMode.CODING).argv

    model_dir = model_argv[model_argv.index('--add-dir') + 1]
    code_dir = code_argv[code_argv.index('--add-dir') + 1]
    assert model_dir != code_dir
