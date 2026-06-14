# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for packaged agent-skill access and per-CLI materialization."""
from __future__ import annotations

from pathlib import Path

from symfluence.resources import get_skills_dir, prepare_agent_context

EXPECTED_SKILLS = [
    'add-data-handler',
    'add-model-handler',
    'add-optimizer',
    'debug-calibration',
    'explore-platform',
    'run-workflow-locally',
]


def test_packaged_skills_present():
    """The four skills ship inside the package (guards wheel packaging)."""
    skills_dir = get_skills_dir()
    names = sorted(p.name for p in skills_dir.iterdir() if (p / 'SKILL.md').is_file())
    assert names == EXPECTED_SKILLS


def test_claude_native_builds_add_dir(tmp_path, monkeypatch):
    monkeypatch.delenv('SYMFLUENCE_NO_SKILLS', raising=False)
    extra_args, _ = prepare_agent_context('claude_native', tmp_path)
    assert extra_args[0] == '--add-dir'
    skills_root = Path(extra_args[1]) / '.claude' / 'skills'
    for name in EXPECTED_SKILLS:
        assert (skills_root / name / 'SKILL.md').is_file()


def test_agents_md_written(tmp_path, monkeypatch):
    monkeypatch.delenv('SYMFLUENCE_NO_SKILLS', raising=False)
    extra_args, _ = prepare_agent_context('agents_md', tmp_path)
    agents_md = tmp_path / 'AGENTS.md'
    assert extra_args == []
    assert agents_md.is_file()
    text = agents_md.read_text(encoding='utf-8')
    for name in EXPECTED_SKILLS:
        assert name in text


def test_agents_md_non_clobber(tmp_path, monkeypatch):
    monkeypatch.delenv('SYMFLUENCE_NO_SKILLS', raising=False)
    agents_md = tmp_path / 'AGENTS.md'
    agents_md.write_text('SENTINEL', encoding='utf-8')
    prepare_agent_context('agents_md', tmp_path)
    assert agents_md.read_text(encoding='utf-8') == 'SENTINEL'  # left unchanged


def test_no_skills_env_skips_materialization(tmp_path, monkeypatch):
    monkeypatch.setenv('SYMFLUENCE_NO_SKILLS', '1')
    extra_args, _ = prepare_agent_context('agents_md', tmp_path)
    assert extra_args == []
    assert not (tmp_path / 'AGENTS.md').exists()
