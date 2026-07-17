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
    extra_args, _, delivered = prepare_agent_context('claude_native', tmp_path)
    assert delivered is True
    assert extra_args[0] == '--add-dir'
    skills_root = Path(extra_args[1]) / '.claude' / 'skills'
    for name in EXPECTED_SKILLS:
        assert (skills_root / name / 'SKILL.md').is_file()


def test_agents_md_written(tmp_path, monkeypatch):
    monkeypatch.delenv('SYMFLUENCE_NO_SKILLS', raising=False)
    extra_args, _, delivered = prepare_agent_context('agents_md', tmp_path)
    assert delivered is True
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
    extra_args, _, delivered = prepare_agent_context('agents_md', tmp_path)
    assert delivered is False
    assert extra_args == []
    assert not (tmp_path / 'AGENTS.md').exists()


def test_foreign_agents_md_reports_undelivered(tmp_path, monkeypatch):
    """A user-authored AGENTS.md is untouched AND the gap is reported honestly."""
    monkeypatch.delenv('SYMFLUENCE_NO_SKILLS', raising=False)
    (tmp_path / 'AGENTS.md').write_text('my own instructions', encoding='utf-8')
    _, messages, delivered = prepare_agent_context(
        'agents_md', tmp_path, preamble='IDENTITY')
    assert delivered is False
    assert any('NOT injected' in m for m in messages)


def test_symfluence_generated_agents_md_is_refreshed(tmp_path, monkeypatch):
    """An AGENTS.md we generated earlier is regenerated (stale preamble refresh)."""
    monkeypatch.delenv('SYMFLUENCE_NO_SKILLS', raising=False)
    prepare_agent_context('agents_md', tmp_path, preamble='OLD PREAMBLE')
    _, _, delivered = prepare_agent_context(
        'agents_md', tmp_path, preamble='NEW PREAMBLE')
    assert delivered is True
    text = (tmp_path / 'AGENTS.md').read_text(encoding='utf-8')
    assert 'NEW PREAMBLE' in text
    assert 'OLD PREAMBLE' not in text


def test_claude_native_skill_filter_and_cache_scope(tmp_path, monkeypatch):
    monkeypatch.delenv('SYMFLUENCE_NO_SKILLS', raising=False)
    monkeypatch.setattr('tempfile.gettempdir', lambda: str(tmp_path / 'cache'))
    subset = ('explore-platform', 'debug-calibration')

    extra_args, _, delivered = prepare_agent_context(
        'claude_native', tmp_path, skills=subset, cache_scope='model')

    assert delivered is True
    cache_root = Path(extra_args[1])
    assert cache_root.name == 'model'
    names = sorted(p.name for p in (cache_root / '.claude' / 'skills').iterdir())
    assert names == sorted(subset)


def test_claude_native_copies_whole_skill_dir(tmp_path, monkeypatch):
    """Reference/asset files next to SKILL.md must survive materialization."""
    monkeypatch.delenv('SYMFLUENCE_NO_SKILLS', raising=False)
    monkeypatch.setattr('tempfile.gettempdir', lambda: str(tmp_path / 'cache'))

    skills_src = tmp_path / 'skills-src'
    skill = skills_src / 'with-assets'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('---\nname: with-assets\n---\nbody',
                                    encoding='utf-8')
    (skill / 'reference.md').write_text('extra material', encoding='utf-8')
    import symfluence.resources.manager as manager
    monkeypatch.setattr(manager, 'get_skills_dir', lambda: skills_src)

    extra_args, _, delivered = prepare_agent_context('claude_native', tmp_path)

    assert delivered is True
    materialized = Path(extra_args[1]) / '.claude' / 'skills' / 'with-assets'
    assert (materialized / 'SKILL.md').is_file()
    assert (materialized / 'reference.md').is_file()


def test_agents_md_skill_filter(tmp_path, monkeypatch):
    monkeypatch.delenv('SYMFLUENCE_NO_SKILLS', raising=False)
    _, _, delivered = prepare_agent_context(
        'agents_md', tmp_path, skills=('explore-platform',))
    assert delivered is True
    text = (tmp_path / 'AGENTS.md').read_text(encoding='utf-8')
    assert '## explore-platform' in text
    assert '## add-model-handler' not in text


def test_parse_frontmatter_contract(tmp_path):
    from symfluence.resources import parse_frontmatter

    good = tmp_path / 'good.md'
    good.write_text('---\nname: x\ndescription: y\n---\nBody --- with a ruler\n',
                    encoding='utf-8')
    meta, body = parse_frontmatter(good)
    assert meta['name'] == 'x'
    assert body == 'Body --- with a ruler'

    no_fence = tmp_path / 'nofence.md'
    no_fence.write_text('Just text\n---\nnot: frontmatter\n---\n', encoding='utf-8')
    assert parse_frontmatter(no_fence) is None

    bad_yaml = tmp_path / 'bad.md'
    bad_yaml.write_text('---\n{not: [valid\n---\nbody\n', encoding='utf-8')
    assert parse_frontmatter(bad_yaml) is None
