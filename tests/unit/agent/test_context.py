# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the SYMFLUENCE agent identity / project-context builder."""
from __future__ import annotations

from symfluence.agent import context


def _write_config(path, **keys):
    path.write_text(
        "\n".join(f"{k}: {v}" for k, v in keys.items()) + "\n", encoding='utf-8'
    )


def test_detects_flat_config_and_domain_dirs(tmp_path):
    _write_config(tmp_path / 'config_bow.yaml',
                  DOMAIN_NAME='bow', HYDROLOGICAL_MODEL='SUMMA')
    (tmp_path / 'domain_bow').mkdir()
    (tmp_path / 'not_a_domain').mkdir()

    detected = context.detect_project_context(tmp_path)

    assert len(detected['configs']) == 1
    _, summary = detected['configs'][0]
    assert summary['DOMAIN_NAME'] == 'bow'
    assert detected['domains'] == ['domain_bow']


def test_detects_nested_config_keys(tmp_path):
    (tmp_path / 'config.yaml').write_text(
        "domain:\n  DOMAIN_NAME: nested_basin\n", encoding='utf-8'
    )
    detected = context.detect_project_context(tmp_path)
    assert detected['configs'][0][1]['DOMAIN_NAME'] == 'nested_basin'


def test_rejects_non_symfluence_yaml(tmp_path):
    (tmp_path / 'docker-compose.yaml').write_text(
        "services:\n  web:\n    image: nginx\n", encoding='utf-8'
    )
    (tmp_path / 'broken.yaml').write_text("{not: [valid", encoding='utf-8')

    detected = context.detect_project_context(tmp_path)
    assert detected['configs'] == []


def test_prefers_0_config_files_dir(tmp_path):
    config_dir = tmp_path / '0_config_files'
    config_dir.mkdir()
    _write_config(config_dir / 'config_a.yaml', DOMAIN_NAME='primary')
    _write_config(tmp_path / 'z_other.yaml', DOMAIN_NAME='secondary')

    detected = context.detect_project_context(tmp_path)
    names = [p.name for p, _ in detected['configs']]
    assert names[0] == 'config_a.yaml'


def test_identity_prompt_contains_identity_context_and_rules(tmp_path):
    _write_config(tmp_path / 'config.yaml', DOMAIN_NAME='bow',
                  EXPERIMENT_ID='run_1')
    prompt = context.build_identity_prompt(tmp_path)

    assert 'symfluence agent' in prompt
    assert 'DOMAIN_NAME=bow' in prompt
    assert 'symfluence list' in prompt
    assert 'symfluence workflow' in prompt


def test_identity_prompt_without_config_suggests_templates(tmp_path):
    prompt = context.build_identity_prompt(tmp_path)
    assert 'No SYMFLUENCE config file detected' in prompt
