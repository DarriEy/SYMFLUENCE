# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the platform-inspection helpers behind the agent's MCP tools."""
from __future__ import annotations

import pytest

from symfluence.agent import inspection


@pytest.fixture
def domain_tree(tmp_path):
    """A minimal domain_X tree with logs and one finished DDS calibration."""
    data = tmp_path / 'data'
    domain = data / 'domain_X'
    log_dir = domain / '_workLog_X'
    log_dir.mkdir(parents=True)
    (log_dir / 'symfluence_general_X_20260101.log').write_text(
        'INFO start\nWARNING soil moisture drifted\nINFO done\n', encoding='utf-8')
    (log_dir / 'symfluence_general_X_20260102.log').write_text(
        'INFO newest run\nERROR calibration exploded\n', encoding='utf-8')

    run = domain / 'optimization' / 'DDS_exp1'
    run.mkdir(parents=True)
    (run / 'optimization_history.csv').write_text(
        'generation,best_score,mean_score\n1,0.41,0.30\n2,0.55,0.42\n3,0.52,0.44\n',
        encoding='utf-8')
    (run / 'optimization_metadata.csv').write_text(
        'algorithm,experiment_id,target_metric,best_score,completed_at\n'
        'DDS,exp1,KGE,0.55,2026-01-02T10:00:00\n',
        encoding='utf-8')
    (run / 'best_parameters.csv').write_text('param,value\nk,1.5\n', encoding='utf-8')
    final = run / 'final_evaluation'
    final.mkdir()
    (final / 'metrics.csv').write_text(
        'KGE_calib,NSE_calib,notes\n0.55,0.48,fine\n', encoding='utf-8')

    (domain / 'simulations').mkdir()

    config = tmp_path / 'config.yaml'
    config.write_text(
        f"DOMAIN_NAME: X\nSYMFLUENCE_DATA_DIR: {data}\nEXPERIMENT_ID: exp1\n",
        encoding='utf-8')
    return config


def test_resolve_domain(domain_tree):
    domain, domain_dir = inspection.resolve_domain(domain_tree)
    assert domain == 'X'
    assert domain_dir.name == 'domain_X'
    assert domain_dir.is_dir()


def test_resolve_domain_requires_keys(tmp_path):
    config = tmp_path / 'c.yaml'
    config.write_text('DOMAIN_NAME: X\n', encoding='utf-8')
    with pytest.raises(ValueError, match='SYMFLUENCE_DATA_DIR'):
        inspection.resolve_domain(config)


def test_read_run_log_tails_newest(domain_tree):
    result = inspection.read_run_log(domain_tree, tail_lines=10)
    assert result['log_path'].endswith('20260102.log')
    assert 'ERROR calibration exploded' in result['lines'][-1]


def test_read_run_log_grep(domain_tree):
    result = inspection.read_run_log(domain_tree, grep='ERROR')
    assert result['lines'] == ['ERROR calibration exploded']


def test_list_domains(domain_tree):
    result = inspection.list_domains(config_path=domain_tree)
    assert len(result['domains']) == 1
    entry = result['domains'][0]
    assert entry['name'] == 'X'
    assert entry['experiments'] == ['DDS_exp1']
    assert entry['has_optimization'] is True
    assert entry['has_simulations'] is True


def test_calibration_status(domain_tree):
    status = inspection.calibration_status(domain_tree)
    assert status['algorithm'] == 'DDS'
    assert status['metric'] == 'KGE'
    assert status['iterations'] == 3
    assert status['best_score'] == pytest.approx(0.55)
    assert status['best_iteration'] == '2'
    assert status['in_progress'] is False


def test_calibration_status_in_progress(domain_tree):
    run_dir = inspection.resolve_domain(domain_tree)[1] / 'optimization' / 'DDS_exp1'
    (run_dir / 'optimization_metadata.csv').unlink()
    status = inspection.calibration_status(domain_tree)
    assert status['in_progress'] is True
    assert status['best_score'] == pytest.approx(0.55)


def test_get_results_summary(domain_tree):
    summary = inspection.get_results_summary(domain_tree, experiment_id='exp1')
    assert summary['metrics']['KGE'] == pytest.approx(0.55)
    assert summary['metrics']['KGE_calib'] == pytest.approx(0.55)
    assert summary['metrics']['NSE_calib'] == pytest.approx(0.48)
    assert any(s.endswith('metrics.csv') for s in summary['sources'])


def test_pick_experiment_unknown_id(domain_tree):
    with pytest.raises(ValueError, match='exp999'):
        inspection.calibration_status(domain_tree, experiment_id='exp999')


def test_get_plot_paths(domain_tree):
    run_dir = inspection.resolve_domain(domain_tree)[1] / 'optimization' / 'DDS_exp1'
    plots_dir = run_dir / 'plots'
    plots_dir.mkdir()
    (plots_dir / 'convergence.png').write_bytes(b'png')
    (run_dir / 'final_evaluation' / 'hydrograph.pdf').write_bytes(b'pdf')

    result = inspection.get_plot_paths(domain_tree)
    from pathlib import Path
    names = [Path(p).name for p in result['plots']]
    assert 'convergence.png' in names
    assert 'hydrograph.pdf' in names


def test_compare_experiments(domain_tree):
    run2 = inspection.resolve_domain(domain_tree)[1] / 'optimization' / 'PSO_exp2'
    run2.mkdir()
    (run2 / 'optimization_history.csv').write_text(
        'generation,best_score\n1,0.61\n2,0.70\n', encoding='utf-8')
    # No metadata: exp2 is still in progress.

    result = inspection.compare_experiments(domain_tree)
    rows = result['experiments']
    assert [r['run_name'] for r in rows] == ['PSO_exp2', 'DDS_exp1']  # 0.70 > 0.55
    assert rows[0]['in_progress'] is True
    assert rows[0]['best_score'] == pytest.approx(0.70)
    assert rows[1]['algorithm'] == 'DDS'
    assert rows[1]['in_progress'] is False


class _AcceptAll:
    @staticmethod
    def from_file(path):
        return object()


class _RejectAll:
    @staticmethod
    def from_file(path):
        raise ValueError('bad config')


def test_update_config_rewrites_in_place(domain_tree, monkeypatch):
    import symfluence.core.config.models as models
    monkeypatch.setattr(models, 'SymfluenceConfig', _AcceptAll)

    original = domain_tree.read_text(encoding='utf-8')
    result = inspection.update_config(
        domain_tree, {'EXPERIMENT_ID': 'exp2', 'NEW_KEY': 42})

    assert result['changed_keys'] == ['EXPERIMENT_ID']
    assert result['appended_keys'] == ['NEW_KEY']
    text = domain_tree.read_text(encoding='utf-8')
    assert 'EXPERIMENT_ID: exp2' in text
    assert 'NEW_KEY: 42' in text
    assert 'DOMAIN_NAME: X' in text  # untouched lines survive

    # The original was backed up verbatim.
    from pathlib import Path
    backup = Path(result['backup_path'])
    assert backup.read_text(encoding='utf-8') == original


def test_update_config_dry_run_writes_nothing(domain_tree, monkeypatch):
    import symfluence.core.config.models as models
    monkeypatch.setattr(models, 'SymfluenceConfig', _AcceptAll)

    before = domain_tree.read_text(encoding='utf-8')
    result = inspection.update_config(
        domain_tree, {'EXPERIMENT_ID': 'exp2'}, dry_run=True)
    assert result['dry_run'] is True
    assert 'backup_path' not in result
    assert domain_tree.read_text(encoding='utf-8') == before


def test_update_config_rejects_invalid_result(domain_tree, monkeypatch):
    import symfluence.core.config.models as models
    monkeypatch.setattr(models, 'SymfluenceConfig', _RejectAll)

    before = domain_tree.read_text(encoding='utf-8')
    with pytest.raises(ValueError, match='would become invalid'):
        inspection.update_config(domain_tree, {'EXPERIMENT_ID': 'exp2'})
    assert domain_tree.read_text(encoding='utf-8') == before


def test_update_config_requires_changes(domain_tree):
    with pytest.raises(ValueError, match='non-empty'):
        inspection.update_config(domain_tree, {})
