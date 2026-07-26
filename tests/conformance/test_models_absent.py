# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Prove the framework remains importable with the model suite absent."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.smoke]

_SOURCE_ROOT = Path(__file__).resolve().parents[2] / 'src'

_MODELS_ABSENT_SMOKE = r'''
import importlib.abc
import importlib.metadata
import sys

source_root = sys.argv[1]
sys.path.insert(0, source_root)

# Distribution metadata belongs to the physically installed packages. An
# installation without symfluence-models has no model entry points.
importlib.metadata.entry_points = lambda **kwargs: []

attempts = []


class _ModelsBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'symfluence.models' or fullname.startswith('symfluence.models.'):
            attempts.append(fullname)
            raise ImportError(f'blocked absent model package: {fullname}')
        return None


sys.meta_path.insert(0, _ModelsBlocker())

import symfluence
from symfluence.core.contracts import contract_version
from symfluence.core.registries import R
from symfluence.project.manager_factory import LazyManagerDict
from symfluence.project.workflow_orchestrator import WorkflowOrchestrator

assert contract_version('models')
assert R.registered_models() == []
assert attempts == [], f'framework attempted absent model imports: {attempts}'
assert not any(
    name == 'symfluence.models' or name.startswith('symfluence.models.')
    for name in sys.modules
)
assert LazyManagerDict is not None
assert WorkflowOrchestrator is not None
print('MODELS-ABSENT-OK')
'''


def test_framework_imports_with_model_distribution_physically_absent():
    proc = subprocess.run(
        [sys.executable, '-I', '-c', _MODELS_ABSENT_SMOKE, str(_SOURCE_ROOT)],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, (
        f'models-absent smoke failed:\n--- stdout ---\n{proc.stdout}'
        f'\n--- stderr ---\n{proc.stderr}'
    )
    assert 'MODELS-ABSENT-OK' in proc.stdout
