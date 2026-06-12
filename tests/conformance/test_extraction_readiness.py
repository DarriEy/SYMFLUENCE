# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Extraction-readiness guard (design §5).

The contract module must be extractable into a standalone
``symfluence-data-interface`` package: stdlib(+pydantic) only, with **no
``symfluence.*`` import occurring directly or transitively**. Proven by
executing the module file in an isolated subprocess (``python -I``: stripped
``sys.path``, no env hooks) with a meta-path finder that records and blocks
every attempted ``symfluence`` import.

``errors.py`` ships alongside the contract at extraction time; its framework
base class (``DataAcquisitionError``) is a *guarded* import, so it must load
with symfluence blocked (taxonomy degrades to plain ``Exception``) while
in-tree it subclasses ``DataAcquisitionError``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import symfluence.data.backends.contract as contract_module
import symfluence.data.backends.errors as errors_module

pytestmark = [pytest.mark.unit]

_CONTRACT_PATH = Path(contract_module.__file__)
_ERRORS_PATH = Path(errors_module.__file__)

# Loads a module from a file path with all symfluence imports blocked by a
# recording meta-path finder, then runs assertions passed via argv.
_ISOLATED_LOADER = r'''
import importlib.util
import sys

attempts = []


class _SymfluenceBlocker:
    def find_spec(self, name, path=None, target=None):
        if name == "symfluence" or name.startswith("symfluence."):
            attempts.append(name)
            raise ImportError(f"blocked symfluence import: {name}")
        return None


sys.meta_path.insert(0, _SymfluenceBlocker())

module_path, mode = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("_isolated_module", module_path)
module = importlib.util.module_from_spec(spec)
# dataclasses' KW_ONLY detection dereferences sys.modules[cls.__module__].
sys.modules["_isolated_module"] = module
spec.loader.exec_module(module)
del sys.modules["_isolated_module"]

assert not any(
    name == "symfluence" or name.startswith("symfluence.") for name in sys.modules
), "symfluence leaked into sys.modules"

if mode == "contract":
    # HARD CONSTRAINT: zero symfluence import attempts, even transitively.
    assert attempts == [], f"contract module attempted symfluence imports: {attempts}"
    for name in (
        "PROTOCOL_VERSION", "GridClass", "SchemaId", "CredentialContext",
        "DatasetCapability", "AcquisitionRequest", "AcquisitionResult",
        "AcquisitionBackend", "build_manifest", "validate_manifest",
    ):
        assert hasattr(module, name), f"missing public API: {name}"
    # The types must be constructible without the framework.
    cap = module.DatasetCapability(
        dataset_id="X", grid_class=module.GridClass.REGULAR_LATLON,
        schema=module.SchemaId.NATIVE_RAW, variables=frozenset({"air_temperature"}),
        temporal=None, auth=frozenset(), parity_grade=None,
    )
    assert cap.schema.value == "native-raw"
elif mode == "errors":
    # The guarded framework-base import is the only permitted attempt (the
    # blocker fires on the parent "symfluence" package first); with symfluence
    # blocked the taxonomy must still load on a plain Exception base.
    assert attempts, "errors module should attempt the guarded framework-base import"
    allowed = {"symfluence", "symfluence.core", "symfluence.core.exceptions"}
    assert set(attempts) <= allowed, attempts
    assert issubclass(module.AcquisitionError, Exception)
    for name in (
        "DatasetUnsupported", "AuthRequired", "WindowOutOfRange",
        "UpstreamOutage", "IntegrityError",
    ):
        assert issubclass(getattr(module, name), module.AcquisitionError), name
else:
    raise SystemExit(f"unknown mode {mode!r}")

print("ISOLATED-IMPORT-OK")
'''


def _run_isolated(module_path: Path, mode: str) -> None:
    proc = subprocess.run(
        [sys.executable, '-I', '-c', _ISOLATED_LOADER, str(module_path), mode],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, (
        f'isolated import of {module_path.name} ({mode}) failed:\n'
        f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}'
    )
    assert 'ISOLATED-IMPORT-OK' in proc.stdout


def test_contract_module_has_no_symfluence_imports_transitively():
    _run_isolated(_CONTRACT_PATH, 'contract')


def test_errors_module_loads_standalone_with_symfluence_blocked():
    _run_isolated(_ERRORS_PATH, 'errors')


def test_errors_subclass_data_acquisition_error_in_tree():
    """In-tree, the taxonomy keeps every existing catch site working."""
    from symfluence.core.exceptions import DataAcquisitionError

    assert issubclass(errors_module.AcquisitionError, DataAcquisitionError)
    assert issubclass(errors_module.IntegrityError, DataAcquisitionError)
