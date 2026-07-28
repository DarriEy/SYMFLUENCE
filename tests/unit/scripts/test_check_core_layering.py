# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The layering guard sees string-built (runtime) upward edges, not just imports.

``scripts/check_core_layering.py`` used to walk only ``import`` /
``from ... import`` AST nodes, so a module path assembled as a string literal or
an f-string and resolved through ``importlib.import_module`` was invisible to
it — e.g.::

    importlib.import_module(f"symfluence.models.{model.lower()}.calibration.worker")

That is a genuine runtime edge from ``core`` into the models layer, and it is
how a "removable" layer quietly becomes non-removable. These tests pin the
detector's behaviour, including the false-positive filters that keep prose out
of the scan.
"""
from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
GUARD_PATH = REPO_ROOT / "scripts" / "check_core_layering.py"


@pytest.fixture(scope="module")
def guard():
    spec = importlib.util.spec_from_file_location("_check_core_layering_str", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _scan(guard, tmp_path: Path, source: str, name: str = "probe.py"):
    root = tmp_path / "fakecore"
    root.mkdir(exist_ok=True)
    target = root / name
    target.write_text(textwrap.dedent(source), encoding="utf-8")
    return guard._scan_file(target, root)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_detects_plain_string_module_path(guard, tmp_path):
    violations = _scan(guard, tmp_path, '''
        import importlib

        def load():
            return importlib.import_module("symfluence.models.fuse.runner")
    ''')
    assert [(v.module, v.deferred, v.string_built) for v in violations] == [
        ("symfluence.models.fuse.runner", True, True)
    ]


def test_detects_fstring_literal_prefix(guard, tmp_path):
    """``f"symfluence.models.{x}..."`` has a constant leading segment."""
    violations = _scan(guard, tmp_path, '''
        import importlib

        def load(model):
            return importlib.import_module(
                f"symfluence.models.{model.lower()}.calibration.worker"
            )
    ''')
    assert len(violations) == 1
    assert violations[0].string_built is True
    assert violations[0].module.startswith("symfluence.models")


def test_module_level_string_reports_module_level_severity(guard, tmp_path):
    violations = _scan(guard, tmp_path, '''
        _LAZY = {"Thing": "symfluence.optimization.workers.thing"}
    ''')
    assert len(violations) == 1
    assert violations[0].deferred is False
    assert violations[0].string_built is True
    assert "MODULE-LEVEL" in violations[0].describe()
    assert "string-built import" in violations[0].describe()


def test_static_import_is_not_labelled_string_built(guard, tmp_path):
    violations = _scan(guard, tmp_path, '''
        from symfluence.models.fuse import runner
    ''')
    assert len(violations) == 1
    assert violations[0].string_built is False
    assert "string-built" not in violations[0].describe()


# ---------------------------------------------------------------------------
# False-positive filters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source", [
    # Prose mentioning a module is not an edge.
    '"""See symfluence.models.fuse for the runner."""',
    'MSG = "no runner registered; install symfluence.models.fuse"',
    # A path into core itself is not an upper layer.
    'PATH = "symfluence.core.modeling.base"',
    # Not a symfluence path at all.
    'PATH = "numpy.linalg"',
    # f-string that starts dynamic tells us nothing.
    'X = f"{base}.preprocessor.Thing"',
])
def test_non_edges_are_ignored(guard, tmp_path, source):
    assert _scan(guard, tmp_path, source) == []


def test_fstring_into_core_is_ignored(guard, tmp_path):
    assert _scan(guard, tmp_path, 'X = f"symfluence.core.{name}.base"') == []


# ---------------------------------------------------------------------------
# Allow-list mechanism
# ---------------------------------------------------------------------------

def test_allow_list_covers_string_built_edges_at_both_severities(guard, tmp_path, monkeypatch):
    """A module-level string literal executes nothing, so it stays allow-listable.

    It is data in a lazy-import map that some ``__getattr__`` resolves at call
    time — not import-time coupling — which is why the allow-list governs
    string-built edges wherever the literal sits.
    """
    source = '''
        _LAZY = {"Thing": "symfluence.models.fuse.calibration.targets"}

        def load():
            import importlib
            return importlib.import_module("symfluence.models.fuse.calibration.targets")
    '''
    assert len(_scan(guard, tmp_path, source)) == 2

    monkeypatch.setattr(guard, "ALLOWED_DEFERRED", [
        ("probe.py", "symfluence.models", "test allowance"),
    ])
    assert _scan(guard, tmp_path, source) == []


# ---------------------------------------------------------------------------
# Live tree
# ---------------------------------------------------------------------------

def test_guard_passes_on_the_real_tree(guard):
    """Every string-built edge in the tree is either absent or justified."""
    violations = guard.find_violations() + guard.find_boundary_violations()
    assert violations == [], "\n".join(v.describe() for v in violations)


def test_find_violations_detects_a_string_built_edge_in_a_tree(guard, tmp_path):
    """``find_violations`` — not just ``_scan_file`` — sees a string-built edge.

    ``test_guard_passes_on_the_real_tree`` above can only ever assert an empty
    list, so a ``find_violations`` that returned ``[]`` unconditionally would
    satisfy it. This drives the same entry point over a tree with a known edge
    planted in it, so the empty result on the real tree means something.
    """
    root = tmp_path / "synthetic_core"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "uniquely_named_probe_module.py").write_text(
        textwrap.dedent('''
            import importlib

            def load():
                return importlib.import_module("symfluence.models.fuse.runner")
        '''),
        encoding="utf-8",
    )
    (root / "innocent.py").write_text("VALUE = 1\n", encoding="utf-8")

    violations = guard.find_violations(root)

    assert [(v.module, v.deferred, v.string_built) for v in violations] == [
        ("symfluence.models.fuse.runner", True, True)
    ]


def test_the_real_trees_allowances_are_load_bearing(guard, monkeypatch):
    """Stripping the allow-list must expose string-built edges in the REAL tree.

    Guards against the detector silently regressing to a no-op. The previous
    version of this test grepped the free-text *reason* strings of
    ``ALLOWED_DEFERRED`` for words like "importlib" — prose that stays true
    however thoroughly the scan is neutered, which is exactly what happened
    when the detector was gutted and this test stayed green. What matters is
    that running the scan over ``src/symfluence/core`` with nothing allowed
    still finds string-built edges: if it does not, either the tree genuinely
    has none left (remove this test and the allowances deliberately) or the
    scan stopped working.
    """
    monkeypatch.setattr(guard, "ALLOWED_DEFERRED", [])

    violations = guard.find_violations()
    string_built = [v for v in violations if v.string_built]

    assert string_built, (
        "with the allow-list emptied, the scan of src/symfluence/core found no "
        "string-built upward edges at all — the string scan is a no-op, or the "
        "tree no longer contains the edges the allow-list exists for"
    )


# ---------------------------------------------------------------------------
# Coupling layer coverage
# ---------------------------------------------------------------------------

def test_coupling_is_a_guarded_upper_layer(guard):
    """``coupling`` must be scanned, not exempt by omission.

    It was absent from UPPER_LAYERS for a long time, so core's edges into the
    dCoupler stack were invisible *by configuration* rather than because the
    scan could not see them — the ALLOWED_DEFERRED entry for
    ``modeling/coupling.py`` sat inert as a result.
    """
    assert "coupling" in guard.UPPER_LAYERS


def test_a_new_core_to_coupling_edge_is_caught(guard, tmp_path):
    """The existing edges are allow-listed; an unlisted one must still fail."""
    violations = _scan(guard, tmp_path, '''
        from symfluence.coupling.graph_builder import GraphBuilder
    ''', name="brand_new_module.py")
    assert violations, "a module-level core -> coupling import was not flagged"
    assert not violations[0].deferred
    assert "symfluence.coupling" in violations[0].module


def test_bootstrap_bmi_adapter_declarations_are_allow_listed(guard):
    """The 13 lazy BMI adapter paths are declarations, not imports.

    They are registered with ``R.bmi_adapters.add_lazy``, so the coupling layer
    is imported only when a graph resolves an adapter. They must therefore be
    visible to the guard and explicitly allowed — not silently unscanned.
    """
    assert any(
        path.endswith("_bootstrap.py") and prefix == "symfluence.coupling.adapters"
        for path, prefix, _reason in guard.ALLOWED_DEFERRED
    ), "the BMI adapter seam lost its explicit allowance"
