# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Self-test for scripts/check_model_raises.py (review item 11 guard)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_GUARD_PATH = Path(__file__).resolve().parents[3] / "scripts" / "check_model_raises.py"
_spec = importlib.util.spec_from_file_location("check_model_raises", _GUARD_PATH)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


def _scan(tmp_path, body: str) -> list[str]:
    (tmp_path / "mod.py").write_text(body, encoding="utf-8")
    return guard.collect_matches(tmp_path)


def test_flags_generic_builtin_raise(tmp_path):
    matches = _scan(tmp_path, "def f():\n    raise ValueError('bad')\n")
    assert len(matches) == 1 and "ValueError" in matches[0]


def test_ignores_custom_exception(tmp_path):
    body = (
        "from symfluence.core.exceptions import ModelExecutionError\n"
        "def f():\n    raise ModelExecutionError('bad')\n"
    )
    assert _scan(tmp_path, body) == []


def test_ignores_bare_reraise(tmp_path):
    body = "def f():\n    try:\n        g()\n    except Exception:\n        raise\n"
    assert _scan(tmp_path, body) == []


@pytest.mark.parametrize("exc", ["NotImplementedError", "StopIteration", "KeyboardInterrupt"])
def test_exempts_control_flow_exceptions(tmp_path, exc):
    assert _scan(tmp_path, f"def f():\n    raise {exc}\n") == []


@pytest.mark.parametrize(
    "exc,method",
    [("KeyError", "__getitem__"), ("AttributeError", "__getattr__"), ("IndexError", "__getitem__")],
)
def test_exempts_data_model_protocol_raises(tmp_path, exc, method):
    body = f"class C:\n    def {method}(self, k):\n        raise {exc}(k)\n"
    assert _scan(tmp_path, body) == []


def test_flags_protocol_exception_outside_protocol_method(tmp_path):
    # KeyError is only exempt inside __getitem__/__missing__, not a normal method.
    body = "class C:\n    def lookup(self, k):\n        raise KeyError(k)\n"
    assert len(_scan(tmp_path, body)) == 1


def test_context_and_format(tmp_path):
    body = "class C:\n    def m(self):\n        raise RuntimeError('x')\n"
    matches = _scan(tmp_path, body)
    assert matches and matches[0].endswith("mod.py:C.m:raise RuntimeError('x')")


def test_in_tree_adapters_are_clean():
    """The four migrated adapters contain no generic raises (the live guard target)."""
    import symfluence.models as m

    models_root = Path(m.__file__).resolve().parent
    for adapter in ("summa", "fuse", "ngen", "mizuroute"):
        hits = [r for r in guard.collect_matches(models_root / adapter)]
        assert hits == [], f"{adapter} has generic raises: {hits}"
