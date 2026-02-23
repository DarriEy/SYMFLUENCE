"""Comprehensive tests for the unified Registry[T] class."""

from __future__ import annotations

import warnings

import pytest

from symfluence.core.registry import Registry, _LazyEntry

# ======================================================================
# Fixtures
# ======================================================================


class _DummyRunner:
    model_name = "DUMMY"

    def run(self, **kw):
        return None


class _DummyPreprocessor:
    MODEL_NAME = "DUMMY"

    def run_preprocessing(self):
        return True


@pytest.fixture()
def reg():
    """Fresh registry with UPPERCASE normalization (default)."""
    return Registry("test")


@pytest.fixture()
def lower_reg():
    """Fresh registry with lowercase normalization."""
    return Registry("test_lower", normalize=str.lower)


# ======================================================================
# add / get / __getitem__ / __contains__
# ======================================================================


class TestAddAndLookup:
    def test_add_direct(self, reg):
        reg.add("summa", _DummyRunner)
        assert reg.get("summa") is _DummyRunner
        assert reg.get("SUMMA") is _DummyRunner
        assert "SUMMA" in reg

    def test_add_decorator(self, reg):
        @reg.add("fuse")
        class FuseRunner:
            pass

        assert reg["FUSE"] is FuseRunner

    def test_get_returns_none_on_miss(self, reg):
        assert reg.get("NONEXISTENT") is None

    def test_get_returns_custom_default(self, reg):
        sentinel = object()
        assert reg.get("NONEXISTENT", sentinel) is sentinel

    def test_getitem_raises_keyerror(self, reg):
        with pytest.raises(KeyError, match="unknown key"):
            reg["NONEXISTENT"]

    def test_contains_false_on_miss(self, reg):
        assert "MISSING" not in reg

    def test_key_normalization_uppercase(self, reg):
        reg.add("lowercase_key", _DummyRunner)
        assert reg.get("LOWERCASE_KEY") is _DummyRunner

    def test_key_normalization_lowercase(self, lower_reg):
        lower_reg.add("ERA5", _DummyRunner)
        assert lower_reg.get("era5") is _DummyRunner
        assert "era5" in lower_reg


# ======================================================================
# Metadata
# ======================================================================


class TestMetadata:
    def test_meta_stored_on_add(self, reg):
        reg.add("summa", _DummyRunner, runner_method="run_summa", version=3)
        meta = reg.meta("summa")
        assert meta["runner_method"] == "run_summa"
        assert meta["version"] == 3

    def test_meta_empty_when_none(self, reg):
        reg.add("summa", _DummyRunner)
        assert reg.meta("summa") == {}

    def test_meta_missing_key(self, reg):
        assert reg.meta("MISSING") == {}


# ======================================================================
# Aliases
# ======================================================================


class TestAliases:
    def test_alias_resolves_to_canonical(self, reg):
        reg.add("SACSMA", _DummyRunner)
        reg.alias("SAC-SMA", "SACSMA")
        assert reg.get("SAC-SMA") is _DummyRunner
        assert reg["SAC-SMA"] is _DummyRunner
        assert "SAC-SMA" in reg

    def test_alias_with_normalization(self, lower_reg):
        lower_reg.add("semidistributed", _DummyRunner)
        lower_reg.alias("subset", "semidistributed")
        assert lower_reg.get("SUBSET") is _DummyRunner  # normalized to lowercase

    def test_alias_before_canonical_exists(self, reg):
        # Alias can be created before the canonical key is registered
        reg.alias("ALIAS", "CANON")
        assert reg.get("ALIAS") is None  # canonical not yet present
        reg.add("CANON", _DummyRunner)
        assert reg.get("ALIAS") is _DummyRunner


# ======================================================================
# Lazy imports
# ======================================================================


class TestLazyImports:
    def test_add_lazy_resolves_on_get(self, reg):
        # Use a class that definitely exists
        reg.add_lazy("MATH_LOG", "math.log")
        import math
        assert reg.get("MATH_LOG") is math.log

    def test_add_lazy_resolves_on_getitem(self, reg):
        reg.add_lazy("MATH_LOG", "math.log")
        import math
        assert reg["MATH_LOG"] is math.log

    def test_add_lazy_caches_after_resolve(self, reg):
        reg.add_lazy("MATH_LOG", "math.log")
        _ = reg.get("MATH_LOG")
        # Second access should not be a _LazyEntry
        assert not isinstance(reg._entries.get("MATH_LOG"), _LazyEntry)

    def test_add_lazy_bad_path_raises(self, reg):
        reg.add_lazy("BAD", "nonexistent.module.Class")
        with pytest.raises((ImportError, ModuleNotFoundError)):
            reg["BAD"]

    def test_add_lazy_with_metadata(self, reg):
        reg.add_lazy("MATH_LOG", "math.log", kind="function")
        assert reg.meta("MATH_LOG") == {"kind": "function"}


# ======================================================================
# Discovery
# ======================================================================


class TestDiscovery:
    def test_keys(self, reg):
        reg.add("B", _DummyRunner)
        reg.add("A", _DummyPreprocessor)
        assert reg.keys() == ["A", "B"]

    def test_items(self, reg):
        reg.add("A", _DummyRunner)
        items = reg.items()
        assert items == [("A", _DummyRunner)]

    def test_len(self, reg):
        assert len(reg) == 0
        reg.add("X", _DummyRunner)
        assert len(reg) == 1

    def test_iter(self, reg):
        reg.add("C", _DummyRunner)
        reg.add("A", _DummyRunner)
        assert list(reg) == ["A", "C"]

    def test_repr(self, reg):
        assert "test" in repr(reg)
        assert "0 entries" in repr(reg)

    def test_bool_always_true(self, reg):
        assert bool(reg) is True

    def test_summary(self, reg):
        reg.add("A", _DummyRunner)
        reg.alias("B", "A")
        s = reg.summary()
        assert s["name"] == "test"
        assert s["entries"] == 1
        assert s["aliases"] == 1
        assert "A" in s["keys"]


# ======================================================================
# Lifecycle: clear, freeze, remove
# ======================================================================


class TestLifecycle:
    def test_clear(self, reg):
        reg.add("A", _DummyRunner)
        reg.alias("B", "A")
        reg.clear()
        assert len(reg) == 0
        assert reg.get("A") is None
        assert reg.get("B") is None

    def test_freeze_blocks_add(self, reg):
        reg.freeze()
        with pytest.raises(RuntimeError, match="frozen"):
            reg.add("A", _DummyRunner)

    def test_freeze_blocks_add_lazy(self, reg):
        reg.freeze()
        with pytest.raises(RuntimeError, match="frozen"):
            reg.add_lazy("A", "math.log")

    def test_freeze_blocks_alias(self, reg):
        reg.freeze()
        with pytest.raises(RuntimeError, match="frozen"):
            reg.alias("A", "B")

    def test_freeze_blocks_remove(self, reg):
        reg.add("A", _DummyRunner)
        reg.freeze()
        with pytest.raises(RuntimeError, match="frozen"):
            reg.remove("A")

    def test_freeze_allows_read(self, reg):
        reg.add("A", _DummyRunner)
        reg.freeze()
        assert reg.get("A") is _DummyRunner
        assert reg["A"] is _DummyRunner
        assert "A" in reg

    def test_clear_unfreezes(self, reg):
        reg.freeze()
        reg.clear()
        # Should not raise
        reg.add("A", _DummyRunner)

    def test_remove(self, reg):
        reg.add("A", _DummyRunner, x=1)
        reg.remove("A")
        assert reg.get("A") is None
        assert reg.meta("A") == {}

    def test_remove_nonexistent_is_noop(self, reg):
        reg.remove("MISSING")  # should not raise


# ======================================================================
# Protocol validation (advisory)
# ======================================================================


class TestProtocolValidation:
    def test_protocol_warns_on_missing_attrs(self):
        from symfluence.models.base.protocols import ModelRunner

        reg = Registry("runners", protocol=ModelRunner)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            class BadRunner:
                pass

            reg.add("BAD", BadRunner)
            # Should have warned
            protocol_warnings = [x for x in w if "may not satisfy" in str(x.message)]
            assert len(protocol_warnings) >= 1

    def test_protocol_no_warn_on_conformant(self):
        from symfluence.models.base.protocols import ModelRunner

        reg = Registry("runners", protocol=ModelRunner)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reg.add("GOOD", _DummyRunner)
            protocol_warnings = [x for x in w if "may not satisfy" in str(x.message)]
            assert len(protocol_warnings) == 0

    def test_no_protocol_no_validation(self, reg):
        # reg has no protocol — should never warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            class Anything:
                pass

            reg.add("X", Anything)
            protocol_warnings = [x for x in w if "may not satisfy" in str(x.message)]
            assert len(protocol_warnings) == 0


# ======================================================================
# Edge cases
# ======================================================================


class TestEdgeCases:
    def test_overwrite_entry(self, reg):
        reg.add("A", _DummyRunner)
        reg.add("A", _DummyPreprocessor)
        assert reg["A"] is _DummyPreprocessor

    def test_alias_chains_are_single_hop(self, reg):
        reg.add("REAL", _DummyRunner)
        reg.alias("A1", "REAL")
        reg.alias("A2", "A1")
        # A2 -> A1, but A1 is not in _entries (it's only in _aliases)
        # So A2 resolves to A1 which is an alias that points to REAL,
        # but _resolve_alias only does one hop.
        assert reg.get("A2") is None  # single-hop: A2 -> A1 (not in entries)
        assert reg.get("A1") is _DummyRunner  # A1 -> REAL (in entries)

    def test_add_returns_value_in_direct_form(self, reg):
        result = reg.add("A", _DummyRunner)
        assert result is _DummyRunner

    def test_items_resolves_lazy(self, reg):
        reg.add_lazy("MATH_LOG", "math.log")
        items = reg.items()
        import math
        assert items[0] == ("MATH_LOG", math.log)
