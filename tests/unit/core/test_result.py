"""Tests for symfluence.core.result module."""

import pytest

from symfluence.core.result import Result, ValidationError, collect_results

# =============================================================================
# ValidationError
# =============================================================================

class TestValidationError:
    """Tests for ValidationError dataclass."""

    def test_basic_creation(self):
        err = ValidationError(field="name", message="is required")
        assert err.field == "name"
        assert err.message == "is required"

    def test_with_value_and_suggestion(self):
        err = ValidationError(field="port", message="out of range", value=99999, suggestion="use 0-65535")
        assert err.value == 99999
        assert err.suggestion == "use 0-65535"

    def test_str_basic(self):
        err = ValidationError(field="x", message="bad")
        assert "x: bad" in str(err)

    def test_str_with_value(self):
        err = ValidationError(field="x", message="bad", value=42)
        s = str(err)
        assert "42" in s
        assert "got:" in s

    def test_str_with_suggestion(self):
        err = ValidationError(field="x", message="bad", suggestion="try y")
        assert "try y" in str(err)

    def test_frozen(self):
        err = ValidationError(field="x", message="bad")
        with pytest.raises(AttributeError):
            err.field = "y"

    def test_defaults_none(self):
        err = ValidationError(field="x", message="bad")
        assert err.value is None
        assert err.suggestion is None


# =============================================================================
# Result.ok / Result.err
# =============================================================================

class TestResultCreation:
    """Tests for Result creation class methods."""

    def test_ok_creates_success(self):
        r = Result.ok(42)
        assert r.is_ok
        assert not r.is_err
        assert r.value == 42

    def test_err_creates_failure(self):
        err = ValidationError(field="x", message="bad")
        r = Result.err(err)
        assert r.is_err
        assert not r.is_ok
        assert len(r.errors) == 1

    def test_err_multiple_errors(self):
        e1 = ValidationError(field="a", message="bad a")
        e2 = ValidationError(field="b", message="bad b")
        r = Result.err(e1, e2)
        assert len(r.errors) == 2

    def test_ok_empty_errors(self):
        r = Result.ok("hello")
        assert r.errors == ()


# =============================================================================
# unwrap / unwrap_or / unwrap_or_else
# =============================================================================

class TestResultUnwrap:
    """Tests for Result unwrap methods."""

    def test_unwrap_success(self):
        assert Result.ok(42).unwrap() == 42

    def test_unwrap_raises_on_error(self):
        r = Result.err(ValidationError(field="x", message="bad"))
        with pytest.raises(ValueError, match="Unwrap called on error"):
            r.unwrap()

    def test_unwrap_raises_on_none_value(self):
        r = Result(value=None, errors=())
        with pytest.raises(ValueError, match="value is None"):
            r.unwrap()

    def test_unwrap_or_returns_value_on_success(self):
        assert Result.ok(42).unwrap_or(0) == 42

    def test_unwrap_or_returns_default_on_error(self):
        r = Result.err(ValidationError(field="x", message="bad"))
        assert r.unwrap_or(0) == 0

    def test_unwrap_or_else_returns_value_on_success(self):
        assert Result.ok(42).unwrap_or_else(lambda: 0) == 42

    def test_unwrap_or_else_calls_func_on_error(self):
        r = Result.err(ValidationError(field="x", message="bad"))
        assert r.unwrap_or_else(lambda: 99) == 99


# =============================================================================
# map
# =============================================================================

class TestResultMap:
    """Tests for Result.map."""

    def test_map_transforms_success(self):
        r = Result.ok(5).map(lambda x: x * 2)
        assert r.unwrap() == 10

    def test_map_preserves_errors(self):
        err = ValidationError(field="x", message="bad")
        r = Result.err(err).map(lambda x: x * 2)
        assert r.is_err
        assert r.errors[0] == err

    def test_map_chaining(self):
        r = Result.ok(3).map(lambda x: x + 1).map(lambda x: x * 10)
        assert r.unwrap() == 40


# =============================================================================
# format_errors / first_error
# =============================================================================

class TestResultFormatting:
    """Tests for Result error formatting."""

    def test_format_errors(self):
        e1 = ValidationError(field="a", message="bad a")
        e2 = ValidationError(field="b", message="bad b")
        r = Result.err(e1, e2)
        formatted = r.format_errors()
        assert "a: bad a" in formatted
        assert "b: bad b" in formatted

    def test_format_errors_with_prefix(self):
        e = ValidationError(field="x", message="bad")
        r = Result.err(e)
        formatted = r.format_errors(prefix=">> ")
        assert formatted.startswith(">> ")

    def test_first_error_returns_first(self):
        e1 = ValidationError(field="a", message="first")
        e2 = ValidationError(field="b", message="second")
        r = Result.err(e1, e2)
        assert r.first_error() == e1

    def test_first_error_returns_none_on_success(self):
        r = Result.ok(42)
        assert r.first_error() is None


# =============================================================================
# from_optional / from_legacy
# =============================================================================

class TestResultFactories:
    """Tests for Result factory methods."""

    def test_from_optional_with_value(self):
        err = ValidationError(field="x", message="missing")
        r = Result.from_optional(42, err)
        assert r.is_ok
        assert r.unwrap() == 42

    def test_from_optional_with_none(self):
        err = ValidationError(field="x", message="missing")
        r = Result.from_optional(None, err)
        assert r.is_err
        assert r.first_error() == err

    def test_from_legacy_valid(self):
        r = Result.from_legacy(is_valid=True, error_msg=None, value=42)
        assert r.is_ok

    def test_from_legacy_invalid(self):
        r = Result.from_legacy(is_valid=False, error_msg="something broke")
        assert r.is_err
        assert "something broke" in str(r.first_error())

    def test_from_legacy_invalid_none_msg(self):
        r = Result.from_legacy(is_valid=False, error_msg=None)
        assert r.is_err
        assert "Validation failed" in str(r.first_error())


# =============================================================================
# collect_results
# =============================================================================

class TestCollectResults:
    """Tests for collect_results utility."""

    def test_all_ok(self):
        results = [Result.ok(1), Result.ok(2), Result.ok(3)]
        combined = collect_results(results)
        assert combined.is_ok
        assert combined.unwrap() == [1, 2, 3]

    def test_some_errors(self):
        e = ValidationError(field="x", message="bad")
        results = [Result.ok(1), Result.err(e), Result.ok(3)]
        combined = collect_results(results)
        assert combined.is_err
        assert len(combined.errors) == 1

    def test_all_errors(self):
        e1 = ValidationError(field="a", message="bad a")
        e2 = ValidationError(field="b", message="bad b")
        results = [Result.err(e1), Result.err(e2)]
        combined = collect_results(results)
        assert combined.is_err
        assert len(combined.errors) == 2

    def test_empty_list(self):
        combined = collect_results([])
        assert combined.is_ok
        assert combined.unwrap() == []
