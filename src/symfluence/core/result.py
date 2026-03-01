# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Result type for operations that can fail with validation errors.

This module provides a Result[T] pattern for handling validation and
operation outcomes in a composable, type-safe way. It replaces the
tuple (bool, Optional[str]) pattern used throughout the codebase.
"""

from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar, cast

T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True)
class ValidationError:
    """
    Single validation error with context.

    Attributes:
        field: Name of the field or parameter that failed validation
        message: Human-readable error message
        value: The invalid value (optional)
        suggestion: Suggested fix or valid value (optional)
    """

    field: str
    message: str
    value: Optional[Any] = None
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        """Format error for display."""
        result = f"{self.field}: {self.message}"
        if self.value is not None:
            result += f" (got: {self.value!r})"
        if self.suggestion:
            result += f" - {self.suggestion}"
        return result


@dataclass(frozen=True)
class Result(Generic[T]):
    """
    Result type for operations that can fail with validation errors.

    A Result can be either successful (containing a value) or failed
    (containing one or more validation errors). This pattern provides
    explicit error handling without exceptions.

    Example:
        >>> def validate_port(port: int) -> Result[int]:
        ...     if 0 <= port <= 65535:
        ...         return Result.ok(port)
        ...     return Result.err(ValidationError(
        ...         field="port",
        ...         message="Port must be 0-65535",
        ...         value=port
        ...     ))
        >>>
        >>> result = validate_port(8080)
        >>> if result.is_ok:
        ...     print(f"Using port {result.unwrap()}")
        >>> else:
        ...     print(result.format_errors())
    """

    value: Optional[T] = None
    errors: tuple[ValidationError, ...] = ()

    @property
    def is_ok(self) -> bool:
        """Return True if this Result is successful."""
        return len(self.errors) == 0

    @property
    def is_err(self) -> bool:
        """Return True if this Result contains errors."""
        return len(self.errors) > 0

    def unwrap(self) -> T:
        """
        Get the success value, raising ValueError if errors exist.

        Returns:
            The success value

        Raises:
            ValueError: If the Result contains errors
        """
        if self.errors:
            raise ValueError(f"Unwrap called on error Result:\n{self.format_errors()}")
        if self.value is None:
            raise ValueError("Result value is None despite no errors")
        return cast(T, self.value)

    def unwrap_or(self, default: T) -> T:
        """
        Get the success value or return a default.

        Args:
            default: Value to return if Result contains errors

        Returns:
            The success value or the default
        """
        if self.is_ok:
            if self.value is None:
                raise ValueError("Result value is None despite no errors")
            return cast(T, self.value)
        return default

    def unwrap_or_else(self, f: Callable[[], T]) -> T:
        """
        Get the success value or compute a default.

        Args:
            f: Function to call if Result contains errors

        Returns:
            The success value or the computed default
        """
        if self.is_ok:
            if self.value is None:
                raise ValueError("Result value is None despite no errors")
            return cast(T, self.value)
        return f()

    def map(self, f: Callable[[T], U]) -> "Result[U]":
        """
        Apply a function to the success value.

        Args:
            f: Function to apply

        Returns:
            New Result with transformed value, or same errors
        """
        if self.is_ok:
            if self.value is None:
                raise ValueError("Result value is None despite no errors")
            return Result.ok(f(cast(T, self.value)))
        return Result(errors=self.errors)

    def format_errors(self, prefix: str = "  - ") -> str:
        """
        Format all errors for display.

        Args:
            prefix: String to prefix each error line

        Returns:
            Formatted error string
        """
        return "\n".join(f"{prefix}{error}" for error in self.errors)

    def first_error(self) -> Optional[ValidationError]:
        """
        Get the first error, if any.

        Returns:
            First ValidationError or None if successful
        """
        return self.errors[0] if self.errors else None

    @classmethod
    def ok(cls, value: T) -> "Result[T]":
        """
        Create a successful Result.

        Args:
            value: The success value

        Returns:
            Result containing the value
        """
        return cls(value=value)

    @classmethod
    def err(cls, *errors: ValidationError) -> "Result[T]":
        """
        Create a failed Result.

        Args:
            *errors: One or more ValidationError instances

        Returns:
            Result containing the errors
        """
        return cls(errors=tuple(errors))

    @classmethod
    def from_optional(
        cls,
        value: Optional[T],
        error: ValidationError,
    ) -> "Result[T]":
        """
        Create a Result from an optional value.

        Args:
            value: Optional value
            error: Error to use if value is None

        Returns:
            Result.ok(value) if value is not None, else Result.err(error)
        """
        if value is not None:
            return cls.ok(value)
        return cls.err(error)

    @classmethod
    def from_legacy(
        cls,
        is_valid: bool,
        error_msg: Optional[str],
        field: str = "validation",
        value: Optional[T] = None,
    ) -> "Result[T]":
        """
        Create a Result from legacy (bool, error_msg) tuple pattern.

        This is a bridge method for migrating from the old validation pattern.

        Args:
            is_valid: Whether validation passed
            error_msg: Error message if validation failed
            field: Field name for the error
            value: Optional success value

        Returns:
            Result based on the legacy validation outcome
        """
        if is_valid:
            # Cast handles the case where value may be None but user expects Result[T]
            return cls.ok(cast(T, value))
        return cls.err(
            ValidationError(field=field, message=error_msg or "Validation failed")
        )


def collect_results(results: list["Result[T]"]) -> "Result[list[T]]":
    """
    Collect multiple Results into a single Result.

    If all Results are successful, returns Result.ok with list of values.
    If any Result has errors, returns Result.err with all errors combined.

    Args:
        results: List of Results to collect

    Returns:
        Combined Result
    """
    all_errors: list[ValidationError] = []
    values: list[T] = []

    for result in results:
        if result.is_err:
            all_errors.extend(result.errors)
        else:
            values.append(result.unwrap())

    if all_errors:
        return Result.err(*all_errors)
    return Result.ok(values)
