# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
State-capable model mixin.

Provides an opt-in mixin that model runners can inherit to declare
support for state save/restore operations. Models implement the abstract
methods to define how their internal state is persisted and loaded.
"""

from abc import abstractmethod
from pathlib import Path
from typing import List, Optional

from .types import ModelState, StateFormat


class StateCapableMixin:
    """Opt-in mixin for model runners that support state save/restore.

    Models inheriting this mixin declare ``supports_state = True`` and
    must implement the three abstract methods. Optional hooks provide
    defaults that can be overridden for model-specific behavior.

    Example::

        class MyRunner(BaseModelRunner, StateCapableMixin):
            def get_state_format(self):
                return StateFormat.FILE_NETCDF

            def save_state(self, target_dir, timestamp, ensemble_member=None):
                ...

            def load_state(self, state):
                ...
    """

    supports_state: bool = True

    # ------------------------------------------------------------------
    # Abstract methods — must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def get_state_format(self) -> StateFormat:
        """Return the native state storage format for this model."""
        ...

    @abstractmethod
    def save_state(
        self,
        target_dir: Path,
        timestamp: str,
        ensemble_member: Optional[int] = None,
    ) -> ModelState:
        """Save the current model state.

        Args:
            target_dir: Directory to write state files into.
            timestamp: ISO-format timestamp of the state snapshot.
            ensemble_member: Optional ensemble member index.

        Returns:
            A ModelState describing the saved state.
        """
        ...

    @abstractmethod
    def load_state(self, state: ModelState) -> None:
        """Restore model state from a previously saved ModelState.

        Args:
            state: The ModelState to restore.
        """
        ...

    # ------------------------------------------------------------------
    # Optional overrides with sensible defaults
    # ------------------------------------------------------------------

    def get_state_variables(self) -> List[str]:
        """Return the list of state variable names (default: empty)."""
        return []

    def get_state_directory(self) -> Optional[Path]:
        """Return the model's default state directory (default: None)."""
        return None

    def get_state_file_pattern(self) -> str:
        """Return a glob pattern for state files (default: ``*.nc``)."""
        return "*.nc"

    def get_state_size(self) -> Optional[int]:
        """Return the total number of state elements (default: None).

        Useful for dCoupler JAX models where state is a flat array.
        """
        return None

    def supports_ensemble_state(self) -> bool:
        """Whether the model supports per-member state management."""
        return False

    def validate_state(self, state: ModelState) -> bool:
        """Validate that a state is compatible with this model.

        Default implementation checks that the model name matches.
        Override for stricter validation (e.g. variable checks).

        Args:
            state: The state to validate.

        Returns:
            True if the state is valid for this model.
        """
        expected = getattr(self, '_get_model_name', lambda: '')()
        return state.metadata.model_name == expected
