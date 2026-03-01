# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
State vector assembly and disassembly.

Handles conversion between per-member dictionaries of named arrays
and the flat (n_members, n_state) matrices used by EnKFAlgorithm.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StateVariableSpec:
    """Specification for a single state variable.

    Attributes:
        name: Variable name (must match dict keys in member states).
        size: Number of elements for this variable.
        lower_bound: Minimum physical value (None = no bound).
        upper_bound: Maximum physical value (None = no bound).
    """
    name: str
    size: int = 1
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class StateVector:
    """Assembles/disassembles state vectors for EnKF.

    Maps between per-member dictionaries ``{'snow': array, 'sm': array, ...}``
    and flat 2-D matrices ``(n_members, n_state)`` used by the EnKF algorithm.

    Args:
        specs: Ordered list of StateVariableSpec defining the state layout.
    """

    def __init__(self, specs: List[StateVariableSpec]):
        self.specs = specs
        self._offsets: List[Tuple[int, int]] = []

        offset = 0
        for spec in specs:
            self._offsets.append((offset, offset + spec.size))
            offset += spec.size

        self.n_state = offset

    def assemble(self, member_states: List[Dict[str, np.ndarray]]) -> np.ndarray:
        """Build a (n_members, n_state) matrix from per-member dictionaries.

        Args:
            member_states: List of dicts, one per ensemble member.

        Returns:
            State matrix of shape (n_members, n_state).
        """
        n_members = len(member_states)
        X = np.zeros((n_members, self.n_state))

        for i, state in enumerate(member_states):
            for spec, (start, end) in zip(self.specs, self._offsets):
                val = state[spec.name]
                X[i, start:end] = np.atleast_1d(val).ravel()[:end - start]

        return X

    def disassemble(self, state_matrix: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """Split a (n_members, n_state) matrix back into per-member dicts.

        Args:
            state_matrix: State matrix of shape (n_members, n_state).

        Returns:
            List of dictionaries, one per ensemble member.
        """
        n_members = state_matrix.shape[0]
        members = []

        for i in range(n_members):
            state = {}
            for spec, (start, end) in zip(self.specs, self._offsets):
                val = state_matrix[i, start:end]
                state[spec.name] = val[0] if spec.size == 1 else val
            members.append(state)

        return members

    def enforce_bounds(self, state_matrix: np.ndarray) -> np.ndarray:
        """Clip state variables to their physical bounds.

        Args:
            state_matrix: State matrix of shape (n_members, n_state).

        Returns:
            Clipped state matrix.
        """
        X = state_matrix.copy()
        for spec, (start, end) in zip(self.specs, self._offsets):
            if spec.lower_bound is not None:
                X[:, start:end] = np.maximum(X[:, start:end], spec.lower_bound)
            if spec.upper_bound is not None:
                X[:, start:end] = np.minimum(X[:, start:end], spec.upper_bound)
        return X

    def augment_with_predictions(
        self,
        state_matrix: np.ndarray,
        predictions: np.ndarray,
    ) -> np.ndarray:
        """Append predicted observables to the state vector.

        Used for state augmentation in the EnKF, where predicted
        observations are appended to the state vector so the analysis
        step can update both states and predictions simultaneously.

        Args:
            state_matrix: State matrix (n_members, n_state).
            predictions: Predicted observations (n_members, n_obs) or (n_members,).

        Returns:
            Augmented matrix (n_members, n_state + n_obs).
        """
        if predictions.ndim == 1:
            predictions = predictions[:, np.newaxis]
        return np.hstack([state_matrix, predictions])

    def split_augmented(
        self,
        augmented: np.ndarray,
        n_obs: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split an augmented state matrix back into state and predictions.

        Args:
            augmented: Augmented matrix (n_members, n_state + n_obs).
            n_obs: Number of observation variables appended.

        Returns:
            Tuple of (state_matrix, predictions).
        """
        return augmented[:, :-n_obs], augmented[:, -n_obs:]
