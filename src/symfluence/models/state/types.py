# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
State management data types.

Core data structures for model state save/restore operations:
- StateFormat: Enumeration of supported state storage formats
- StateMetadata: Immutable metadata describing a saved state
- ModelState: Container for state data (files and/or arrays)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class StateFormat(Enum):
    """Supported state storage formats."""
    FILE_NETCDF = auto()
    FILE_BINARY = auto()
    FILE_TEXT = auto()
    MEMORY_ARRAY = auto()
    MEMORY_DICT = auto()
    COMPOSITE = auto()


@dataclass(frozen=True)
class StateMetadata:
    """Immutable metadata describing a saved model state.

    Attributes:
        model_name: Name of the model that produced this state.
        timestamp: ISO-format timestamp of the state snapshot.
        format: Storage format used for the state data.
        variables: List of state variable names.
        shape: Optional shape information per variable.
        ensemble_member: Optional ensemble member index.
        extra: Arbitrary extra metadata.
    """
    model_name: str
    timestamp: str
    format: StateFormat
    variables: List[str] = field(default_factory=list)
    shape: Optional[Dict[str, Any]] = None
    ensemble_member: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelState:
    """Container for model state data.

    Holds both file-based state (paths to state files) and memory-based
    state (numpy arrays keyed by variable name). A state may use one or
    both depending on the model's StateFormat.

    Attributes:
        metadata: Immutable metadata describing the state.
        files: List of file paths for file-based states.
        arrays: Dictionary of numpy arrays for memory-based states.
    """
    metadata: StateMetadata
    files: List[Path] = field(default_factory=list)
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def is_file_based(self) -> bool:
        """True if this state contains file-based data."""
        return len(self.files) > 0

    @property
    def is_memory_based(self) -> bool:
        """True if this state contains in-memory array data."""
        return len(self.arrays) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for transport via WorkerTask/WorkerResult additional_data.

        Returns:
            Dictionary with serializable state representation.
        """
        return {
            'metadata': {
                'model_name': self.metadata.model_name,
                'timestamp': self.metadata.timestamp,
                'format': self.metadata.format.name,
                'variables': self.metadata.variables,
                'shape': self.metadata.shape,
                'ensemble_member': self.metadata.ensemble_member,
                'extra': self.metadata.extra,
            },
            'files': [str(f) for f in self.files],
            'arrays': {k: v.tolist() for k, v in self.arrays.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelState':
        """Reconstruct a ModelState from a serialized dictionary.

        Args:
            data: Dictionary produced by to_dict().

        Returns:
            Reconstructed ModelState instance.
        """
        meta = data['metadata']
        metadata = StateMetadata(
            model_name=meta['model_name'],
            timestamp=meta['timestamp'],
            format=StateFormat[meta['format']],
            variables=meta.get('variables', []),
            shape=meta.get('shape'),
            ensemble_member=meta.get('ensemble_member'),
            extra=meta.get('extra', {}),
        )
        files = [Path(f) for f in data.get('files', [])]
        arrays = {k: np.array(v) for k, v in data.get('arrays', {}).items()}
        return cls(metadata=metadata, files=files, arrays=arrays)
