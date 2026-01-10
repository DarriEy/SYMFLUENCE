"""
Data structures for discretization artifacts and results.

Defines dataclasses for storing geospatial discretization outputs
including HRU paths and processing metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union


@dataclass
class DiscretizationArtifacts:
    method: str
    hru_paths: Optional[Union[Path, Dict[str, Path]]] = None
    metadata: Dict[str, str] = field(default_factory=dict)
