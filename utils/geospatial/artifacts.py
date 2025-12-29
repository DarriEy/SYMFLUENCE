from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union


@dataclass
class DelineationArtifacts:
    method: str
    river_basins_path: Optional[Path] = None
    river_network_path: Optional[Path] = None
    pour_point_path: Optional[Path] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class DiscretizationArtifacts:
    method: str
    hru_paths: Optional[Union[Path, Dict[str, Path]]] = None
    metadata: Dict[str, str] = field(default_factory=dict)
