from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

from utils.geospatial.artifacts import DiscretizationArtifacts
from utils.geospatial.discretization_utils import DomainDiscretizer


class DomainDiscretizationRunner:
    """
    Wraps domain discretization with explicit artifact tracking.
    """

    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.discretizer = DomainDiscretizer(self.config, self.logger)

    def discretize_domain(
        self,
    ) -> Tuple[Optional[Union[object, Dict[str, object]]], DiscretizationArtifacts]:
        method = self.config.get("DOMAIN_DISCRETIZATION")
        hru_paths = self.discretizer.discretize_domain()
        artifacts = DiscretizationArtifacts(method=method, hru_paths=hru_paths)
        return hru_paths, artifacts
