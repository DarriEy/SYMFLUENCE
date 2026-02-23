"""
Pour point workflow setup for domain initialization.

Provides functionality to create a new SYMFLUENCE domain from pour point
coordinates, generating configuration files and defining initial workflow steps
for watershed delineation and discretization.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from symfluence.cli.binary_service import BinaryManager
from symfluence.core.exceptions import ConfigurationError, FileOperationError


@dataclass(frozen=True)
class PourPointWorkflowResult:
    """Result container for pour point workflow setup.

    Attributes:
        config_file: Path to generated configuration YAML file.
        coordinates: Pour point coordinates as 'lat/lon' string.
        domain_name: Name assigned to the new domain.
        domain_definition_method: Method used for domain definition.
        bounding_box_coords: Bounding box as 'lat_max/lon_min/lat_min/lon_max'.
        template_used: Path to the configuration template that was used.
        setup_steps: List of workflow steps to execute after setup.
        used_auto_bounding_box: True if bounding box was auto-generated.
    """
    config_file: Path
    coordinates: str
    domain_name: str
    domain_definition_method: str
    bounding_box_coords: str
    template_used: Path
    setup_steps: List[str]
    used_auto_bounding_box: bool


def setup_pour_point_workflow(
    coordinates: str,
    domain_def_method: str,
    domain_name: str,
    bounding_box_coords: Optional[str] = None,
    symfluence_code_dir: Optional[str] = None,
    output_dir: Optional[Path] = None,
    template_path: Optional[Path] = None,
) -> PourPointWorkflowResult:
    """Set up a new domain from pour point coordinates.

    Creates a configuration file for a new watershed domain based on pour point
    coordinates. Optionally auto-generates a bounding box if not provided.

    Args:
        coordinates: Pour point as 'latitude/longitude' string.
        domain_def_method: Domain definition method (e.g., 'subset', 'lumped').
        domain_name: Identifier for the new domain.
        bounding_box_coords: Optional bounding box as 'lat_max/lon_min/lat_min/lon_max'.
            If None, auto-generates ±0.5° around pour point.
        symfluence_code_dir: Optional path to SYMFLUENCE installation.
        output_dir: Directory for output config file. Defaults to current directory.
        template_path: Optional path to config template. Falls back to package default.

    Returns:
        PourPointWorkflowResult with paths and configuration details.

    Raises:
        ConfigurationError: If coordinates format is invalid.
        FileOperationError: If template file cannot be found.
    """
    from symfluence.core.config.models import SymfluenceConfig

    bounding_box_coords, used_auto_bbox = _parse_coordinates_and_bbox(
        coordinates,
        bounding_box_coords,
    )

    resolved_template = _resolve_template_path(template_path, symfluence_code_dir)
    # Use SymfluenceConfig to load template
    config_obj = SymfluenceConfig.from_file(resolved_template)
    config = config_obj.to_dict(flatten=True)

    binary_manager = BinaryManager()
    config = binary_manager._ensure_valid_config_paths(config, resolved_template)

    config['DOMAIN_NAME'] = domain_name
    config['POUR_POINT_COORDS'] = coordinates
    config['DOMAIN_DEFINITION_METHOD'] = domain_def_method
    config['BOUNDING_BOX_COORDS'] = bounding_box_coords

    if 'EXPERIMENT_ID' not in config or config['EXPERIMENT_ID'] == 'run_1':
        config['EXPERIMENT_ID'] = 'pour_point_setup'

    # Re-validate with new values
    final_config_obj = SymfluenceConfig(**config)

    output_dir = output_dir or Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_config_path = output_dir / f"config_{domain_name}.yaml"

    with open(output_config_path, 'w', encoding='utf-8') as handle:
        yaml.dump(final_config_obj.to_dict(flatten=True), handle, default_flow_style=False, sort_keys=False)

    return PourPointWorkflowResult(
        config_file=output_config_path.resolve(),
        coordinates=coordinates,
        domain_name=domain_name,
        domain_definition_method=domain_def_method,
        bounding_box_coords=bounding_box_coords,
        template_used=resolved_template,
        setup_steps=[
            'setup_project',
            'create_pour_point',
            'define_domain',
            'discretize_domain',
        ],
        used_auto_bounding_box=used_auto_bbox,
    )


def _parse_coordinates_and_bbox(
    coordinates: str,
    bounding_box_coords: Optional[str],
) -> Tuple[str, bool]:
    """Parse coordinates and resolve bounding box.

    If bounding_box_coords is None, auto-generates a ±0.5° box around the pour point.

    Returns:
        Tuple of (bounding_box_string, was_auto_generated).
    """
    try:
        lat, lon = map(float, coordinates.split('/'))
    except ValueError as exc:
        raise ConfigurationError(
            f"Invalid pour point coordinates format: {coordinates}. Expected 'lat/lon'."
        ) from exc

    if bounding_box_coords:
        return bounding_box_coords, False

    lat_max = lat + 0.5
    lat_min = lat - 0.5
    lon_max = lon + 0.5
    lon_min = lon - 0.5
    auto_bbox = f"{lat_max}/{lon_min}/{lat_min}/{lon_max}"

    return auto_bbox, True


def _resolve_template_path(
    template_path: Optional[Path],
    symfluence_code_dir: Optional[str],
) -> Path:
    """Resolve configuration template path with fallback search.

    Checks explicit path, then local directories, then package resources.

    Returns:
        Resolved Path to an existing template file.

    Raises:
        FileOperationError: If no template can be found.
    """
    if template_path:
        resolved = Path(template_path)
        if resolved.exists():
            return resolved
        raise FileOperationError(f"Config template not found at: {resolved}")

    # Try local paths first for backward compatibility
    possible_locations = [
        Path("./config_template.yaml"),
        Path("../config_template.yaml"),
    ]

    for location in possible_locations:
        if location.exists():
            return location

    # Fallback to package template
    try:
        from symfluence.resources import get_config_template
        return get_config_template()
    except FileNotFoundError:
        raise FileOperationError(
            "Config template not found. Tried local paths and package template."
        ) from None
