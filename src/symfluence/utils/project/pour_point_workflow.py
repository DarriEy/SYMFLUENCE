from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from symfluence.utils.cli.binary_manager import BinaryManager
from symfluence.utils.exceptions import ConfigurationError, FileOperationError


@dataclass(frozen=True)
class PourPointWorkflowResult:
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
    bounding_box_coords, used_auto_bbox = _parse_coordinates_and_bbox(
        coordinates,
        bounding_box_coords,
    )

    resolved_template = _resolve_template_path(template_path, symfluence_code_dir)
    config = _load_config(resolved_template)

    binary_manager = BinaryManager()
    config = binary_manager._ensure_valid_config_paths(config, resolved_template)

    config['DOMAIN_NAME'] = domain_name
    config['POUR_POINT_COORDS'] = coordinates
    config['DOMAIN_DEFINITION_METHOD'] = domain_def_method
    config['BOUNDING_BOX_COORDS'] = bounding_box_coords

    if 'EXPERIMENT_ID' not in config or config['EXPERIMENT_ID'] == 'run_1':
        config['EXPERIMENT_ID'] = 'pour_point_setup'

    output_dir = output_dir or Path("0_config_files")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_config_path = output_dir / f"config_{domain_name}.yaml"

    with open(output_config_path, 'w') as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False)

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
    if template_path:
        resolved = Path(template_path)
        if resolved.exists():
            return resolved
        raise FileOperationError(f"Config template not found at: {resolved}")

    possible_locations = [
        Path("./0_config_files/config_template.yaml"),
        Path("../0_config_files/config_template.yaml"),
        Path("../../0_config_files/config_template.yaml"),
    ]

    if symfluence_code_dir:
        possible_locations.insert(
            0,
            Path(symfluence_code_dir) / "0_config_files" / "config_template.yaml",
        )

    for location in possible_locations:
        if location.exists():
            return location

    raise FileOperationError(
        "Config template not found. Tried: "
        + ", ".join(str(path) for path in possible_locations)
    )


def _load_config(template_path: Path) -> Dict[str, Any]:
    try:
        with open(template_path, 'r') as handle:
            return yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise FileOperationError(
            f"Config template not found at: {template_path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Failed to parse config template: {template_path}"
        ) from exc
