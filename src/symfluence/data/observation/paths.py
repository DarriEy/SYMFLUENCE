"""Shared observation output path conventions.

This module centralizes canonical and legacy observation file naming conventions
used by workflow orchestration and evaluator path resolution.
"""

from pathlib import Path
from typing import Dict, List, Sequence


def first_existing_path(candidates: Sequence[Path], default: Path | None = None) -> Path:
    """Return the first existing candidate path, else fallback to default/first."""
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if default is not None:
        return default

    if candidates:
        return candidates[0]

    raise ValueError("No path candidates were provided.")


def streamflow_observation_candidates(project_dir: Path, domain_name: str) -> List[Path]:
    """Candidate streamflow observation paths ordered by preference."""
    obs_root = project_dir / "observations" / "streamflow"
    return [
        obs_root / "preprocessed" / f"{domain_name}_streamflow_processed.csv",
        obs_root / "processed" / f"{domain_name}_streamflow_processed.csv",
        obs_root / "preprocessed" / f"{domain_name}_grdc_streamflow_processed.csv",
        obs_root / "processed" / f"{domain_name}_grdc_streamflow_processed.csv",
    ]


def snow_observation_candidates(project_dir: Path, domain_name: str, target: str | None) -> List[Path]:
    """Candidate snow observation paths ordered by target-specific preference."""
    obs_root = project_dir / "observations" / "snow"
    target_norm = (target or "").lower()

    if target_norm == "swe":
        return [
            obs_root / "swe" / "processed" / f"{domain_name}_swe_processed.csv",
            obs_root / "swe" / "preprocessed" / f"{domain_name}_swe_processed.csv",
            obs_root / "processed" / f"{domain_name}_snow_processed.csv",
            obs_root / "preprocessed" / f"{domain_name}_snow_processed.csv",
            obs_root / "preprocessed" / f"{domain_name}_canswe_swe_processed.csv",
            obs_root / "preprocessed" / f"{domain_name}_norswe_swe_processed.csv",
        ]

    if target_norm == "sca":
        return [
            obs_root / "sca" / "processed" / f"{domain_name}_sca_processed.csv",
            obs_root / "sca" / "preprocessed" / f"{domain_name}_sca_processed.csv",
            obs_root / "processed" / f"{domain_name}_modis_snow_processed.csv",
            obs_root / "preprocessed" / f"{domain_name}_modis_snow_processed.csv",
        ]

    if target_norm == "snow_depth":
        return [
            obs_root / "depth" / "processed" / f"{domain_name}_snow_depth_processed.csv",
            obs_root / "depth" / "preprocessed" / f"{domain_name}_snow_depth_processed.csv",
            obs_root / "processed" / f"{domain_name}_snow_depth_processed.csv",
            obs_root / "preprocessed" / f"{domain_name}_snow_depth_processed.csv",
        ]

    return [
        obs_root / "processed" / f"{domain_name}_snow_processed.csv",
        obs_root / "preprocessed" / f"{domain_name}_snow_processed.csv",
    ]


def soil_moisture_observation_candidates(project_dir: Path, domain_name: str, target: str | None) -> List[Path]:
    """Candidate soil moisture observation paths ordered by target."""
    obs_root = project_dir / "observations" / "soil_moisture"
    target_norm = (target or "").lower()

    if target_norm == "sm_point":
        return [
            obs_root / "point" / "processed" / f"{domain_name}_sm_processed.csv",
            obs_root / "processed" / f"{domain_name}_sm_processed.csv",
            obs_root / "preprocessed" / f"{domain_name}_sm_processed.csv",
        ]

    if target_norm == "sm_smap":
        return [
            obs_root / "smap" / "processed" / f"{domain_name}_smap_processed.csv",
            obs_root / "processed" / f"{domain_name}_smap_processed.csv",
            obs_root / "preprocessed" / f"{domain_name}_smap_processed.csv",
        ]

    if target_norm == "sm_ismn":
        return [
            obs_root / "ismn" / "processed" / f"{domain_name}_ismn_processed.csv",
            obs_root / "processed" / f"{domain_name}_ismn_processed.csv",
            obs_root / "preprocessed" / f"{domain_name}_ismn_processed.csv",
        ]

    if target_norm == "sm_esa":
        return [
            obs_root / "esa_sm" / "processed" / f"{domain_name}_esa_processed.csv",
            obs_root / "processed" / f"{domain_name}_esa_processed.csv",
            obs_root / "preprocessed" / f"{domain_name}_esa_processed.csv",
        ]

    return [
        obs_root / "processed" / f"{domain_name}_sm_processed.csv",
        obs_root / "preprocessed" / f"{domain_name}_sm_processed.csv",
    ]


def et_observation_candidates(
    project_dir: Path,
    domain_name: str,
    source: str | None,
    *,
    fluxnet_station: str = "",
) -> List[Path]:
    """Candidate ET observation paths ordered by source preference."""
    source_norm = (source or "").lower()
    et_root = project_dir / "observations" / "et"

    if source_norm in {"mod16", "modis", "modis_et", "mod16a2"}:
        return [
            et_root / "preprocessed" / f"{domain_name}_modis_et_processed.csv",
            et_root / "processed" / f"{domain_name}_modis_et_processed.csv",
        ]

    if source_norm in {"fluxcom", "fluxcom_et"}:
        return [
            et_root / "preprocessed" / f"{domain_name}_fluxcom_et_processed.csv",
            et_root / "processed" / f"{domain_name}_fluxcom_et_processed.csv",
        ]

    if source_norm == "gleam":
        return [
            et_root / "preprocessed" / f"{domain_name}_gleam_et_processed.csv",
            et_root / "processed" / f"{domain_name}_gleam_et_processed.csv",
        ]

    if source_norm == "fluxnet":
        candidates = [
            et_root / "preprocessed" / f"{domain_name}_fluxnet_et_processed.csv",
            et_root / "processed" / f"{domain_name}_fluxnet_et_processed.csv",
            project_dir / "observations" / "energy_fluxes" / "processed" / f"{domain_name}_fluxnet_processed.csv",
        ]
        if fluxnet_station:
            candidates.append(project_dir / "observations" / "fluxnet" / f"{domain_name}_FLUXNET_{fluxnet_station}.csv")
        return candidates

    if source_norm == "openet":
        return [
            et_root / "preprocessed" / f"{domain_name}_openet_et_processed.csv",
            et_root / "processed" / f"{domain_name}_openet_et_processed.csv",
        ]

    return [
        project_dir / "observations" / "energy_fluxes" / "processed" / f"{domain_name}_fluxnet_processed.csv",
    ]


def groundwater_observation_candidates(project_dir: Path, domain_name: str, target: str | None) -> List[Path]:
    """Candidate groundwater observation paths ordered by target."""
    target_norm = (target or "").lower()
    obs_root = project_dir / "observations" / "groundwater"

    if target_norm == "gw_grace":
        return [
            obs_root / "grace" / "processed" / f"{domain_name}_grace_processed.csv",
            obs_root / "grace" / "preprocessed" / f"{domain_name}_grace_processed.csv",
        ]

    return [
        obs_root / "depth" / "processed" / f"{domain_name}_gw_processed.csv",
        obs_root / "depth" / "preprocessed" / f"{domain_name}_gw_processed.csv",
    ]


def tws_observation_candidates(project_dir: Path, domain_name: str, target: str | None) -> List[Path]:
    """Candidate TWS/mass-balance observation paths ordered by legacy search rules."""
    obs_base = project_dir / "observations"
    target_norm = (target or "").lower()

    if target_norm == "stor_mb":
        search_dirs = [
            obs_base / "storage" / "mass_balance",
            obs_base / "mass_balance",
        ]
        patterns = [
            f"{domain_name}_mass_balance.csv",
            "obs_mass_balance.csv",
            "mass_balance.csv",
        ]
    else:
        search_dirs = [
            obs_base / "storage" / "grace",
            obs_base / "grace",
        ]
        patterns = [
            f"{domain_name}_HRUs_GRUs_grace_tws_anomaly.csv",
            f"{domain_name}_HRUs_elevation_grace_tws_anomaly_by_hru.csv",
            f"{domain_name}_grace_tws_processed.csv",
            f"{domain_name}_grace_tws_anomaly.csv",
            "grace_tws_anomaly.csv",
            "tws_anomaly.csv",
        ]

    candidates: List[Path] = []
    for obs_dir in search_dirs:
        preprocessed_dir = obs_dir / "preprocessed"
        for pattern in patterns:
            candidates.append(preprocessed_dir / pattern)
            candidates.append(obs_dir / pattern)

    return candidates


def tws_default_observation_path(project_dir: Path, domain_name: str) -> Path:
    """Default TWS observation fallback path."""
    return project_dir / "observations" / "grace" / "preprocessed" / f"{domain_name}_grace_tws_processed.csv"


def observation_output_candidates_by_family(project_dir: Path, domain_name: str) -> Dict[str, List[Path]]:
    """Canonical + legacy candidate observation output paths by family."""

    def _dedupe(paths: List[Path]) -> List[Path]:
        seen = set()
        unique_paths: List[Path] = []
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            unique_paths.append(path)
        return unique_paths

    return {
        "snow": _dedupe(
            snow_observation_candidates(project_dir, domain_name, "swe")
            + snow_observation_candidates(project_dir, domain_name, "sca")
            + snow_observation_candidates(project_dir, domain_name, "snow_depth")
            + snow_observation_candidates(project_dir, domain_name, None)
        ),
        "soil_moisture": _dedupe(
            soil_moisture_observation_candidates(project_dir, domain_name, "sm_point")
            + soil_moisture_observation_candidates(project_dir, domain_name, "sm_smap")
            + soil_moisture_observation_candidates(project_dir, domain_name, "sm_ismn")
            + soil_moisture_observation_candidates(project_dir, domain_name, "sm_esa")
            + soil_moisture_observation_candidates(project_dir, domain_name, None)
        ),
        "streamflow": streamflow_observation_candidates(project_dir, domain_name),
        "et": _dedupe(
            et_observation_candidates(project_dir, domain_name, "mod16")
            + et_observation_candidates(project_dir, domain_name, "fluxnet")
            + et_observation_candidates(project_dir, domain_name, "openet")
            + et_observation_candidates(project_dir, domain_name, "fluxcom")
            + et_observation_candidates(project_dir, domain_name, "gleam")
        ),
    }
