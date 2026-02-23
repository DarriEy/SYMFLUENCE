"""Shared workflow step metadata for CLI, GUI, TUI, and tests.

This module centralizes canonical workflow step names, descriptions, and aliases
so all frontends present the same workflow contract.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Canonical workflow steps (must stay in orchestrator execution order).
WORKFLOW_STEP_ITEMS: List[Tuple[str, str]] = [
    ("setup_project", "Initialize project directory structure and shapefiles"),
    ("create_pour_point", "Create pour point shapefile from coordinates"),
    ("acquire_attributes", "Download and process geospatial attributes (soil, land class, etc.)"),
    ("define_domain", "Define hydrological domain boundaries and river basins"),
    ("discretize_domain", "Discretize domain into HRUs or other modeling units"),
    ("process_observed_data", "Process observational data (streamflow, etc.)"),
    ("acquire_forcings", "Acquire meteorological forcing data"),
    ("model_agnostic_preprocessing", "Run model-agnostic preprocessing of forcing and attribute data"),
    ("build_model_ready_store", "Build model-ready forcing/attributes/observations data store"),
    ("model_specific_preprocessing", "Setup model-specific input files and configuration"),
    ("run_model", "Execute the hydrological model simulation"),
    ("postprocess_results", "Postprocess and finalize model results"),
    ("calibrate_model", "Run model calibration and parameter optimization"),
    ("run_benchmarking", "Run benchmarking analysis against observations"),
    ("run_decision_analysis", "Run decision analysis for model comparison"),
    ("run_sensitivity_analysis", "Run sensitivity analysis on model parameters"),
]

WORKFLOW_STEP_NAMES: List[str] = [name for name, _ in WORKFLOW_STEP_ITEMS]
WORKFLOW_STEP_DESCRIPTION_MAP: Dict[str, str] = dict(WORKFLOW_STEP_ITEMS)

# Short aliases for workflow steps (alias -> canonical name)
WORKFLOW_STEP_ALIASES: Dict[str, str] = {
    "setup": "setup_project",
    "pour_point": "create_pour_point",
    "pp": "create_pour_point",
    "attributes": "acquire_attributes",
    "attrs": "acquire_attributes",
    "domain": "define_domain",
    "discretize": "discretize_domain",
    "obs_data": "process_observed_data",
    "obs": "process_observed_data",
    "forcings": "acquire_forcings",
    "agnostic_prep": "model_agnostic_preprocessing",
    "map": "model_agnostic_preprocessing",
    "build_store": "build_model_ready_store",
    "store": "build_model_ready_store",
    "model_store": "build_model_ready_store",
    "specific_prep": "model_specific_preprocessing",
    "msp": "model_specific_preprocessing",
    "model": "run_model",
    "calibrate": "calibrate_model",
    "cal": "calibrate_model",
    "benchmark": "run_benchmarking",
    "bench": "run_benchmarking",
    "decision": "run_decision_analysis",
    "sensitivity": "run_sensitivity_analysis",
    "sa": "run_sensitivity_analysis",
    "postprocess": "postprocess_results",
    "post": "postprocess_results",
}

WORKFLOW_STEP_ALIAS_REVERSE: Dict[str, List[str]] = {}
for _alias, _canonical in WORKFLOW_STEP_ALIASES.items():
    WORKFLOW_STEP_ALIAS_REVERSE.setdefault(_canonical, []).append(_alias)


def resolve_workflow_step_name(name: str) -> str:
    """Resolve canonical workflow step name from a name or alias.

    Raises:
        ValueError: If step name/alias is unknown.
    """
    if name in WORKFLOW_STEP_NAMES:
        return name
    if name in WORKFLOW_STEP_ALIASES:
        return WORKFLOW_STEP_ALIASES[name]

    all_accepted = sorted(set(WORKFLOW_STEP_NAMES) | set(WORKFLOW_STEP_ALIASES.keys()))
    raise ValueError(
        "unknown step "
        f"'{name}'. Valid steps and aliases:\n  "
        + "\n  ".join(all_accepted)
    )
