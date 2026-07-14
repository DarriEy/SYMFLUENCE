# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Configuration-hash stage markers for SYMFLUENCE workflow invalidation.

Each workflow stage writes a JSON marker after successful completion.
The marker includes a SHA-256 hash of the config sections relevant to
that stage.  On subsequent runs the orchestrator compares the stored
hash with the current config — if they differ the stage is re-executed
even when its output files already exist.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:
    from symfluence.symfluence_version import __version__
except ImportError:
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("symfluence")
    except (ImportError, PackageNotFoundError):
        __version__ = "0.0.0"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage → config-section mapping
# ---------------------------------------------------------------------------

STAGE_CONFIG_SECTIONS: Dict[str, List[str]] = {
    "setup_project": ["domain"],
    "create_pour_point": ["domain"],
    "acquire_attributes": ["domain", "data"],
    "define_domain": ["domain"],
    "discretize_domain": ["domain"],
    "process_observed_data": ["evaluation", "data"],
    "acquire_forcings": ["forcing", "domain"],
    "run_model_agnostic_preprocessing": ["forcing", "domain"],
    # Model-agnostic product: the store links the basin-averaged forcing and
    # builds domain-level observation/attribute NetCDFs. It must NOT hash the
    # ``model`` section — doing so makes every model in a shared domain see the
    # other models' store as stale and rebuild shared artefacts underneath them.
    "build_model_ready_store": ["data", "forcing"],
    "preprocess_models": ["model", "forcing", "domain"],
    "run_models": ["model", "domain"],
    "postprocess_results": ["model", "evaluation"],
    "calibrate_model": ["optimization", "model", "evaluation", "domain"],
    "run_benchmarking": ["evaluation"],
    "run_decision_analysis": ["evaluation", "model"],
    "run_sensitivity_analysis": ["evaluation", "optimization"],
}

# Stages whose products live in the shared, domain-level tree (shapefiles,
# data/forcing/basin_averaged_data, data/model_ready, data/observations).
# Every experiment in a domain consumes the *same* files, so these stages must
# hash to the same value across experiments — otherwise concurrent workflows on
# one domain each judge the shared product stale and regenerate it while the
# others are reading it.
DOMAIN_SHARED_STAGES: frozenset = frozenset({
    "setup_project",
    "create_pour_point",
    "acquire_attributes",
    "define_domain",
    "discretize_domain",
    "process_observed_data",
    "acquire_forcings",
    "run_model_agnostic_preprocessing",
    "build_model_ready_store",
})

# Config fields scoped to a single experiment. They select *where a run writes
# its own outputs*, never the content of a domain-shared product, so they are
# excluded from the hash of every stage in DOMAIN_SHARED_STAGES.
EXPERIMENT_SCOPED_FIELDS: Dict[str, List[str]] = {
    "domain": ["experiment_id"],
}

MARKER_DIR_NAME = "stage_markers"

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def compute_config_hash(
    config: Any,
    sections: Sequence[str],
    exclude: Optional[Mapping[str, Sequence[str]]] = None,
) -> str:
    """
    Compute a deterministic SHA-256 hex digest over the given config sections.

    Sections are sorted alphabetically before serialization so the hash is
    independent of the order they are listed in ``STAGE_CONFIG_SECTIONS``.

    Args:
        config: Typed ``SymfluenceConfig``.
        sections: Config sections to hash.
        exclude: Optional ``{section: [field, ...]}`` of fields to drop before
            hashing (see ``EXPERIMENT_SCOPED_FIELDS``).
    """
    combined: Dict[str, Any] = {}
    for section in sorted(sections):
        section_obj = getattr(config, section, None)
        if section_obj is None:
            continue
        dumped = section_obj.model_dump(by_alias=False)
        for field in (exclude or {}).get(section, []):
            dumped.pop(field, None)
        combined[section] = dumped

    payload = json.dumps(combined, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_stage_hash(config: Any, stage_name: str) -> str:
    """
    Compute the invalidation hash for *stage_name*.

    Resolves the stage's config sections and drops experiment-scoped fields for
    domain-shared stages, so that two experiments sharing a domain agree on the
    hash of every shared product.  Returns ``""`` for unknown stages (no
    sections mapped → the orchestrator falls back to output-existence checks).
    """
    sections = STAGE_CONFIG_SECTIONS.get(stage_name, [])
    if not sections:
        return ""
    exclude = EXPERIMENT_SCOPED_FIELDS if stage_name in DOMAIN_SHARED_STAGES else None
    return compute_config_hash(config, sections, exclude=exclude)


# ---------------------------------------------------------------------------
# Marker dataclass
# ---------------------------------------------------------------------------


@dataclass
class StageMarker:
    """Metadata written after a stage completes successfully."""

    stage: str
    completed_utc: str
    config_hash: str
    symfluence_version: str
    git_commit: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _marker_dir(project_dir: Path) -> Path:
    return project_dir / ".symfluence" / MARKER_DIR_NAME


def _marker_path(project_dir: Path, stage_name: str) -> Path:
    return _marker_dir(project_dir) / f"{stage_name}.json"


def _current_git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_marker(
    project_dir: Path,
    stage_name: str,
    config_hash: str,
    git_commit: Optional[str] = None,
) -> Path:
    """Write a JSON marker for *stage_name* and return its path."""
    marker = StageMarker(
        stage=stage_name,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        config_hash=config_hash,
        symfluence_version=__version__,
        git_commit=git_commit or _current_git_commit(),
    )
    path = _marker_path(project_dir, stage_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(marker), indent=2), encoding="utf-8")
    return path


def read_marker(project_dir: Path, stage_name: str) -> Optional[StageMarker]:
    """Read a stage marker, returning ``None`` if missing or corrupt."""
    path = _marker_path(project_dir, stage_name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return StageMarker(
            stage=data["stage"],
            completed_utc=data["completed_utc"],
            config_hash=data["config_hash"],
            symfluence_version=data["symfluence_version"],
            git_commit=data.get("git_commit"),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("Corrupt stage marker for '%s' — will re-execute", stage_name)
        return None


def is_stage_current(
    project_dir: Path,
    stage_name: str,
    current_hash: str,
) -> bool:
    """Return ``True`` only when a marker exists and its hash matches."""
    marker = read_marker(project_dir, stage_name)
    if marker is None:
        return False
    return marker.config_hash == current_hash


def clear_markers(
    project_dir: Path,
    stage_names: Optional[Sequence[str]] = None,
) -> None:
    """Remove all markers, or only those for the given stage names."""
    d = _marker_dir(project_dir)
    if not d.exists():
        return
    if stage_names is None:
        for f in d.glob("*.json"):
            f.unlink()
    else:
        for name in stage_names:
            p = _marker_path(project_dir, name)
            if p.exists():
                p.unlink()
