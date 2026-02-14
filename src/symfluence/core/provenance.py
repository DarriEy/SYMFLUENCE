"""
Provenance tracking for SYMFLUENCE workflow runs.

Captures framework version, git state, dependency versions, platform details,
and per-step timing into a self-documenting run manifest (JSON).
"""

import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from symfluence.symfluence_version import __version__
except ImportError:
    __version__ = "0+unknown"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _git_info(repo_dir: Optional[Path] = None) -> Dict[str, Optional[str]]:
    """Return git commit, branch, and dirty status for *repo_dir*."""
    cwd = str(repo_dir) if repo_dir else None
    info: Dict[str, Optional[str]] = {
        "commit": None,
        "commit_short": None,
        "branch": None,
        "dirty": None,
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
        info["commit"] = commit
        info["commit_short"] = commit[:8]
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
        info["dirty"] = bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return info


def _dependency_versions() -> Dict[str, str]:
    """Return installed versions of key scientific dependencies."""
    packages = [
        "numpy", "xarray", "pandas", "scipy", "netCDF4", "geopandas",
        "shapely", "rasterio", "pyproj", "dask", "jax", "torch",
    ]
    versions: Dict[str, str] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return versions


def _platform_info() -> Dict[str, str]:
    """Return OS, architecture, hostname, and Python version."""
    return {
        "os": platform.system(),
        "os_version": platform.release(),
        "arch": platform.machine(),
        "hostname": platform.node(),
        "python": platform.python_version(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_run_id(experiment_id: str, git_short: Optional[str] = None) -> str:
    """Build a human-readable, unique run identifier."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = git_short if git_short else "nogit"
    return f"{experiment_id}_{ts}_{suffix}"


@dataclass
class RunProvenance:
    """Immutable record of everything needed to reproduce a run."""

    run_id: str
    experiment_id: str
    domain_name: str
    config_path: Optional[str]
    symfluence_version: str
    git: Dict[str, Optional[str]]
    platform: Dict[str, str]
    dependencies: Dict[str, str]
    start_utc: str
    end_utc: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "running"
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "domain_name": self.domain_name,
            "config_path": self.config_path,
            "symfluence_version": self.symfluence_version,
            "git": self.git,
            "platform": self.platform,
            "dependencies": self.dependencies,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "elapsed_seconds": self.elapsed_seconds,
            "steps": list(self.steps),
            "status": self.status,
            "errors": list(self.errors),
        }

    def write(self, output_dir: Path) -> Path:
        """Write ``run_manifest.json`` into *output_dir* and return its path."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "run_manifest.json"
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path


def capture_provenance(
    experiment_id: str,
    domain_name: str,
    config_path: Optional[str] = None,
    repo_dir: Optional[Path] = None,
) -> RunProvenance:
    """Create a fresh ``RunProvenance`` capturing the current environment."""
    git = _git_info(repo_dir)
    return RunProvenance(
        run_id=make_run_id(experiment_id, git.get("commit_short")),
        experiment_id=experiment_id,
        domain_name=domain_name,
        config_path=str(config_path) if config_path else None,
        symfluence_version=__version__,
        git=git,
        platform=_platform_info(),
        dependencies=_dependency_versions(),
        start_utc=datetime.now(timezone.utc).isoformat(),
    )


def record_step(
    prov: Optional[RunProvenance],
    name: str,
    duration_s: float,
    status: str = "completed",
    error: Optional[str] = None,
) -> None:
    """Append a step record to *prov*.  No-op when *prov* is ``None``."""
    if prov is None:
        return
    entry: Dict[str, Any] = {
        "name": name,
        "duration_s": round(duration_s, 3),
        "status": status,
    }
    if error is not None:
        entry["error"] = error
    prov.steps.append(entry)


def finalize(
    prov: Optional[RunProvenance],
    status: str,
    errors: Optional[List[str]] = None,
) -> Optional[RunProvenance]:
    """Stamp *prov* with end time, elapsed duration, and final status."""
    if prov is None:
        return None
    now = datetime.now(timezone.utc)
    prov.end_utc = now.isoformat()
    start = datetime.fromisoformat(prov.start_utc)
    prov.elapsed_seconds = round((now - start).total_seconds(), 3)
    prov.status = status
    if errors:
        prov.errors.extend(errors)
    return prov
