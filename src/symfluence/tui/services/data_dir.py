"""
Service for scanning SYMFLUENCE_DATA_DIR for domain directories.

Provides domain discovery and metadata without requiring a loaded config.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class DomainInfo:
    """Summary of a domain directory."""

    name: str
    path: Path
    run_count: int = 0
    last_run: Optional[datetime] = None
    last_status: str = "unknown"
    experiments: List[str] = field(default_factory=list)


class DataDirService:
    """Scan SYMFLUENCE_DATA_DIR for domain_* directories and run metadata."""

    def __init__(self, data_dir: Optional[str] = None):
        self._data_dir = self._resolve_data_dir(data_dir)

    @property
    def data_dir(self) -> Optional[Path]:
        return self._data_dir

    @staticmethod
    def _resolve_data_dir(explicit: Optional[str] = None) -> Optional[Path]:
        """Resolve the data directory from explicit path, env var, or default."""
        if explicit:
            p = Path(explicit)
            return p if p.is_dir() else None

        for var in ("SYMFLUENCE_DATA_DIR", "SYMFLUENCE_DATA"):
            val = os.environ.get(var)
            if val:
                p = Path(val)
                if p.is_dir():
                    return p

        return None

    def list_domains(self) -> List[DomainInfo]:
        """Return a DomainInfo for every domain_* directory found."""
        if not self._data_dir or not self._data_dir.is_dir():
            return []

        domains = []
        for entry in sorted(self._data_dir.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("domain_"):
                continue

            domain_name = entry.name[len("domain_"):]
            info = DomainInfo(name=domain_name, path=entry)

            # Scan _workLog for run summaries
            log_dir = entry / f"_workLog_{domain_name}"
            if log_dir.is_dir():
                summaries = sorted(log_dir.glob("run_summary_*.json"))
                info.run_count = len(summaries)
                if summaries:
                    info.last_run = self._parse_timestamp(summaries[-1].stem)
                    info.last_status = self._read_status(summaries[-1])

            # Scan optimization dir for experiment IDs
            opt_dir = entry / "optimization"
            if opt_dir.is_dir():
                for f in opt_dir.glob("*_parallel_iteration_results.csv"):
                    exp_id = f.stem.replace("_parallel_iteration_results", "")
                    if exp_id and exp_id not in info.experiments:
                        info.experiments.append(exp_id)
                info.experiments.sort()

            domains.append(info)

        return domains

    @staticmethod
    def _parse_timestamp(stem: str) -> Optional[datetime]:
        """Parse timestamp from run_summary_YYYYMMDD_HHMMSS filename."""
        parts = stem.replace("run_summary_", "")
        try:
            return datetime.strptime(parts, "%Y%m%d_%H%M%S")
        except ValueError:
            return None

    @staticmethod
    def _read_status(summary_path: Path) -> str:
        """Read status field from a run summary JSON."""
        try:
            with open(summary_path, encoding='utf-8') as f:
                data = json.load(f)
            return data.get("status", "unknown")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return "unknown"
