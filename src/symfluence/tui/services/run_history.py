"""
Service for parsing _workLog run summary JSON files.

Provides run history browsing across one or more domains.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunSummary:
    """Parsed content of a single run_summary_*.json file."""

    file_path: Path
    timestamp: Optional[datetime] = None
    domain: str = ""
    experiment_id: str = ""
    status: str = "unknown"
    execution_time: float = 0.0
    steps_completed: List[str] = field(default_factory=list)
    total_steps: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    total_errors: int = 0
    warnings: List[str] = field(default_factory=list)
    total_warnings: int = 0
    model: str = ""
    algorithm: str = ""
    config_snapshot: Optional[Dict[str, Any]] = None


class RunHistoryService:
    """Parse and browse run history for a domain."""

    def __init__(self, domain_path: Path):
        self._domain_path = Path(domain_path)
        self._domain_name = self._domain_path.name.replace("domain_", "", 1)

    @property
    def log_dir(self) -> Path:
        return self._domain_path / f"_workLog_{self._domain_name}"

    def list_runs(self) -> List[RunSummary]:
        """Return all run summaries, newest first."""
        if not self.log_dir.is_dir():
            return []

        runs = []
        for f in sorted(self.log_dir.glob("run_summary_*.json"), reverse=True):
            run = self._parse_summary(f)
            if run:
                runs.append(run)
        return runs

    def load_config_snapshot(self, run: RunSummary) -> Optional[Dict[str, Any]]:
        """Load the config YAML logged alongside a run.

        Looks for config_*.yaml files with a timestamp close to the run.
        """
        if not self.log_dir.is_dir():
            return None

        # Find config file closest to the run timestamp
        candidates = sorted(self.log_dir.glob("config_*.yaml"), reverse=True)
        if not candidates:
            return None

        # Try to match by timestamp prefix
        if run.timestamp:
            ts_str = run.timestamp.strftime("%Y%m%d_%H%M%S")
            for c in candidates:
                if ts_str in c.name:
                    return self._load_yaml(c)

        # Fall back to most recent
        return self._load_yaml(candidates[0])

    @staticmethod
    def _parse_summary(path: Path) -> Optional[RunSummary]:
        """Parse a run_summary JSON into a RunSummary dataclass."""
        try:
            with open(path) as f:
                data = json.load(f)

            ts = None
            ts_str = data.get("timestamp")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str)
                except (ValueError, TypeError):
                    pass

            steps = data.get("steps_completed", [])
            # Steps may be dicts with 'cli' key or plain strings
            step_names = []
            for s in steps:
                if isinstance(s, dict):
                    step_names.append(s.get("cli", s.get("name", str(s))))
                else:
                    step_names.append(str(s))

            config = data.get("configuration", {})

            return RunSummary(
                file_path=path,
                timestamp=ts,
                domain=data.get("domain", ""),
                experiment_id=data.get("experiment_id", ""),
                status=data.get("status", "unknown"),
                execution_time=data.get("execution_time_seconds", 0.0),
                steps_completed=step_names,
                total_steps=data.get("total_steps_completed", len(step_names)),
                errors=data.get("errors", []),
                total_errors=data.get("total_errors", 0),
                warnings=data.get("warnings", []),
                total_warnings=data.get("total_warnings", 0),
                model=config.get("hydrological_model", ""),
                algorithm=config.get("optimization_algorithm", ""),
            )
        except Exception:
            return None

    @staticmethod
    def _load_yaml(path: Path) -> Optional[Dict[str, Any]]:
        """Load a YAML config file."""
        try:
            import yaml
            with open(path) as f:
                return yaml.safe_load(f)
        except Exception:
            return None
