"""
Notebook service for launching Jupyter examples.

Handles Jupyter notebook discovery and launching with proper kernel setup.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .base import BaseService
from ..console import Console


class NotebookService(BaseService):
    """
    Service for managing Jupyter notebook operations.

    Handles:
    - Example notebook discovery
    - Virtual environment detection
    - IPython kernel setup
    - Jupyter Lab/Notebook launching
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the NotebookService.

        Args:
            console: Console instance for output. If None, uses global console.
        """
        super().__init__(console=console)

    def launch_example_notebook(
        self,
        example_id: str,
        repo_root: Optional[Path] = None,
        venv_candidates: Optional[List[str]] = None,
        prefer_lab: bool = True,
    ) -> int:
        """
        Launch an example notebook bound to the repo's root venv.

        Args:
            example_id: Example identifier (e.g., '1a', '02b', 'quickstart')
            repo_root: Root directory of the repository
            venv_candidates: List of virtual environment directory names to search
            prefer_lab: If True, prefer JupyterLab over classic Notebook

        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        repo_root = Path(repo_root) if repo_root else Path.cwd().resolve()
        if not repo_root.exists():
            self._console.error(f"Repo root not found: {repo_root}")
            return 1

        raw = example_id.strip()
        m = re.fullmatch(r"(?i)(\d{1,2})([a-z])", raw)
        if m:
            n = int(m.group(1))
            letter = m.group(2).lower()
            prefix = f"{n:02d}{letter}"
        else:
            prefix = raw.lower()

        if venv_candidates is None:
            venv_candidates = ["venv", ".venv", "env", ".conda", ".virtualenv"]

        def _venv_python(venv: Path) -> Path:
            return venv / ("Scripts" if os.name == "nt" else "bin") / (
                "python.exe" if os.name == "nt" else "python"
            )

        venv_dir = None
        for name in venv_candidates:
            candidate = repo_root / name
            if candidate.exists() and candidate.is_dir():
                venv_dir = candidate
                break

        python_exe = None
        if venv_dir:
            python_exe = _venv_python(venv_dir)
            if not python_exe.exists():
                alt = venv_dir / (
                    "Scripts/python" if os.name == "nt" else "bin/python3"
                )
                if alt.exists():
                    python_exe = alt

        if not python_exe or not python_exe.exists():
            python_exe = Path(sys.executable)
            self._console.warning(
                "Could not find a root venv Python. Falling back to current interpreter."
            )

        examples_root = repo_root / "examples"
        if not examples_root.exists():
            self._console.error(f"'examples/' directory not found at: {examples_root}")
            return 2

        primary_matches = sorted(examples_root.rglob(f"{prefix}_*.ipynb"))
        fallback_matches = (
            [] if primary_matches else sorted(examples_root.rglob(f"{prefix}*.ipynb"))
        )
        matches = primary_matches or fallback_matches
        if not matches:
            self._console.error(
                f"Example notebook not found for ID '{example_id}'."
            )
            return 2

        nb_path = matches[0]
        if len(matches) > 1:
            self._console.info(
                "Multiple notebooks match this prefix; opening the first match:"
            )
            for i, p in enumerate(matches[:10], 1):
                try:
                    self._console.indent(f"[{i}] {p.relative_to(repo_root)}")
                except ValueError:
                    self._console.indent(f"[{i}] {p}")

        try:
            chk = subprocess.run(
                [str(python_exe), "-m", "pip", "show", "ipykernel"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if chk.returncode != 0:
                self._console.info("Installing ipykernel into the repo venv...")
                subprocess.run([str(python_exe), "-m", "pip", "install", "ipykernel"])

            self._console.info(
                f"Ensuring 'symfluence-root' ipykernel is registered for {python_exe}..."
            )
            subprocess.run(
                [
                    str(python_exe),
                    "-m",
                    "ipykernel",
                    "install",
                    "--user",
                    "--name",
                    "symfluence-root",
                    "--display-name",
                    "Python (symfluence-root)",
                ],
                check=True,
            )
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            self._console.warning(f"Could not ensure ipykernel: {e}")

        # Try to launch Jupyter
        tool = "jupyterlab" if prefer_lab else "notebook"
        try:
            subprocess.run([str(python_exe), "-m", tool, str(nb_path)])
            return 0
        except (OSError, subprocess.SubprocessError):
            try:
                subprocess.run(
                    [str(python_exe), "-m", "jupyter", "notebook", str(nb_path)]
                )
                return 0
            except (OSError, subprocess.SubprocessError) as e:
                self._console.error(f"Could not launch Jupyter: {e}")
                return 3

    def list_examples(self, repo_root: Optional[Path] = None) -> List[Path]:
        """
        List all available example notebooks.

        Args:
            repo_root: Root directory of the repository

        Returns:
            List of paths to example notebooks
        """
        repo_root = Path(repo_root) if repo_root else Path.cwd().resolve()
        examples_root = repo_root / "examples"

        if not examples_root.exists():
            self._console.error(f"'examples/' directory not found at: {examples_root}")
            return []

        notebooks = sorted(examples_root.rglob("*.ipynb"))

        self._console.info(f"Found {len(notebooks)} example notebooks:")
        for nb in notebooks:
            try:
                rel_path = nb.relative_to(repo_root)
                self._console.indent(str(rel_path))
            except ValueError:
                self._console.indent(str(nb))

        return notebooks
