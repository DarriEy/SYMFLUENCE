# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Base service class for SYMFLUENCE CLI services.

Provides shared functionality for all CLI services including:
- Console injection for output
- Configuration loading
- Data directory resolution
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..console import Console, get_console


class BaseService:
    """
    Base class for CLI services with shared functionality.

    All CLI services inherit from this class to get:
    - Console injection for formatted output
    - Configuration loading from SYMFLUENCE instance or template
    - Data directory resolution

    Args:
        console: Optional Console instance for output. Uses global console if not provided.
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the base service.

        Args:
            console: Console instance for output. If None, uses global console.
        """
        self._console = console or get_console()

    def _load_config(self, symfluence_instance=None) -> Dict[str, Any]:
        """
        Load configuration from SYMFLUENCE instance or fall back to template.

        Args:
            symfluence_instance: Optional SYMFLUENCE instance with config attribute.

        Returns:
            Configuration dictionary.
        """
        if symfluence_instance and hasattr(symfluence_instance, "config"):
            return symfluence_instance.config
        if symfluence_instance and hasattr(symfluence_instance, "workflow_orchestrator"):
            return symfluence_instance.workflow_orchestrator.config

        try:
            from symfluence.resources import get_config_template

            config_path = get_config_template()
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return self._ensure_valid_config_paths(config, config_path)
        except (ImportError, FileNotFoundError, yaml.YAMLError) as e:
            self._console.debug(f"Could not load config: {e}")
            return {}

    def _get_data_dir(self, config: Dict[str, Any]) -> Path:
        """
        Get SYMFLUENCE data directory from environment or config.

        Thin wrapper over :meth:`_resolve_data_dir` that discards the
        human-readable reason. See that method for the full resolution order.

        Args:
            config: Configuration dictionary.

        Returns:
            Path to data directory.
        """
        return self._resolve_data_dir(config)[0]

    def _resolve_data_dir(self, config: Dict[str, Any]) -> tuple[Path, str]:
        """
        Resolve the SYMFLUENCE data directory and explain how it was chosen.

        Priority:
        1. SYMFLUENCE_DATA_DIR / SYMFLUENCE_DATA environment variable (explicit).
        2. Config value (if not the 'default' sentinel).
        3. An existing ``SYMFLUENCE_data`` workspace at or above the current
           working directory (so ad-hoc installs land in a workspace the user
           is already sitting in).
        4. Sibling ``SYMFLUENCE_data`` next to a *real* repo checkout, when one
           is detected and the location is writable (developer default).
        5. ``<cwd>/SYMFLUENCE_data`` for ad-hoc installs (no repo, no env/config),
           or when the repo-sibling default is not writable.

        Every candidate after the explicit env/config values is validated for
        writability; non-writable candidates are skipped with a fallback.

        Args:
            config: Configuration dictionary.

        Returns:
            A ``(path, reason)`` tuple. ``reason`` is a short phrase suitable for
            surfacing to the user (e.g. in the installer's location banner).
        """
        # 1. Explicit environment variable — honour verbatim (even if awkward).
        data_dir = os.getenv("SYMFLUENCE_DATA_DIR") or os.getenv("SYMFLUENCE_DATA")
        if data_dir:
            return Path(data_dir), "SYMFLUENCE_DATA_DIR environment variable"

        # 2. Explicit config value (skip 'default' sentinel).
        config_val = config.get("SYMFLUENCE_DATA_DIR")
        if config_val and config_val != "default":
            return Path(config_val), "config SYMFLUENCE_DATA_DIR"

        # 3-5. No explicit value — infer (shared with config-template repair).
        return self._infer_data_dir_fallback()

    def _infer_data_dir_fallback(self) -> tuple[Path, str]:
        """
        Infer a data directory when no explicit env/config value applies.

        Shared by :meth:`_resolve_data_dir` (``binary install`` et al.) and
        :meth:`_ensure_valid_config_paths` (config-template repair) so the
        cwd-aware inference is identical in both. Order:

        1. An existing, writable ``SYMFLUENCE_data`` workspace near cwd.
        2. Sibling ``SYMFLUENCE_data`` next to a *real* repo checkout, when the
           location is writable (developer default).
        3. ``<cwd>/SYMFLUENCE_data`` for ad-hoc installs.
        4. The computed default, even if not writable, as a last resort so
           callers have a concrete path to report against.

        Returns:
            A ``(path, reason)`` tuple.
        """
        # 1. Reuse an existing workspace found at or above the current directory.
        existing = self._find_existing_data_dir()
        if existing is not None and self._is_writable_dir(existing):
            return existing, "existing workspace near the current directory"

        # 2. Repo-sibling default — only when running from a real checkout and
        #    the location is actually writable.
        from symfluence.core.config.factories import _resolve_default_data_dir
        if self._running_from_repo():
            sibling = Path(_resolve_default_data_dir())
            if self._is_writable_dir(sibling):
                return sibling, "sibling of the SYMFLUENCE repository"

        # 3. Ad-hoc install (no env, no config, no repo) — infer from cwd.
        cwd = Path.cwd()
        if cwd.name == ".ipynb_checkpoints":
            cwd = cwd.parent
        cwd_data = cwd / "SYMFLUENCE_data"
        if self._is_writable_dir(cwd_data):
            return cwd_data, "current working directory (ad-hoc install)"

        # 4. Last resort.
        return Path(_resolve_default_data_dir()), "default data directory"

    @staticmethod
    def _looks_like_workspace(path: Path) -> bool:
        """
        Whether ``path`` looks like a populated SYMFLUENCE_data workspace.

        Requiring a marker (``installs``/``.symfluence``/a ``domain_*`` dir)
        before reusing a directory avoids adopting an unrelated directory that
        merely happens to be named ``SYMFLUENCE_data``.
        """
        if not path.is_dir():
            return False
        if (path / "installs").is_dir() or (path / ".symfluence").is_dir():
            return True
        try:
            return any(path.glob("domain_*"))
        except OSError:
            return False

    @classmethod
    def _find_existing_data_dir(cls, max_levels: int = 3) -> Optional[Path]:
        """
        Look for an existing SYMFLUENCE_data *workspace* at or above cwd.

        Walks up at most ``max_levels`` directories and returns a ``SYMFLUENCE_data``
        directory (the directory itself, or a child) only when it looks like a
        real workspace (:meth:`_looks_like_workspace`). The bounded depth +
        marker requirement keep an ad-hoc install from adopting a coincidentally
        named or far-away directory.

        Returns:
            The existing workspace directory, or ``None`` if none is found nearby.
        """
        start = Path.cwd()
        if start.name == ".ipynb_checkpoints":
            start = start.parent
        for depth, directory in enumerate([start, *start.parents]):
            if depth > max_levels:
                break
            if directory.name == "SYMFLUENCE_data" and cls._looks_like_workspace(directory):
                return directory
            candidate = directory / "SYMFLUENCE_data"
            if cls._looks_like_workspace(candidate):
                return candidate
        return None

    @staticmethod
    def _running_from_repo() -> bool:
        """
        Detect whether SYMFLUENCE is running from a real source checkout.

        Returns True when the installed package sits inside a directory tree that
        contains a ``pyproject.toml`` or ``.git`` (a developer checkout), and
        False for an installed/site-packages ("ad-hoc") install.
        """
        try:
            import symfluence

            candidate = Path(symfluence.__file__).parent
            for _ in range(5):
                candidate = candidate.parent
                if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
                    return True
        except (ImportError, AttributeError, OSError, TypeError):
            pass
        return False

    @staticmethod
    def _is_writable_dir(path: Path) -> bool:
        """
        Check whether ``path`` is (or could be) created as a writable directory.

        Walks up to the nearest existing ancestor and tests write access there,
        so a not-yet-created target is judged by whether it *could* be made.
        """
        probe = path
        while not probe.exists():
            parent = probe.parent
            if parent == probe:  # reached filesystem root without an existing dir
                return False
            probe = parent
        # If the only existing ancestor is the filesystem root/anchor itself
        # while the requested target is deeper, treat it as not writable:
        # creating a brand-new top-level directory at the root isn't a valid
        # data-dir location (and on Windows a bare "/foo" is drive-relative, so
        # the current drive root spuriously satisfies os.access(..., W_OK)).
        if probe != path and probe == Path(probe.anchor):
            return False
        return os.access(probe, os.W_OK)

    def _ensure_valid_config_paths(
        self, config: Dict[str, Any], config_path: Path
    ) -> Dict[str, Any]:
        """
        Ensure SYMFLUENCE_DATA_DIR and SYMFLUENCE_CODE_DIR paths exist and are valid.

        Args:
            config: Configuration dictionary to validate.
            config_path: Path to the configuration file.

        Returns:
            Updated configuration dictionary with valid paths.
        """
        data_dir = config.get("SYMFLUENCE_DATA_DIR")
        code_dir = config.get("SYMFLUENCE_CODE_DIR")

        data_dir_valid = False
        code_dir_valid = False

        # Treat 'default' sentinel as unset — let the resolver handle it
        if data_dir == "default":
            data_dir = None
        if code_dir == "default":
            code_dir = None

        if data_dir:
            try:
                data_path = Path(data_dir)
                if data_path.exists():
                    test_file = data_path / ".symfluence_test"
                    try:
                        test_file.touch()
                        test_file.unlink()
                        data_dir_valid = True
                    except (PermissionError, OSError):
                        pass
                else:
                    try:
                        data_path.mkdir(parents=True, exist_ok=True)
                        data_dir_valid = True
                    except (PermissionError, OSError):
                        pass
            except (ValueError, OSError):
                pass

        if code_dir:
            try:
                code_path = Path(code_dir)
                if code_path.exists() and os.access(code_path, os.R_OK):
                    code_dir_valid = True
            except (ValueError, OSError):
                pass

        if not data_dir_valid or not code_dir_valid:
            # Only warn when an explicitly configured path failed validation.
            # Unset paths ('default' sentinel) are resolved quietly below.
            bad_paths = []
            if data_dir and not data_dir_valid:
                bad_paths.append(f"SYMFLUENCE_DATA_DIR: {data_dir}")
            if code_dir and not code_dir_valid:
                bad_paths.append(f"SYMFLUENCE_CODE_DIR: {code_dir}")
            if bad_paths:
                self._console.warning(
                    "Detected invalid or inaccessible paths in config template: "
                    + "; ".join(bad_paths)
                )

            from symfluence.core.config.factories import _resolve_default_code_dir

            if not code_dir_valid:
                new_code_dir = _resolve_default_code_dir()
                config["SYMFLUENCE_CODE_DIR"] = new_code_dir
                self._console.success(f"SYMFLUENCE_CODE_DIR set to: {new_code_dir}")

            if not data_dir_valid:
                # Same cwd-aware inference as `binary install`, so the two paths
                # cannot disagree on where the data dir lives.
                new_data_dir, _reason = self._infer_data_dir_fallback()
                config["SYMFLUENCE_DATA_DIR"] = str(new_data_dir)
                try:
                    new_data_dir.mkdir(parents=True, exist_ok=True)
                    self._console.success(f"SYMFLUENCE_DATA_DIR set to: {new_data_dir}")
                except OSError:
                    pass

        return config
