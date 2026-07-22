# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Embedded-R runtime setup for the GR models.

Two problems are solved here, both of which only bite when R runs *embedded*
inside the SYMFLUENCE process (via rpy2) rather than as a standalone
``Rscript`` process.

**1. Windows DLL search path.**

R package shared objects such as ``stats.dll`` link against ``Rblas.dll`` and
``Rlapack.dll``, which live in ``R_HOME/bin/x64`` and nowhere else. In a
standalone ``Rscript`` process those two DLLs are already loaded (the
interpreter binary sits in the same directory), so the imports resolve by
module name. Embedded, only ``R.dll`` gets loaded up front — ``R.dll`` imports
``Rblas.dll`` but *not* ``Rlapack.dll`` — and R's own ``dyn.load()`` then calls
``LoadLibrary`` through the legacy search order, which searches ``PATH`` but
*not* directories registered with ``os.add_dll_directory()``.

rpy2 only registers ``R_HOME/bin/x64`` via ``os.add_dll_directory()``. That is
enough to load ``R.dll`` itself but not enough for ``dyn.load()``, so unless
``R_HOME/bin/x64`` is also on ``PATH`` every compiled R package fails to load
with::

    unable to load shared object '.../library/stats/libs/x64/stats.dll':
      LoadLibrary failure:  The specified module could not be found.

which R reports at startup as the misleading::

    package 'stats' in options("defaultPackages") was not found

:func:`configure_r_dll_search` puts R's binary directory on ``PATH`` before
rpy2 is imported, which fixes every compiled R package at once.

**2. Fail fast on a missing/unloadable airGR.**

The GR runner used to react to a failed ``library(airGR)`` by calling
``install.packages()`` against CRAN mid-run. On a machine with no outbound
network that leaves the embedded interpreter wedged — observed as a SIGSEGV
after the workflow had already spent 53 minutes preprocessing. Callers should
instead invoke :func:`ensure_airgr_available` before doing any real work so a
broken R shows up as an actionable error in seconds.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from symfluence.core.exceptions import ModelExecutionError

logger = logging.getLogger(__name__)

# Handles returned by os.add_dll_directory() must stay alive: the directory is
# removed from the search path as soon as the handle is garbage-collected.
_dll_directory_handles: list = []

_r_bin_dir: Optional[Path] = None
_configured = False


def _registry_r_home() -> Optional[Path]:
    """Read R's install path from the Windows registry, if recorded there."""
    try:
        import winreg
    except ImportError:  # pragma: no cover - non-Windows
        return None

    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(root, r"SOFTWARE\R-core\R") as key:
                install_path, _ = winreg.QueryValueEx(key, "InstallPath")
        except OSError:
            continue
        if install_path:
            return Path(install_path)
    return None


def _candidate_r_homes() -> list[Path]:
    """Return plausible R_HOME directories, most authoritative first."""
    candidates: list[Path] = []

    env_home = os.environ.get("R_HOME")
    if env_home:
        candidates.append(Path(env_home))

    # An R on PATH points at its own installation: .../R-x.y.z/bin[/x64]/R.exe
    from shutil import which

    for exe_name in ("Rscript", "R"):
        exe = which(exe_name)
        if not exe:
            continue
        bin_dir = Path(exe).parent
        # bin/x64 -> R_HOME is two levels up; bin -> one level up.
        candidates.append(bin_dir.parent.parent if bin_dir.name in ("x64", "i386", "arm64") else bin_dir.parent)

    registry_home = _registry_r_home()
    if registry_home:
        candidates.append(registry_home)

    for program_files in (os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432")):
        if not program_files:
            continue
        r_root = Path(program_files) / "R"
        if r_root.is_dir():
            # Newest version last in sorted order; prefer it.
            candidates.extend(sorted((d for d in r_root.iterdir() if d.is_dir()), reverse=True))

    return candidates


def _r_bin_for(r_home: Path) -> Optional[Path]:
    """Return the architecture-specific binary directory inside ``r_home``."""
    if sys.platform == "win32":
        arch_dir = "arm64" if os.environ.get("PROCESSOR_ARCHITECTURE", "").upper() == "ARM64" else "x64"
        subdirs = (Path("bin") / arch_dir, Path("bin"))
        marker = "R.dll"
    else:
        subdirs = (Path("lib"), Path("bin"))
        marker = None

    for subdir in subdirs:
        candidate = r_home / subdir
        if candidate.is_dir() and (marker is None or (candidate / marker).is_file()):
            return candidate
    return None


def configure_r_dll_search(force: bool = False) -> Optional[Path]:
    """Make R's own shared libraries discoverable by the embedded interpreter.

    Must be called *before* rpy2 is imported. Idempotent and safe to call on
    every platform: outside Windows it is a no-op, since ELF/Mach-O package
    objects record their dependency paths at link time.

    Args:
        force: Re-run the detection even if a previous call already succeeded.

    Returns:
        The R binary directory that was added to ``PATH``, or ``None`` if the
        platform does not need the fix or no R installation could be located.
    """
    global _configured, _r_bin_dir

    if _configured and not force:
        return _r_bin_dir
    if sys.platform != "win32":
        _configured = True
        return None

    for r_home in _candidate_r_homes():
        bin_dir = _r_bin_for(r_home)
        if bin_dir is None:
            continue

        bin_str = str(bin_dir)
        path_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
        if not any(os.path.normcase(entry.rstrip("\\/")) == os.path.normcase(bin_str) for entry in path_entries):
            os.environ["PATH"] = os.pathsep.join([bin_str, *path_entries])
            logger.debug("Added %s to PATH for the embedded R interpreter", bin_str)

        # Belt and braces: this is what lets rpy2's cffi loader find R.dll.
        _dll_directory_handles.append(os.add_dll_directory(bin_str))

        # rpy2 shells out to `R RHOME` when R_HOME is unset; setting it here
        # skips that subprocess. Forward slashes: a backslash-separated
        # R_HOME is rejected by the embedded interpreter's startup code.
        os.environ.setdefault("R_HOME", r_home.as_posix())

        _r_bin_dir = bin_dir
        _configured = True
        return bin_dir

    logger.debug("No R installation found; leaving the DLL search path untouched")
    _configured = True
    return None


# The R side of the check. Returns TRUE when airGR (and, implicitly, the base
# packages it depends on) can actually be loaded in this interpreter.
_AIRGR_LOADABLE = """
    tryCatch({
        suppressMessages(loadNamespace("airGR"))
        TRUE
    }, error = function(e) FALSE)
"""

_AIRGR_DIAGNOSIS = """
    paste(
        c(
            paste("R.home():", R.home()),
            paste(".libPaths():", paste(.libPaths(), collapse = ", ")),
            paste("error:", tryCatch({
                suppressMessages(loadNamespace("airGR"))
                "none"
            }, error = function(e) conditionMessage(e)))
        ),
        collapse = " | "
    )
"""

_airgr_verified = False


def ensure_airgr_available(robjects: Any, force: bool = False) -> None:
    """Raise unless the embedded R can load the ``airGR`` package.

    Called before any expensive work so that a broken R installation costs
    seconds instead of an hour of preprocessing followed by a segfault.

    Args:
        robjects: The imported ``rpy2.robjects`` module.
        force: Re-run the check even if it already passed in this process.

    Raises:
        ModelExecutionError: If ``airGR`` cannot be loaded.
    """
    global _airgr_verified

    if _airgr_verified and not force:
        return
    if robjects is None:
        raise ModelExecutionError(
            "GR models require R and rpy2, but rpy2 is not available in this process."
        )

    if bool(robjects.r(_AIRGR_LOADABLE)[0]):
        _airgr_verified = True
        return

    diagnosis = str(robjects.r(_AIRGR_DIAGNOSIS)[0])
    _raise_airgr_error(diagnosis)


def verify_gr_r_runtime() -> None:
    """Check the whole GR R stack (rpy2 + embedded R + airGR) in one call.

    Intended for entry points that would otherwise do a lot of work before
    touching R at all, such as GR preprocessing.

    Raises:
        ImportError: If rpy2 cannot be imported.
        ModelExecutionError: If the embedded R cannot load airGR.
    """
    configure_r_dll_search()

    import contextlib
    import io

    try:
        # rpy2 is noisy on stderr while it picks an interface mode.
        with contextlib.redirect_stderr(io.StringIO()):
            import rpy2.robjects as robjects
    except Exception as exc:  # noqa: BLE001 - Broad exception required for rpy2 import failures
        # Importing rpy2 starts the embedded interpreter, so a broken R shows
        # up here in many shapes: ImportError, OSError, RuntimeError, and — when
        # `R CMD config` produces no output because no POSIX shell is on PATH —
        # an IndexError from deep inside rpy2.situation.
        hint = (
            " On Windows, rpy2 runs `R CMD config --ldflags` while it starts up; "
            "that needs a POSIX shell and make (e.g. Rtools or MSYS2) on PATH, and "
            "returns nothing without them."
            if sys.platform == "win32"
            else ""
        )
        # ModelExecutionError, not ImportError: this runs from the GR
        # preprocessing path, not at module import, nothing catches
        # ImportError around it, and the sibling _raise_airgr_error already
        # reports the same class of failure this way.
        raise ModelExecutionError(
            "GR models require a working R and rpy2, but the embedded R could not "
            f"be started ({type(exc).__name__}: {exc})."
            f"{hint}"
            " Install R and rpy2, or use a different model. "
            "See https://rpy2.github.io/doc/latest/html/overview.html#installation"
        ) from exc

    ensure_airgr_available(robjects)


def _raise_airgr_error(diagnosis: str) -> None:
    """Raise the shared, actionable 'airGR is unusable' error."""
    raise ModelExecutionError(
        "The airGR R package could not be loaded by the embedded R interpreter, "
        "so the GR model cannot run.\n"
        f"Embedded R reports: {diagnosis}\n"
        "If airGR is listed under .libPaths() and standalone `Rscript -e 'library(airGR)'` "
        "works, this is a shared-library resolution problem rather than a missing package: "
        "on Windows, R's bin directory (R_HOME/bin/x64, holding Rblas.dll and Rlapack.dll) "
        "must be on PATH for the embedded interpreter to dyn.load() compiled R packages. "
        "Otherwise install it with: Rscript -e 'install.packages(\"airGR\")'"
    )
