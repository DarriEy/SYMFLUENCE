# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Completeness tests for data-handler registration.

Handlers are imported through *explicit* lists, not auto-discovery:

* ``acquisition/handlers/__init__.py`` imports every module named in its
  ``_handler_modules`` list (and swallows import errors at debug level).
* ``observation/handlers/__init__.py`` imports its handler classes explicitly
  via ``from .<module> import <Handler>``.

The failure mode these tests guard against: a developer drops a correctly
``@register``-decorated handler file into the directory but forgets to wire it
into the import list. The decorator then never runs, so the handler is silently
absent from the registry — with no error anywhere. This is exactly how the
``modis.py`` (``MODIS_SNOW``) handler became unreachable.

These checks are deliberately **static** (they read source files, they do not
import handler modules). That keeps them deterministic and independent of which
optional acquisition dependencies happen to be installed in the test
environment — a handler whose heavy dependency (e.g. ``cdsapi``) is absent
still *should* be wired into the import list, and that is what we verify.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# Locate the handler directories from the top-level package path, WITHOUT
# importing the handler subpackages. The observation handlers' __init__ uses
# plain (non-guarded) ``from .<module> import`` statements, so importing it
# would raise if an optional dependency is missing in the test environment —
# which would turn these static checks into environment-dependent failures.
# Deriving paths from ``symfluence.__file__`` avoids that entirely.
import symfluence

_SRC_ROOT = Path(symfluence.__file__).parent
_ACQ_HANDLER_DIR = _SRC_ROOT / "data" / "acquisition" / "handlers"
_OBS_HANDLER_DIR = _SRC_ROOT / "data" / "observation" / "handlers"


def _modules_with_decorator(handler_dir: Path, registry_namespace: str) -> list[str]:
    """Return module stems whose source contains ``@R.<namespace>.add(``."""
    found = []
    for path in sorted(handler_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        if f"R.{registry_namespace}.add(" in path.read_text(encoding="utf-8"):
            found.append(path.stem)
    return found


@pytest.mark.data
class TestAcquisitionRegistrationCompleteness:
    """Every decorated acquisition handler must be in the import list."""

    def test_all_decorated_modules_are_in_import_list(self):
        """A handler file with @R.acquisition_handlers.add must be listed.

        Scans the handler directory for the register decorator and asserts each
        such module name appears in ``_handler_modules`` in ``__init__.py``.
        This is the exact "file exists but isn't imported" footgun. Purely
        static — does not import any handler module.
        """
        init_src = (_ACQ_HANDLER_DIR / "__init__.py").read_text(encoding="utf-8")
        # Isolate the _handler_modules list literal.
        list_body = init_src.split("_handler_modules", 1)[1].split("]", 1)[0]
        listed = set(re.findall(r"['\"]([a-zA-Z0-9_]+)['\"]", list_body))

        decorated = _modules_with_decorator(_ACQ_HANDLER_DIR, "acquisition_handlers")
        missing = sorted(m for m in decorated if m not in listed)

        assert not missing, (
            "These acquisition handler modules use @R.acquisition_handlers.add "
            "but are NOT in _handler_modules in handlers/__init__.py, so they "
            f"will never register: {missing}. Add them to the list."
        )


@pytest.mark.data
class TestObservationRegistrationCompleteness:
    """Every decorated observation handler must be imported by __init__.

    Note: the observation package imports handlers explicitly via
    ``from .<module> import <Handler>``, but a decorated module may also register
    transitively (imported by another module that __init__ imports). To avoid
    false positives we only flag a module as un-wired when *neither* a direct
    relative import of that module *nor* an import of its handler class name
    appears anywhere in ``__init__.py``.
    """

    def test_all_decorated_modules_are_reachable(self):
        """A handler file with @R.observation_handlers.add must be reachable.

        Purely static: for each decorated observation module, require that its
        module stem is referenced by a relative import in ``__init__.py``. This
        catches the "added a handler file but forgot to import it" footgun
        without importing any handler module (so it is independent of which
        optional dependencies are installed).
        """
        init_src = (_OBS_HANDLER_DIR / "__init__.py").read_text(encoding="utf-8")
        decorated = _modules_with_decorator(_OBS_HANDLER_DIR, "observation_handlers")

        missing = []
        for module in decorated:
            # `from .module import ...` (also covers the multiline `import (` form).
            if not re.search(rf"from\s+\.{re.escape(module)}\s+import", init_src):
                missing.append(module)

        assert not missing, (
            "These observation handler modules use @R.observation_handlers.add "
            "but are NOT imported in observation/handlers/__init__.py, so they "
            f"will never register: {missing}. Add a `from .<module> import ...`."
        )
