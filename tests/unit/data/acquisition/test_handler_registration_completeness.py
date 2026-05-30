# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Completeness tests for data-handler registration.

Handlers are imported through *explicit* lists, not auto-discovery:

* ``acquisition/handlers/__init__.py`` imports every module named in its
  ``_handler_modules`` list (and swallows import errors at debug level).
* ``observation/handlers/__init__.py`` imports its handler classes explicitly.

The failure mode these tests guard against: a developer drops a correctly
``@register``-decorated handler file into the directory but forgets to add it to
the import list. The decorator then never runs, so the handler is silently
absent from the registry — with no error anywhere. These tests cross-check the
files on disk against what actually got registered, so that omission fails CI
instead of failing silently at runtime.
"""

import re
from pathlib import Path

import pytest

import symfluence.data.acquisition  # noqa: F401 - triggers handler imports
import symfluence.data.observation.handlers  # noqa: F401 - triggers handler imports
from symfluence.data.acquisition import handlers as acq_handlers
from symfluence.data.acquisition.registry import AcquisitionRegistry
from symfluence.data.observation.registry import ObservationRegistry

_ACQ_HANDLER_DIR = Path(acq_handlers.__file__).parent
_OBS_HANDLER_DIR = Path(symfluence.data.observation.handlers.__file__).parent


def _modules_with_decorator(handler_dir: Path, decorator: str) -> list[str]:
    """Return module stems whose source contains ``<decorator>.register(``."""
    found = []
    for path in sorted(handler_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        if f"{decorator}.register" in path.read_text(encoding="utf-8"):
            found.append(path.stem)
    return found


def _registered_names_in_file(path: Path, decorator: str) -> list[str]:
    """Extract the string keys passed to ``<decorator>.register('name')``."""
    pattern = rf"{decorator}\.register\(\s*['\"]([^'\"]+)['\"]"
    return re.findall(pattern, path.read_text(encoding="utf-8"))


@pytest.mark.data
class TestAcquisitionRegistrationCompleteness:
    """Every decorated acquisition handler must actually be imported."""

    def test_all_decorated_modules_are_in_import_list(self):
        """A handler file with @AcquisitionRegistry.register must be listed.

        Scans the handler directory for the register decorator and asserts each
        such module name appears in ``_handler_modules`` in ``__init__.py``.
        This is the exact "file exists but isn't imported" footgun.
        """
        init_src = (_ACQ_HANDLER_DIR / "__init__.py").read_text(encoding="utf-8")
        # Isolate the _handler_modules list literal.
        list_body = init_src.split("_handler_modules", 1)[1].split("]", 1)[0]
        listed = set(re.findall(r"['\"]([a-zA-Z0-9_]+)['\"]", list_body))

        decorated = _modules_with_decorator(_ACQ_HANDLER_DIR, "AcquisitionRegistry")
        missing = sorted(m for m in decorated if m not in listed)

        assert not missing, (
            "These acquisition handler modules use @AcquisitionRegistry.register "
            "but are NOT in _handler_modules in handlers/__init__.py, so they "
            f"will never register: {missing}. Add them to the list."
        )

    def test_import_list_has_no_silent_failures(self):
        """No module in the import list should have failed to import.

        ``handlers/__init__.py`` records failed imports in ``_failed`` and
        swallows the error at debug level. With the dev dependencies installed,
        every listed handler should import cleanly.
        """
        failed = getattr(acq_handlers, "_failed", [])
        assert not failed, (
            "Acquisition handler modules failed to import (registration "
            f"silently skipped): {failed}"
        )

    def test_decorated_names_are_retrievable(self):
        """Each registered key resolves back to a handler class."""
        datasets = {name.lower() for name in AcquisitionRegistry.list_datasets()}
        for module in _modules_with_decorator(_ACQ_HANDLER_DIR, "AcquisitionRegistry"):
            names = _registered_names_in_file(
                _ACQ_HANDLER_DIR / f"{module}.py", "AcquisitionRegistry"
            )
            for name in names:
                assert name.lower() in datasets, (
                    f"{module}.py registers '{name}' but it is not in "
                    "AcquisitionRegistry.list_datasets()"
                )


@pytest.mark.data
class TestObservationRegistrationCompleteness:
    """Every decorated observation handler must actually be imported."""

    def test_all_decorated_names_are_registered(self):
        """A handler file with @ObservationRegistry.register must be importable.

        The observation package uses explicit ``from .module import Handler``
        imports; a new handler that isn't added there never registers.
        """
        missing = []
        for path in sorted(_OBS_HANDLER_DIR.glob("*.py")):
            if path.name == "__init__.py":
                continue
            for name in _registered_names_in_file(path, "ObservationRegistry"):
                if not ObservationRegistry.is_registered(name):
                    missing.append(f"{path.stem}:{name}")

        assert not missing, (
            "These observation handlers use @ObservationRegistry.register but "
            "are not imported by observation/handlers/__init__.py, so they will "
            f"never register: {missing}. Add the import."
        )
