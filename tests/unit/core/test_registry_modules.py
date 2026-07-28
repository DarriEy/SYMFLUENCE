# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""``Registry.add_module`` / ``load_modules`` — declared side-effect modules.

Some components register through a decorator that only fires when a particular
module is imported (forcing adapters, ``symfluence init`` presets).  Rather than
have the framework hunt for those modules on disk — impossible once the model
suite is a separate distribution, and blind to external plugin packages — the
owning package declares them and the consumer drains the declarations.
"""

from __future__ import annotations

import sys

import pytest

from symfluence.core.registry import Registry, model_manifest

pytestmark = [pytest.mark.unit, pytest.mark.quick]

_MISSING = "symfluence_probe_module_that_does_not_exist"


@pytest.fixture
def probe_module(tmp_path, monkeypatch):
    """Write an importable module whose *import* registers into a registry."""
    monkeypatch.syspath_prepend(str(tmp_path))
    created: list[str] = []

    def _make(name: str, body: str) -> str:
        (tmp_path / f"{name}.py").write_text(body, encoding="utf-8")
        created.append(name)
        return name

    yield _make
    for name in created:
        sys.modules.pop(name, None)


def test_declared_module_is_not_imported_until_drained(probe_module):
    reg: Registry = Registry("demo")
    name = probe_module(
        "symfluence_probe_lazy",
        "import symfluence_probe_lazy_marker\n",  # would fail if imported
    )
    reg.add_module(name)

    assert reg.declared_modules() == (name,)
    # Reads do not drain: draining is explicit and opt-in, so a plain registry
    # read returns exactly what was registered eagerly, as it does today.
    assert reg.keys() == []
    assert name not in sys.modules


def test_load_modules_imports_the_module_for_its_side_effects(probe_module):
    from symfluence.core.registries import R

    name = probe_module(
        "symfluence_probe_adapter",
        "from symfluence.core.registries import R\n"
        "@R.forcing_adapters.add('PROBEMODEL')\n"
        "class _ProbeAdapter:\n"
        "    pass\n",
    )
    try:
        R.forcing_adapters.add_module(name)
        assert "PROBEMODEL" not in R.forcing_adapters

        R.forcing_adapters.load_modules()

        assert "PROBEMODEL" in R.forcing_adapters
        assert R.forcing_adapters["PROBEMODEL"].__name__ == "_ProbeAdapter"
    finally:
        R.forcing_adapters.remove("PROBEMODEL")
        R.forcing_adapters._modules.remove(name)
        R.forcing_adapters._loaded_modules.discard(name)


def test_load_modules_tolerates_a_missing_module():
    reg: Registry = Registry("demo")
    reg.add_module(_MISSING)
    reg.load_modules()  # an optional capability may simply not be installed
    assert reg.keys() == []


def test_load_modules_imports_each_module_at_most_once(probe_module):
    """A second drain must not re-execute a module's registration."""
    name = probe_module(
        "symfluence_probe_counter",
        "import builtins\n"
        "builtins._symfluence_probe_counter = "
        "getattr(builtins, '_symfluence_probe_counter', 0) + 1\n",
    )
    import builtins

    reg: Registry = Registry("demo")
    reg.add_module(name)
    reg.load_modules()
    reg.load_modules()
    try:
        assert builtins._symfluence_probe_counter == 1
    finally:
        del builtins._symfluence_probe_counter


def test_add_module_is_idempotent():
    reg: Registry = Registry("demo")
    reg.add_module("a.b")
    reg.add_module("a.b")
    assert reg.declared_modules() == ("a.b",)


def test_add_module_respects_freeze():
    reg: Registry = Registry("demo")
    reg.freeze()
    with pytest.raises(RuntimeError):
        reg.add_module("a.b")


def test_model_manifest_declares_capability_modules():
    """The additive manifest fields are the declaration API for any package."""
    from symfluence.core.registries import R

    fa = "symfluence_probe_manifest.forcing_adapter"
    ip = "symfluence_probe_manifest.init_preset"
    try:
        model_manifest("MANIFESTPROBE", forcing_adapter_module=fa, init_preset_module=ip)

        assert fa in R.forcing_adapters.declared_modules()
        assert ip in R.presets.declared_modules()
        # Declaring is not importing: nothing loads until a consumer drains.
        assert fa not in sys.modules
    finally:
        R.forcing_adapters._modules.remove(fa)
        R.presets._modules.remove(ip)
