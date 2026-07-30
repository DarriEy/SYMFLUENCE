# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Registry isolation for tests.

A test that registers a fake component must not leak it into the next test, and
must not have the real component imported over the top of it mid-test. Getting
that right requires knowing how ``Registry`` populates, which is not obvious and
has already cost this project two silent failures — so it lives here as a
supported surface rather than being reimplemented in each repository's conftest.

``symfluence-models`` cannot reasonably reach into ``Registry._entries`` and the
other private attributes this needs; that is the whole reason this module is
public.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Set, Tuple

from symfluence.core.registries import Registries

__all__ = ["registry_snapshot"]

# (entries, meta, aliases, modules, loaded_modules) for one Registry.
_Saved = Tuple[
    Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, str], List[str], Set[str]
]


#: Guard against a pathological declaration cycle in :func:`_spend_lazy_population`.
#: Two passes is the normal cost (one to populate, one to observe no change);
#: anything beyond a handful means declarations are generating each other.
_MAX_SPEND_PASSES = 10


def _fingerprint(registries: Dict[str, Any]) -> Tuple[Tuple[int, int, int], ...]:
    """Cheap structural summary used to detect a fixed point."""
    return tuple(
        (len(reg._entries), len(reg._modules), len(reg._loaded_modules))
        for _, reg in sorted(registries.items())
    )


def _spend_lazy_population(registries: Dict[str, Any]) -> None:
    """Drive every registry to a fixed point before any of them is snapshotted.

    Registries fill in lazily from two sources: a one-shot ``_seeder`` that fires
    on the first READ, and declared side-effect modules imported by
    ``load_modules()``. Both must be fully spent first, for two independent
    reasons:

    1. Snapshotting an unpopulated registry captures nothing, so the restore
       reinstates emptiness — and the entries can never come back, because the
       ``@R.<registry>.add`` decorators only fire on a module's *first* import
       and it is in ``sys.modules`` by then. The registry stays empty for the
       rest of the session.
    2. A test that registers a fake and then reads it would otherwise trigger the
       seeder mid-test, importing the real component over the top of the fake
       between the ``add`` and the ``get``.

    Both have actually happened here: (1) is why a registry-snapshot fixture once
    made every later test see an empty registry, and (2) is what broke
    ``test_lowercase_delineation`` and ``test_lowercase_data_registries`` — the
    real ``LumpedWatershedDelineator`` replacing the fake.

    A single pass over the registries is NOT enough, which is the subtle part.
    Importing one registry's declared modules can append modules to a registry
    that was already visited earlier in the iteration; those stay unspent, so a
    read inside the snapshotted block imports them and their decorators fire
    *after* the clear. That is observable: GNN and LSTM reappeared inside a
    supposedly-empty registry this way. Loop until nothing changes.
    """
    for _ in range(_MAX_SPEND_PASSES):
        before = _fingerprint(registries)
        for registry in registries.values():
            registry._ensure_seeded()
            registry.load_modules()
        if _fingerprint(registries) == before:
            return
    raise RuntimeError(
        'registry population did not reach a fixed point in '
        f'{_MAX_SPEND_PASSES} passes; a declared module is very likely '
        'declaring further modules on import'
    )


def _save(registry: Any) -> _Saved:
    return (
        dict(registry._entries),
        dict(registry._meta),
        dict(registry._aliases),
        list(registry._modules),
        set(registry._loaded_modules),
    )


def _restore(registry: Any, saved: _Saved) -> None:
    entries, meta, aliases, modules, loaded = saved
    registry.clear()
    registry._entries.update(entries)
    registry._meta.update(meta)
    registry._aliases.update(aliases)
    registry._modules[:] = modules
    registry._loaded_modules.clear()
    registry._loaded_modules.update(loaded)


@contextmanager
def registry_snapshot(*, clear: bool = True) -> Iterator[None]:
    """Save every registry, then restore it exactly on exit.

    Args:
        clear: When True (the default) the registries are emptied inside the
            block, so a test starts from a blank slate and sees only what it
            registers itself. Pass False to keep the real components registered
            and still have any additions rolled back on exit.

    Lazy population is spent before snapshotting either way — see
    :func:`_spend_lazy_population` for why that ordering is load-bearing.

    Example:
        >>> with registry_snapshot():
        ...     R.runners.add('FAKE')(FakeRunner)
        ...     assert R.runners.get('FAKE') is FakeRunner
        >>> # registries are back to their real contents here
    """
    saved: Dict[str, _Saved] = {}
    registries = Registries.all_registries()

    # Populate everything to a fixed point BEFORE snapshotting any of it, so a
    # cross-registry declaration cannot import new entries in after the clear.
    _spend_lazy_population(registries)

    for name, registry in registries.items():
        saved[name] = _save(registry)
        if clear:
            registry.clear()
    try:
        yield
    finally:
        for name, registry in registries.items():
            _restore(registry, saved[name])
