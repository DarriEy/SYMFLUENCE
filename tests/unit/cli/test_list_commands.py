# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for `symfluence list` — registry/config introspection."""
from __future__ import annotations

from argparse import Namespace

import pytest

from symfluence.cli.commands.list_commands import LIST_KINDS, ListCommands, _catalog
from symfluence.cli.exit_codes import ExitCode


def test_catalog_covers_all_kinds_and_is_live():
    catalog = _catalog()
    assert set(catalog) == set(LIST_KINDS)
    # These registries are always populated in a normal install.
    assert catalog['models'], "no models registered"
    assert catalog['forcings'], "no forcing handlers registered"
    assert catalog['optimizers'], "no optimizers registered"
    assert catalog['config-keys'], "no config keys"
    # Catalogs are plain string name lists.
    assert all(isinstance(name, str) for name in catalog['models'])


def test_list_overview_returns_success():
    assert ListCommands.list_items(Namespace(kind=None)) == ExitCode.SUCCESS


@pytest.mark.parametrize('kind', LIST_KINDS)
def test_list_each_kind_succeeds(kind):
    assert ListCommands.list_items(Namespace(kind=kind)) == ExitCode.SUCCESS


def test_list_unknown_kind_is_usage_error():
    assert ListCommands.list_items(Namespace(kind='nonsense')) == ExitCode.USAGE_ERROR
