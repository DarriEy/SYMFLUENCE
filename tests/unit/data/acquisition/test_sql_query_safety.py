"""Regression tests for acquisition handlers that must construct SQL strings."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from symfluence.data.acquisition.handlers.gssurgo import GSSURGOAcquirer
from symfluence.data.acquisition.handlers.nws_hydrofabric import NWSHydrofabricAcquirer


def test_gssurgo_rejects_non_identifier_property_names():
    acquirer = object.__new__(GSSURGOAcquirer)
    acquirer._execute_sda_query = Mock()  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="valid identifiers"):
        acquirer._query_horizon_properties(
            Mock(), ["1"], ["sandtotal_r; DROP TABLE mapunit"], 0, 200
        )

    acquirer._execute_sda_query.assert_not_called()


def test_gssurgo_escapes_mukey_literals():
    acquirer = object.__new__(GSSURGOAcquirer)
    acquirer._execute_sda_query = Mock(return_value=pd.DataFrame())  # type: ignore[method-assign]

    acquirer._query_horizon_properties(
        Mock(), ["safe", "quote'value"], ["sandtotal_r"], 0, 200
    )

    query = acquirer._execute_sda_query.call_args.args[1]
    assert "'quote''value'" in query
    assert "'quote'value'" not in query


def test_hydrofabric_quotes_identifiers_and_literals(tmp_path):
    acquirer = object.__new__(NWSHydrofabricAcquirer)
    acquirer.logger = Mock()
    expected = Mock()

    with patch("geopandas.read_file", return_value=expected) as read_file:
        result = acquirer._read_by_ids(
            tmp_path / "fabric.gpkg",
            'layer"name',
            'id"column',
            {"wb-safe", "wb-'quoted"},
        )

    assert result is expected
    sql = read_file.call_args.kwargs["sql"]
    assert '"layer""name"' in sql
    assert '"id""column"' in sql
    assert "'wb-''quoted'" in sql


def test_hydrofabric_empty_id_set_does_not_query(tmp_path):
    acquirer = object.__new__(NWSHydrofabricAcquirer)

    with patch("geopandas.read_file") as read_file:
        assert acquirer._read_by_ids(tmp_path / "fabric.gpkg", "layer", "id", set()) is None

    read_file.assert_not_called()
