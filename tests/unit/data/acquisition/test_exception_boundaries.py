"""Ensure acquisition resilience does not hide programming errors."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from symfluence.data.acquisition.handlers.aridity_index import AridityIndexAcquirer
from symfluence.data.acquisition.handlers.bedrock_depth import BedrockDepthAcquirer
from symfluence.data.acquisition.handlers.cmc_snow import CMCSnowAcquirer


@pytest.mark.parametrize(
    ("handler_type", "args"),
    [
        (AridityIndexAcquirer, ("ai", "dimensionless")),
        (BedrockDepthAcquirer, ()),
    ],
)
def test_summary_helpers_propagate_unexpected_programming_errors(tmp_path, handler_type, args):
    handler = object.__new__(handler_type)
    handler.logger = Mock()

    with patch("rasterio.open", side_effect=TypeError("programming defect")):
        with pytest.raises(TypeError, match="programming defect"):
            handler._log_summary(tmp_path / "data.tif", *args)


def test_cmc_download_retry_propagates_unexpected_programming_errors(tmp_path):
    handler = object.__new__(CMCSnowAcquirer)
    handler.logger = Mock()
    session = Mock()
    session.get.side_effect = TypeError("invalid session contract")

    with pytest.raises(TypeError, match="invalid session contract"):
        handler._download_year(session, 2020, tmp_path, force=True)
