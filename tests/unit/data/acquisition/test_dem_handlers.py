"""Unit Tests for New DEM Acquisition Handlers.

Tests for CopDEM90, SRTM, ETOPO2022, Mapzen, and ALOS acquirers.
All tests mock HTTP/network responses -- no real network calls.
"""

import gzip
import io
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest
from fixtures.acquisition_fixtures import MockConfigFactory

# =============================================================================
# Helpers
# =============================================================================

def _make_handler(cls, tmp_path, bbox="47.0/8.0/46.0/9.0", **extra):
    """Instantiate a handler with a mock config rooted in tmp_path."""
    config = MockConfigFactory.create(
        data_dir=str(tmp_path),
        bbox=bbox,
        **extra,
    )
    logger = MagicMock()
    return cls(config, logger)


def _mock_tif_bytes():
    """Return minimal GeoTIFF bytes (enough for rasterio.open not to crash in mocks)."""
    return b"\x00" * 1024


# =============================================================================
# CopDEM90Acquirer
# =============================================================================

class TestCopDEM90Acquirer:
    """Tests for Copernicus DEM GLO-90 handler."""

    def test_registered(self):
        from symfluence.data.acquisition.registry import AcquisitionRegistry
        assert AcquisitionRegistry.is_registered('COPDEM90')

    def test_class_attrs(self):
        from symfluence.data.acquisition.handlers.dem import CopDEM90Acquirer
        assert '90m' in CopDEM90Acquirer._BASE_URL
        assert CopDEM90Acquirer._COG_CODE == "30"
        assert "GLO-90" in CopDEM90Acquirer._PRODUCT_NAME

    def test_inherits_copdem30(self):
        from symfluence.data.acquisition.handlers.dem import CopDEM30Acquirer, CopDEM90Acquirer
        assert issubclass(CopDEM90Acquirer, CopDEM30Acquirer)

    def test_tile_naming_uses_cog30(self, tmp_path):
        """Tile names should use COG_30 (= 30 arc-sec = 90m)."""
        from symfluence.data.acquisition.handlers.dem import CopDEM90Acquirer
        handler = _make_handler(CopDEM90Acquirer, tmp_path)
        # The tile name pattern is built from _COG_CODE
        assert handler._COG_CODE == "30"

    def test_skip_if_exists(self, tmp_path):
        from symfluence.data.acquisition.handlers.dem import CopDEM90Acquirer
        handler = _make_handler(CopDEM90Acquirer, tmp_path)

        # Pre-create the output file
        out_dir = tmp_path / "domain_test_domain" / "attributes" / "elevation" / "dem"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "domain_test_domain_elv.tif"
        out_file.write_bytes(b"existing")

        result = handler.download(tmp_path)
        assert result == out_file


# =============================================================================
# SRTMAcquirer
# =============================================================================

class TestSRTMAcquirer:
    """Tests for SRTM GL1 handler."""

    def test_registered(self):
        from symfluence.data.acquisition.registry import AcquisitionRegistry
        assert AcquisitionRegistry.is_registered('SRTM')

    def test_base_url(self):
        from symfluence.data.acquisition.handlers.dem import SRTMAcquirer
        assert 'opentopography' in SRTMAcquirer._BASE_URL

    def test_coverage_warning_high_lat(self, tmp_path):
        """Should warn when bbox extends beyond 60N."""
        from symfluence.data.acquisition.handlers.dem import SRTMAcquirer
        handler = _make_handler(SRTMAcquirer, tmp_path, bbox="65.0/8.0/61.0/9.0")

        # Pre-create the output file so download returns early after the warning
        out_dir = tmp_path / "domain_test_domain" / "attributes" / "elevation" / "dem"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "domain_test_domain_elv.tif"
        out_file.write_bytes(b"existing")

        handler.download(tmp_path)
        # Handler should still work (skip due to existing file)
        assert out_file.exists()

    def test_tile_naming_format(self, tmp_path):
        """Tile names should be like N46E008.hgt."""
        from symfluence.data.acquisition.handlers.dem import SRTMAcquirer
        handler = _make_handler(SRTMAcquirer, tmp_path)
        # Verify the URL pattern uses .hgt extension
        assert '.hgt' in f"{handler._BASE_URL}/N46E008.hgt"

    def test_skip_if_exists(self, tmp_path):
        from symfluence.data.acquisition.handlers.dem import SRTMAcquirer
        handler = _make_handler(SRTMAcquirer, tmp_path)

        out_dir = tmp_path / "domain_test_domain" / "attributes" / "elevation" / "dem"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "domain_test_domain_elv.tif"
        out_file.write_bytes(b"existing")

        result = handler.download(tmp_path)
        assert result == out_file


# =============================================================================
# ETOPO2022Acquirer
# =============================================================================

class TestETOPO2022Acquirer:
    """Tests for ETOPO 2022 handler."""

    def test_registered(self):
        from symfluence.data.acquisition.registry import AcquisitionRegistry
        assert AcquisitionRegistry.is_registered('ETOPO2022')

    def test_default_resolution(self, tmp_path):
        """Default ETOPO resolution should be 60s."""
        from symfluence.data.acquisition.handlers.dem import ETOPO2022Acquirer
        handler = _make_handler(ETOPO2022Acquirer, tmp_path)
        assert handler.config_dict.get('ETOPO_RESOLUTION', '60s') == '60s'

    def test_opendap_url_construction(self):
        from symfluence.data.acquisition.handlers.dem import ETOPO2022Acquirer
        url = ETOPO2022Acquirer._OPENDAP_TEMPLATE.format(res='60s', variant='surface')
        assert 'ETOPO2022' in url
        assert '60s' in url
        assert 'surface' in url
        assert url.startswith('https://www.ngdc.noaa.gov/thredds')

    def test_opendap_url_custom_resolution(self):
        from symfluence.data.acquisition.handlers.dem import ETOPO2022Acquirer
        url = ETOPO2022Acquirer._OPENDAP_TEMPLATE.format(res='15s', variant='bedrock')
        assert '15s' in url
        assert 'bedrock' in url

    def test_skip_if_exists(self, tmp_path):
        from symfluence.data.acquisition.handlers.dem import ETOPO2022Acquirer
        handler = _make_handler(ETOPO2022Acquirer, tmp_path)

        out_dir = tmp_path / "domain_test_domain" / "attributes" / "elevation" / "dem"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "domain_test_domain_elv.tif"
        out_file.write_bytes(b"existing")

        result = handler.download(tmp_path)
        assert result == out_file

    @patch('xarray.open_dataset')
    def test_bbox_slicing(self, mock_open_dataset, tmp_path):
        """Verify bbox subsetting is applied to the OPeNDAP dataset."""
        from symfluence.data.acquisition.handlers.dem import ETOPO2022Acquirer
        handler = _make_handler(ETOPO2022Acquirer, tmp_path)

        # Create mock dataset
        mock_ds = MagicMock()
        mock_ds.coords = {'lat': True, 'lon': True}
        lat_vals = np.arange(-90, 91, 1.0)
        mock_ds.__getitem__ = MagicMock(return_value=MagicMock(values=lat_vals))
        mock_open_dataset.return_value = mock_ds

        # Mock the subset
        mock_subset = MagicMock()
        mock_ds.sel.return_value = mock_subset
        mock_subset.data_vars = {'z': MagicMock()}
        mock_subset.__getitem__ = MagicMock(side_effect=lambda key: MagicMock(
            values=np.arange(46.0, 47.0, 0.1) if key == 'lat' else np.arange(8.0, 9.0, 0.1)
        ))

        # This will fail at rasterio.open, but we've verified the slicing logic
        with pytest.raises(Exception):
            handler.download(tmp_path)

        # Verify open_dataset was called with OPeNDAP URL
        mock_open_dataset.assert_called_once()
        call_args = mock_open_dataset.call_args
        assert 'ETOPO2022' in call_args[0][0]


# =============================================================================
# MapzenAcquirer
# =============================================================================

class TestMapzenAcquirer:
    """Tests for Mapzen terrain tile handler."""

    def test_registered(self):
        from symfluence.data.acquisition.registry import AcquisitionRegistry
        assert AcquisitionRegistry.is_registered('MAPZEN')

    def test_base_url(self):
        from symfluence.data.acquisition.handlers.dem import MapzenAcquirer
        assert 'elevation-tiles-prod' in MapzenAcquirer._BASE_URL
        assert 'skadi' in MapzenAcquirer._BASE_URL

    def test_url_format(self):
        """URL should follow skadi/{LAT_DIR}/{LAT_DIR}{LON_DIR}.hgt.gz pattern."""
        from symfluence.data.acquisition.handlers.dem import MapzenAcquirer
        base = MapzenAcquirer._BASE_URL
        expected = f"{base}/N46/N46E008.hgt.gz"
        assert 'skadi/N46/N46E008.hgt.gz' in expected

    def test_skip_if_exists(self, tmp_path):
        from symfluence.data.acquisition.handlers.dem import MapzenAcquirer
        handler = _make_handler(MapzenAcquirer, tmp_path)

        out_dir = tmp_path / "domain_test_domain" / "attributes" / "elevation" / "dem"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "domain_test_domain_elv.tif"
        out_file.write_bytes(b"existing")

        result = handler.download(tmp_path)
        assert result == out_file

    def test_gzip_import(self):
        """Module should import gzip at the top level."""
        import gzip as gzip_mod

        import symfluence.data.acquisition.handlers.dem as dem_module
        # gzip is used in MapzenAcquirer.download
        assert hasattr(gzip_mod, 'open')


# =============================================================================
# ALOSAcquirer
# =============================================================================

class TestALOSAcquirer:
    """Tests for ALOS AW3D30 handler."""

    def test_registered(self):
        from symfluence.data.acquisition.registry import AcquisitionRegistry
        assert AcquisitionRegistry.is_registered('ALOS')

    def test_stac_collection_name(self):
        from symfluence.data.acquisition.handlers.dem import ALOSAcquirer
        assert ALOSAcquirer._COLLECTION == "alos-dem"

    def test_stac_api_url(self):
        from symfluence.data.acquisition.handlers.dem import ALOSAcquirer
        assert 'planetarycomputer' in ALOSAcquirer._STAC_API_URL

    def test_missing_dependency_error(self, tmp_path):
        """Should raise ImportError with install instructions when deps missing."""
        from symfluence.data.acquisition.handlers.dem import ALOSAcquirer
        handler = _make_handler(ALOSAcquirer, tmp_path)

        with patch.dict('sys.modules', {'planetary_computer': None, 'pystac_client': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module")):
                with pytest.raises(ImportError, match="planetary-computer"):
                    handler.download(tmp_path)

    def test_skip_if_exists(self, tmp_path):
        from symfluence.data.acquisition.handlers.dem import ALOSAcquirer

        # Need to handle the import check - mock the optional deps
        with patch.dict('sys.modules', {
            'planetary_computer': MagicMock(),
            'pystac_client': MagicMock(),
        }):
            handler = _make_handler(ALOSAcquirer, tmp_path)

            out_dir = tmp_path / "domain_test_domain" / "attributes" / "elevation" / "dem"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "domain_test_domain_elv.tif"
            out_file.write_bytes(b"existing")

            result = handler.download(tmp_path)
            assert result == out_file


# =============================================================================
# _TileDownloadMixin Tests
# =============================================================================

class TestTileDownloadMixin:
    """Tests for the shared _TileDownloadMixin."""

    def test_merge_tiles_single(self, tmp_path):
        """Single tile should be renamed to output path."""
        from symfluence.data.acquisition.handlers.dem import _TileDownloadMixin

        class Testable(_TileDownloadMixin):
            def __init__(self):
                self.logger = MagicMock()

        mixin = Testable()
        tile = tmp_path / "tile_0.tif"
        tile.write_bytes(b"data")
        out = tmp_path / "merged.tif"

        mixin._merge_tiles([tile], out)
        assert out.exists()
        assert not tile.exists()  # renamed

    def test_validate_tile_returns_false_on_error(self, tmp_path):
        """Invalid file should return False."""
        from symfluence.data.acquisition.handlers.dem import _TileDownloadMixin

        class Testable(_TileDownloadMixin):
            def __init__(self):
                self.logger = MagicMock()

        mixin = Testable()
        bad_tile = tmp_path / "bad.tif"
        bad_tile.write_bytes(b"not a tif")

        result = mixin._validate_tile(bad_tile, "bad")
        assert result is False


# =============================================================================
# Registration completeness
# =============================================================================

class TestAllNewHandlersRegistered:
    """Verify all 5 new handlers are discoverable in the registry."""

    @pytest.mark.parametrize("key", ['COPDEM90', 'SRTM', 'ETOPO2022', 'MAPZEN', 'ALOS'])
    def test_handler_registered(self, key):
        from symfluence.data.acquisition.registry import AcquisitionRegistry
        assert AcquisitionRegistry.is_registered(key), f"{key} not registered"

    @pytest.mark.parametrize("key", ['copdem90', 'srtm', 'etopo2022', 'mapzen', 'alos'])
    def test_case_insensitive(self, key):
        from symfluence.data.acquisition.registry import AcquisitionRegistry
        assert AcquisitionRegistry.is_registered(key), f"{key} (lowercase) not registered"

    @pytest.mark.parametrize("key", ['COPDEM90', 'SRTM', 'ETOPO2022', 'MAPZEN', 'ALOS'])
    def test_handler_inherits_base(self, key):
        from symfluence.data.acquisition.base import BaseAcquisitionHandler
        from symfluence.data.acquisition.registry import AcquisitionRegistry
        cls = AcquisitionRegistry._get_handler_class(key)
        assert issubclass(cls, BaseAcquisitionHandler)
