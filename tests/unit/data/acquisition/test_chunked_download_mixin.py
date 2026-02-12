"""
Unit Tests for ChunkedDownloadMixin.

Tests the chunked download functionality:
- generate_temporal_chunks(): single month, multi-month, yearly, empty range
- generate_year_month_list(): correct (year, month) tuples
- download_chunks_parallel(): success, fail-fast behavior
- merge_netcdf_chunks(): merge, time slice, cleanup
- get_netcdf_encoding(): compression settings
"""

from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, Mock, patch
import concurrent.futures

import numpy as np
import pandas as pd
import pytest
import xarray as xr


# NOTE: macOS ARM HDF5/netCDF4 skip was removed as of Jan 2026.
# The underlying HDF5 attribute issues have been resolved in recent library versions.


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def chunked_mixin():
    """Create an instance of ChunkedDownloadMixin for testing."""
    from symfluence.data.acquisition.mixins.chunked import ChunkedDownloadMixin

    class TestableChunkedMixin(ChunkedDownloadMixin):
        def __init__(self):
            self.logger = MagicMock()

    return TestableChunkedMixin()


@pytest.fixture
def sample_netcdf_dataset():
    """Create a sample xarray dataset for testing."""
    time = pd.date_range("2020-01-01", periods=168, freq="h")  # 1 week
    lat = np.arange(46.0, 47.1, 0.25)
    lon = np.arange(8.0, 9.1, 0.25)

    data = np.random.rand(len(time), len(lat), len(lon))

    ds = xr.Dataset(
        {"temperature": (["time", "lat", "lon"], data.astype(np.float32))},
        coords={"time": time, "lat": lat, "lon": lon}
    )

    return ds


# =============================================================================
# Generate Temporal Chunks Tests
# =============================================================================

@pytest.mark.mixin_chunked
@pytest.mark.acquisition
class TestGenerateTemporalChunks:
    """Tests for generate_temporal_chunks method."""

    def test_single_month_chunk(self, chunked_mixin):
        """Single month range should return one chunk."""
        start = pd.Timestamp("2020-01-01")
        end = pd.Timestamp("2020-01-31")

        chunks = chunked_mixin.generate_temporal_chunks(start, end, freq='MS')

        assert len(chunks) == 1
        assert chunks[0][0] == start
        assert chunks[0][1] == end

    def test_multi_month_chunks(self, chunked_mixin):
        """Multi-month range should return multiple chunks."""
        start = pd.Timestamp("2020-01-15")
        end = pd.Timestamp("2020-03-20")

        chunks = chunked_mixin.generate_temporal_chunks(start, end, freq='MS')

        assert len(chunks) == 3

        # First chunk: Jan 15 - Jan 31
        assert chunks[0][0] == pd.Timestamp("2020-01-15")
        assert chunks[0][1] == pd.Timestamp("2020-01-31")

        # Second chunk: Feb 1 - Feb 29 (2020 is leap year)
        assert chunks[1][0] == pd.Timestamp("2020-02-01")
        assert chunks[1][1] == pd.Timestamp("2020-02-29")

        # Third chunk: Mar 1 - Mar 20
        assert chunks[2][0] == pd.Timestamp("2020-03-01")
        assert chunks[2][1] == pd.Timestamp("2020-03-20")

    def test_yearly_chunks(self, chunked_mixin):
        """Yearly frequency should create year-sized chunks."""
        start = pd.Timestamp("2018-06-01")
        end = pd.Timestamp("2020-06-30")

        chunks = chunked_mixin.generate_temporal_chunks(start, end, freq='YS')

        assert len(chunks) == 3

        # 2018 chunk
        assert chunks[0][0] == pd.Timestamp("2018-06-01")

        # 2020 chunk ends at end date
        assert chunks[-1][1] == pd.Timestamp("2020-06-30")

    def test_empty_range_returns_empty(self, chunked_mixin):
        """Start after end should return empty list."""
        start = pd.Timestamp("2020-03-01")
        end = pd.Timestamp("2020-01-01")

        chunks = chunked_mixin.generate_temporal_chunks(start, end, freq='MS')

        assert chunks == []

    def test_same_day_range(self, chunked_mixin):
        """Same start and end date should return one chunk."""
        date = pd.Timestamp("2020-01-15")

        chunks = chunked_mixin.generate_temporal_chunks(date, date, freq='MS')

        assert len(chunks) == 1
        assert chunks[0] == (date, date)

    def test_partial_start_month(self, chunked_mixin):
        """Start mid-month should correctly bound first chunk."""
        start = pd.Timestamp("2020-01-20")
        end = pd.Timestamp("2020-02-15")

        chunks = chunked_mixin.generate_temporal_chunks(start, end, freq='MS')

        assert len(chunks) == 2
        assert chunks[0][0] == pd.Timestamp("2020-01-20")
        assert chunks[1][1] == pd.Timestamp("2020-02-15")

    def test_daily_chunks(self, chunked_mixin):
        """Daily frequency should create day-sized chunks."""
        start = pd.Timestamp("2020-01-01")
        end = pd.Timestamp("2020-01-03")

        chunks = chunked_mixin.generate_temporal_chunks(start, end, freq='D')

        assert len(chunks) == 3


# =============================================================================
# Generate Year Month List Tests
# =============================================================================

@pytest.mark.mixin_chunked
@pytest.mark.acquisition
class TestGenerateYearMonthList:
    """Tests for generate_year_month_list method."""

    def test_single_month(self, chunked_mixin):
        """Single month should return one (year, month) tuple."""
        start = pd.Timestamp("2020-01-15")
        end = pd.Timestamp("2020-01-31")

        ym_list = chunked_mixin.generate_year_month_list(start, end)

        assert ym_list == [(2020, 1)]

    def test_cross_year_boundary(self, chunked_mixin):
        """Range crossing year should include all months."""
        start = pd.Timestamp("2020-11-01")
        end = pd.Timestamp("2021-02-28")

        ym_list = chunked_mixin.generate_year_month_list(start, end)

        expected = [(2020, 11), (2020, 12), (2021, 1), (2021, 2)]
        assert ym_list == expected

    def test_same_month_partial(self, chunked_mixin):
        """Partial month should return that month."""
        start = pd.Timestamp("2020-03-15")
        end = pd.Timestamp("2020-03-20")

        ym_list = chunked_mixin.generate_year_month_list(start, end)

        assert ym_list == [(2020, 3)]

    def test_multi_year_range(self, chunked_mixin):
        """Multi-year range should include all months."""
        start = pd.Timestamp("2019-01-01")
        end = pd.Timestamp("2020-12-31")

        ym_list = chunked_mixin.generate_year_month_list(start, end)

        assert len(ym_list) == 24  # 2 years * 12 months
        assert ym_list[0] == (2019, 1)
        assert ym_list[-1] == (2020, 12)

    def test_end_month_included(self, chunked_mixin):
        """End month should always be included."""
        start = pd.Timestamp("2020-01-01")
        end = pd.Timestamp("2020-03-15")

        ym_list = chunked_mixin.generate_year_month_list(start, end)

        assert (2020, 3) in ym_list


# =============================================================================
# Download Chunks Parallel Tests
# =============================================================================

@pytest.mark.mixin_chunked
@pytest.mark.acquisition
class TestDownloadChunksParallel:
    """Tests for download_chunks_parallel method."""

    def test_successful_download_all_chunks(self, chunked_mixin, tmp_path):
        """All chunks successfully downloaded should return all paths."""
        chunks = [(2020, 1), (2020, 2), (2020, 3)]

        def mock_download(chunk):
            year, month = chunk
            path = tmp_path / f"{year}_{month:02d}.nc"
            path.touch()
            return path

        result = chunked_mixin.download_chunks_parallel(
            chunks,
            mock_download,
            max_workers=2
        )

        assert len(result) == 3
        assert all(p.exists() for p in result)

    def test_download_function_returns_none(self, chunked_mixin):
        """Downloads returning None should be skipped."""
        chunks = [(2020, 1), (2020, 2), (2020, 3)]

        # Second chunk returns None (skipped)
        def mock_download(chunk):
            if chunk == (2020, 2):
                return None
            return Path(f"/tmp/{chunk[0]}_{chunk[1]}.nc")

        result = chunked_mixin.download_chunks_parallel(
            chunks,
            mock_download,
            max_workers=2
        )

        assert len(result) == 2

    def test_fail_fast_on_error(self, chunked_mixin):
        """With fail_fast=True, should raise on first error."""
        chunks = [(2020, 1), (2020, 2), (2020, 3)]

        def mock_download(chunk):
            if chunk == (2020, 2):
                raise Exception("Download failed")
            return Path(f"/tmp/{chunk[0]}_{chunk[1]}.nc")

        with pytest.raises(Exception) as exc_info:
            chunked_mixin.download_chunks_parallel(
                chunks,
                mock_download,
                fail_fast=True
            )

        assert "Download failed" in str(exc_info.value)

    def test_no_fail_fast_continues(self, chunked_mixin, tmp_path):
        """With fail_fast=False, should continue after errors."""
        chunks = [(2020, 1), (2020, 2), (2020, 3)]

        def mock_download(chunk):
            if chunk == (2020, 2):
                raise Exception("Download failed")
            path = tmp_path / f"{chunk[0]}_{chunk[1]}.nc"
            path.touch()
            return path

        # Should not raise
        result = chunked_mixin.download_chunks_parallel(
            chunks,
            mock_download,
            fail_fast=False
        )

        # Should have 2 successful downloads
        assert len(result) == 2

    def test_empty_chunks_returns_empty(self, chunked_mixin):
        """Empty chunks list should return empty result."""
        result = chunked_mixin.download_chunks_parallel(
            [],
            lambda x: Path(f"/tmp/{x}.nc")
        )

        assert result == []

    def test_logs_progress(self, chunked_mixin, tmp_path):
        """Should log download progress."""
        chunks = [(2020, 1)]

        def mock_download(chunk):
            path = tmp_path / "test.nc"
            path.touch()
            return path

        chunked_mixin.download_chunks_parallel(chunks, mock_download)

        # Should have logged info messages
        chunked_mixin.logger.info.assert_called()


# =============================================================================
# Merge NetCDF Chunks Tests
# =============================================================================

@pytest.mark.mixin_chunked
@pytest.mark.acquisition
class TestMergeNetcdfChunks:
    """Tests for merge_netcdf_chunks method."""

    def test_merge_basic(self, chunked_mixin, tmp_path, sample_netcdf_dataset):
        """Basic merge of multiple files."""
        # Create 3 chunk files
        chunk_files = []
        for i in range(3):
            chunk_path = tmp_path / f"chunk_{i}.nc"
            ds_chunk = sample_netcdf_dataset.isel(time=slice(i*56, (i+1)*56))
            ds_chunk.to_netcdf(chunk_path)
            chunk_files.append(chunk_path)

        output_path = tmp_path / "merged.nc"

        result = chunked_mixin.merge_netcdf_chunks(
            chunk_files,
            output_path,
            cleanup=False
        )

        assert result == output_path
        assert output_path.exists()

        # Verify merged file
        with xr.open_dataset(output_path) as ds:
            assert ds.sizes['time'] == 168

    def test_merge_with_time_slice(self, chunked_mixin, tmp_path, sample_netcdf_dataset):
        """Merge with time slicing."""
        # Create chunk file
        chunk_path = tmp_path / "chunk.nc"
        sample_netcdf_dataset.to_netcdf(chunk_path)

        output_path = tmp_path / "merged.nc"

        # Slice to first 2 days (48 hours)
        time_slice = (
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-02 23:00")
        )

        result = chunked_mixin.merge_netcdf_chunks(
            [chunk_path],
            output_path,
            time_slice=time_slice,
            cleanup=False
        )

        with xr.open_dataset(result) as ds:
            assert ds.sizes['time'] == 48

    def test_merge_cleanup_removes_chunks(self, chunked_mixin, tmp_path, sample_netcdf_dataset):
        """With cleanup=True, chunk files should be deleted."""
        chunk_files = []
        for i in range(2):
            chunk_path = tmp_path / f"chunk_{i}.nc"
            ds_chunk = sample_netcdf_dataset.isel(time=slice(i*84, (i+1)*84))
            ds_chunk.to_netcdf(chunk_path)
            chunk_files.append(chunk_path)

        output_path = tmp_path / "merged.nc"

        chunked_mixin.merge_netcdf_chunks(
            chunk_files,
            output_path,
            cleanup=True
        )

        # Chunk files should be deleted
        for chunk_file in chunk_files:
            assert not chunk_file.exists()

        # Output should exist
        assert output_path.exists()

    def test_merge_no_cleanup(self, chunked_mixin, tmp_path, sample_netcdf_dataset):
        """With cleanup=False, chunk files should remain."""
        chunk_path = tmp_path / "chunk.nc"
        sample_netcdf_dataset.to_netcdf(chunk_path)

        output_path = tmp_path / "merged.nc"

        chunked_mixin.merge_netcdf_chunks(
            [chunk_path],
            output_path,
            cleanup=False
        )

        # Chunk file should still exist
        assert chunk_path.exists()

    def test_merge_empty_list_raises(self, chunked_mixin, tmp_path):
        """Empty chunk list should raise ValueError."""
        output_path = tmp_path / "merged.nc"

        with pytest.raises(ValueError) as exc_info:
            chunked_mixin.merge_netcdf_chunks([], output_path)

        assert "No chunk files" in str(exc_info.value)

    def test_merge_creates_output_dir(self, chunked_mixin, tmp_path, sample_netcdf_dataset):
        """Should create output directory if needed."""
        chunk_path = tmp_path / "chunk.nc"
        sample_netcdf_dataset.to_netcdf(chunk_path)

        # Nested output directory
        output_path = tmp_path / "new_dir" / "sub_dir" / "merged.nc"

        result = chunked_mixin.merge_netcdf_chunks(
            [chunk_path],
            output_path,
            cleanup=False
        )

        assert output_path.exists()

    def test_merge_with_encoding(self, chunked_mixin, tmp_path, sample_netcdf_dataset):
        """Merge with custom encoding."""
        chunk_path = tmp_path / "chunk.nc"
        sample_netcdf_dataset.to_netcdf(chunk_path)

        output_path = tmp_path / "merged.nc"
        encoding = {"temperature": {"zlib": True, "complevel": 5}}

        result = chunked_mixin.merge_netcdf_chunks(
            [chunk_path],
            output_path,
            encoding=encoding,
            cleanup=False
        )

        assert output_path.exists()


# =============================================================================
# Get NetCDF Encoding Tests
# =============================================================================

@pytest.mark.mixin_chunked
@pytest.mark.acquisition
class TestGetNetcdfEncoding:
    """Tests for get_netcdf_encoding method."""

    def test_encoding_with_compression(self, chunked_mixin, sample_netcdf_dataset):
        """With compression=True, should include zlib settings."""
        encoding = chunked_mixin.get_netcdf_encoding(
            sample_netcdf_dataset,
            compression=True,
            complevel=5
        )

        assert 'temperature' in encoding
        assert encoding['temperature']['zlib'] is True
        assert encoding['temperature']['complevel'] == 5

    def test_encoding_without_compression(self, chunked_mixin, sample_netcdf_dataset):
        """With compression=False, should not include zlib."""
        encoding = chunked_mixin.get_netcdf_encoding(
            sample_netcdf_dataset,
            compression=False
        )

        # May have empty dict or no zlib key
        if 'temperature' in encoding:
            assert encoding['temperature'].get('zlib', False) is False

    def test_encoding_time_chunking(self, chunked_mixin, sample_netcdf_dataset):
        """Should set time chunk size."""
        encoding = chunked_mixin.get_netcdf_encoding(
            sample_netcdf_dataset,
            chunk_time=24
        )

        if 'chunksizes' in encoding.get('temperature', {}):
            # Time dimension should be chunked to 24
            chunksizes = encoding['temperature']['chunksizes']
            # Time is first dimension
            assert chunksizes[0] == 24

    def test_encoding_auto_time_chunk(self, chunked_mixin, sample_netcdf_dataset):
        """Should auto-calculate reasonable time chunk."""
        encoding = chunked_mixin.get_netcdf_encoding(sample_netcdf_dataset)

        if 'chunksizes' in encoding.get('temperature', {}):
            chunksizes = encoding['temperature']['chunksizes']
            # Default is min(168, time_size) = 168 for 1 week
            assert chunksizes[0] <= 168

    def test_encoding_multiple_variables(self, chunked_mixin):
        """Should generate encoding for all data variables."""
        ds = xr.Dataset({
            'var1': (['time', 'x'], np.random.rand(100, 10)),
            'var2': (['time', 'x'], np.random.rand(100, 10)),
        })

        encoding = chunked_mixin.get_netcdf_encoding(ds)

        assert 'var1' in encoding
        assert 'var2' in encoding


# =============================================================================
# Edge Cases
# =============================================================================

@pytest.mark.mixin_chunked
@pytest.mark.acquisition
class TestChunkedMixinEdgeCases:
    """Edge case tests for chunked download mixin."""

    def test_chunks_with_leap_year(self, chunked_mixin):
        """Should handle leap years correctly."""
        start = pd.Timestamp("2020-02-01")
        end = pd.Timestamp("2020-02-29")  # Leap year

        chunks = chunked_mixin.generate_temporal_chunks(start, end, freq='MS')

        assert len(chunks) == 1
        assert chunks[0][1] == pd.Timestamp("2020-02-29")

    def test_chunks_without_leap_year(self, chunked_mixin):
        """Should handle non-leap years correctly."""
        start = pd.Timestamp("2021-02-01")
        end = pd.Timestamp("2021-02-28")

        chunks = chunked_mixin.generate_temporal_chunks(start, end, freq='MS')

        assert len(chunks) == 1
        assert chunks[0][1] == pd.Timestamp("2021-02-28")

    def test_single_file_merge(self, chunked_mixin, tmp_path, sample_netcdf_dataset):
        """Merging single file should work (pass-through)."""
        chunk_path = tmp_path / "single.nc"
        sample_netcdf_dataset.to_netcdf(chunk_path)

        output_path = tmp_path / "merged.nc"

        result = chunked_mixin.merge_netcdf_chunks(
            [chunk_path],
            output_path,
            cleanup=False
        )

        assert output_path.exists()
        with xr.open_dataset(output_path) as ds:
            assert ds.sizes['time'] == 168
