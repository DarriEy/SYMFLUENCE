"""Tests for state management types and serialization."""

import numpy as np
import pytest
from pathlib import Path

from symfluence.models.state.types import ModelState, StateMetadata, StateFormat


class TestStateMetadata:
    """Tests for StateMetadata frozen dataclass."""

    def test_creation(self):
        meta = StateMetadata(
            model_name='HBV',
            timestamp='2024-01-01T00:00:00',
            format=StateFormat.MEMORY_ARRAY,
            variables=['snow', 'sm'],
        )
        assert meta.model_name == 'HBV'
        assert meta.format == StateFormat.MEMORY_ARRAY
        assert meta.variables == ['snow', 'sm']
        assert meta.ensemble_member is None

    def test_frozen(self):
        meta = StateMetadata(
            model_name='SUMMA',
            timestamp='2024-01-01',
            format=StateFormat.FILE_NETCDF,
        )
        with pytest.raises(AttributeError):
            meta.model_name = 'other'


class TestModelState:
    """Tests for ModelState dataclass."""

    def test_file_based(self):
        meta = StateMetadata(
            model_name='SUMMA',
            timestamp='2024-01-01',
            format=StateFormat.FILE_NETCDF,
        )
        state = ModelState(metadata=meta, files=[Path('/tmp/state.nc')])
        assert state.is_file_based
        assert not state.is_memory_based

    def test_memory_based(self):
        meta = StateMetadata(
            model_name='HBV',
            timestamp='2024-01-01',
            format=StateFormat.MEMORY_ARRAY,
        )
        state = ModelState(
            metadata=meta,
            arrays={'snow': np.array([10.0]), 'sm': np.array([150.0])},
        )
        assert not state.is_file_based
        assert state.is_memory_based

    def test_serialization_roundtrip(self):
        """Test to_dict / from_dict roundtrip."""
        meta = StateMetadata(
            model_name='HBV',
            timestamp='2024-06-15T12:00:00',
            format=StateFormat.MEMORY_ARRAY,
            variables=['snow', 'sm', 'suz'],
            ensemble_member=3,
            extra={'warmup_days': 365},
        )
        state = ModelState(
            metadata=meta,
            files=[Path('/tmp/state.npz')],
            arrays={
                'snow': np.array([5.0]),
                'sm': np.array([120.0]),
                'suz': np.array([15.0]),
            },
        )

        d = state.to_dict()
        restored = ModelState.from_dict(d)

        assert restored.metadata.model_name == 'HBV'
        assert restored.metadata.timestamp == '2024-06-15T12:00:00'
        assert restored.metadata.format == StateFormat.MEMORY_ARRAY
        assert restored.metadata.variables == ['snow', 'sm', 'suz']
        assert restored.metadata.ensemble_member == 3
        assert restored.metadata.extra == {'warmup_days': 365}
        assert len(restored.files) == 1
        assert restored.files[0] == Path('/tmp/state.npz')
        np.testing.assert_allclose(restored.arrays['snow'], [5.0])
        np.testing.assert_allclose(restored.arrays['sm'], [120.0])
        np.testing.assert_allclose(restored.arrays['suz'], [15.0])


class TestStateFormat:
    """Tests for StateFormat enum."""

    def test_all_formats_exist(self):
        assert StateFormat.FILE_NETCDF
        assert StateFormat.FILE_BINARY
        assert StateFormat.FILE_TEXT
        assert StateFormat.MEMORY_ARRAY
        assert StateFormat.MEMORY_DICT
        assert StateFormat.COMPOSITE

    def test_serialization(self):
        assert StateFormat['MEMORY_ARRAY'] == StateFormat.MEMORY_ARRAY
        assert StateFormat.FILE_NETCDF.name == 'FILE_NETCDF'
