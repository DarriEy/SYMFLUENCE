"""Tests for StateManager operations."""

from pathlib import Path

import numpy as np
import pytest

from symfluence.models.state.exceptions import StateError
from symfluence.models.state.manager import StateManager
from symfluence.models.state.types import ModelState, StateFormat, StateMetadata


class TestStateManagerSerialization:
    """Tests for serialize/deserialize."""

    def test_roundtrip(self):
        meta = StateMetadata(
            model_name='HBV',
            timestamp='2024-01-01',
            format=StateFormat.MEMORY_ARRAY,
            variables=['snow', 'sm'],
        )
        state = ModelState(
            metadata=meta,
            arrays={'snow': np.array([10.0]), 'sm': np.array([150.0])},
        )

        serialized = StateManager.serialize(state)
        restored = StateManager.deserialize(serialized)

        assert restored.metadata.model_name == 'HBV'
        np.testing.assert_allclose(restored.arrays['snow'], [10.0])
        np.testing.assert_allclose(restored.arrays['sm'], [150.0])


class TestEnsembleStateManagement:
    """Tests for create_ensemble, save_ensemble, load_ensemble."""

    def test_create_ensemble(self):
        meta = StateMetadata(
            model_name='HBV',
            timestamp='2024-01-01',
            format=StateFormat.MEMORY_ARRAY,
            variables=['snow', 'sm'],
        )
        base_state = ModelState(
            metadata=meta,
            arrays={'snow': np.array([10.0]), 'sm': np.array([150.0])},
        )

        def perturb(state, idx):
            arrays = {k: v + np.random.randn(*v.shape) for k, v in state.arrays.items()}
            new_meta = StateMetadata(
                model_name=state.metadata.model_name,
                timestamp=state.metadata.timestamp,
                format=state.metadata.format,
                variables=state.metadata.variables,
                ensemble_member=idx,
            )
            return ModelState(metadata=new_meta, arrays=arrays)

        ensemble = StateManager.create_ensemble(base_state, 5, perturb)
        assert len(ensemble) == 5
        assert ensemble[0].metadata.ensemble_member == 0
        assert ensemble[4].metadata.ensemble_member == 4

    def test_save_load_ensemble(self, tmp_path):
        """Test save/load roundtrip for ensemble states."""
        states = []
        for i in range(3):
            meta = StateMetadata(
                model_name='HBV',
                timestamp='2024-01-01',
                format=StateFormat.MEMORY_ARRAY,
                variables=['snow', 'sm'],
                ensemble_member=i,
            )
            state = ModelState(
                metadata=meta,
                arrays={
                    'snow': np.array([10.0 + i]),
                    'sm': np.array([150.0 + i * 10]),
                },
            )
            states.append(state)

        archive_dir = tmp_path / 'ensemble_archive'
        StateManager.save_ensemble(states, archive_dir)

        # Verify directory structure
        for i in range(3):
            member_dir = archive_dir / f"member_{i:03d}"
            assert member_dir.is_dir()
            assert (member_dir / 'state_arrays.npz').exists()

        # Create mock runner for loading
        class MockRunner:
            def _get_model_name(self):
                return 'HBV'
            def get_state_format(self):
                return StateFormat.MEMORY_ARRAY
            def get_state_variables(self):
                return ['snow', 'sm']
            def get_state_file_pattern(self):
                return '*.npz'

        runner = MockRunner()
        loaded = StateManager.load_ensemble(archive_dir, runner, 3)

        assert len(loaded) == 3
        for i, state in enumerate(loaded):
            assert state.metadata.ensemble_member == i
            np.testing.assert_allclose(state.arrays['snow'], [10.0 + i])
            np.testing.assert_allclose(state.arrays['sm'], [150.0 + i * 10])

    def test_load_ensemble_missing_member(self, tmp_path):
        """Test that loading with missing member raises StateError."""
        archive_dir = tmp_path / 'missing_ensemble'
        archive_dir.mkdir()
        (archive_dir / 'member_000').mkdir()

        class MockRunner:
            def _get_model_name(self):
                return 'HBV'
            def get_state_format(self):
                return StateFormat.MEMORY_ARRAY
            def get_state_variables(self):
                return ['snow']
            def get_state_file_pattern(self):
                return '*.npz'

        with pytest.raises(StateError, match="member_001"):
            StateManager.load_ensemble(archive_dir, MockRunner(), 2)
