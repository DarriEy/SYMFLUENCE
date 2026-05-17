"""Tests for ATTRIBUTE_PROFILE config flag and profile-driven acquisition."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from symfluence.core.config.models.domain import DomainConfig
from symfluence.data.acquisition.attribute_profiles import (
    PROFILES,
    ProfileDataset,
)

# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------

class TestProfileDefinitions:
    """Validate the profile registry."""

    def test_core_profile_exists(self):
        assert 'core' in PROFILES

    def test_camels_spat_profile_exists(self):
        assert 'camels_spat' in PROFILES

    def test_core_is_empty(self):
        assert PROFILES['core'] == []

    def test_camels_spat_has_eight_datasets(self):
        assert len(PROFILES['camels_spat']) == 8

    def test_camels_spat_handler_names(self):
        names = {ds.handler_name for ds in PROFILES['camels_spat']}
        expected = {
            'SOILGRIDS_PROPERTIES', 'PELLETIER', 'GLHYMPS',
            'HYDROLAKES', 'MODIS_LAI', 'GLAD_TREE_HEIGHT',
            'WORLDCLIM', 'GLCLU_2019',
        }
        assert names == expected

    def test_full_profile_exists(self):
        assert 'full' in PROFILES

    def test_full_includes_camels_spat(self):
        cs_names = {ds.handler_name for ds in PROFILES['camels_spat']}
        full_names = {ds.handler_name for ds in PROFILES['full']}
        assert cs_names.issubset(full_names)

    def test_full_has_extra_datasets(self):
        full_names = {ds.handler_name for ds in PROFILES['full']}
        extras = {
            'GLACIER', 'GLWD', 'BEDROCK_DEPTH', 'ARIDITY_INDEX',
            'MODIS_NDVI', 'ROOT_ZONE_STORAGE', 'JRC_WATER',
            'WOKAM', 'MERIT_HYDRO',
        }
        assert extras.issubset(full_names)

    def test_full_has_seventeen_datasets(self):
        assert len(PROFILES['full']) == 17

    def test_all_non_fatal(self):
        for name, datasets in PROFILES.items():
            for ds in datasets:
                assert ds.fatal is False, f"{name}/{ds.handler_name} should be non-fatal"

    def test_all_have_override_keys(self):
        for name, datasets in PROFILES.items():
            for ds in datasets:
                assert ds.config_override_key is not None
                assert ds.config_override_key.startswith('DOWNLOAD_')

    def test_profile_dataset_is_frozen(self):
        ds = ProfileDataset(
            handler_name='TEST',
            description='test',
            output_subdir='test',
        )
        with pytest.raises(AttributeError):
            ds.handler_name = 'OTHER'


# ---------------------------------------------------------------------------
# DomainConfig validation
# ---------------------------------------------------------------------------

class TestAttributeProfileConfig:
    """Validate ATTRIBUTE_PROFILE field on DomainConfig."""

    _MINIMAL = {
        'DOMAIN_NAME': 'test',
        'EXPERIMENT_ID': 'run1',
        'EXPERIMENT_TIME_START': '2000-01-01',
        'EXPERIMENT_TIME_END': '2001-01-01',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'elevation',
    }

    def test_default_is_core(self):
        cfg = DomainConfig.model_validate(self._MINIMAL)
        assert cfg.attribute_profile == 'core'

    def test_accepts_core(self):
        data = {**self._MINIMAL, 'ATTRIBUTE_PROFILE': 'core'}
        cfg = DomainConfig.model_validate(data)
        assert cfg.attribute_profile == 'core'

    def test_accepts_camels_spat(self):
        data = {**self._MINIMAL, 'ATTRIBUTE_PROFILE': 'camels_spat'}
        cfg = DomainConfig.model_validate(data)
        assert cfg.attribute_profile == 'camels_spat'

    def test_accepts_full(self):
        data = {**self._MINIMAL, 'ATTRIBUTE_PROFILE': 'full'}
        cfg = DomainConfig.model_validate(data)
        assert cfg.attribute_profile == 'full'

    def test_case_insensitive(self):
        data = {**self._MINIMAL, 'ATTRIBUTE_PROFILE': 'CAMELS_SPAT'}
        cfg = DomainConfig.model_validate(data)
        assert cfg.attribute_profile == 'camels_spat'

    def test_rejects_invalid_profile(self):
        data = {**self._MINIMAL, 'ATTRIBUTE_PROFILE': 'nonexistent'}
        with pytest.raises(Exception):
            DomainConfig.model_validate(data)


# ---------------------------------------------------------------------------
# _acquire_profile_datasets() logic
# ---------------------------------------------------------------------------

def _make_service(profile='core', overrides=None):
    """Create a minimal AcquisitionService for profile testing."""
    from symfluence.data.acquisition.acquisition_service import (
        AcquisitionService,
    )

    mock_config = MagicMock()
    mock_config.domain.attribute_profile = profile
    mock_config.domain.name = 'test_domain'
    mock_config.system.data_dir = '/tmp/test'
    mock_config.domain.bounding_box_coords = '51.76/-116.55/50.95/-115.5'

    logger = logging.getLogger('test_profiles')
    logger.setLevel(logging.DEBUG)

    with patch.object(
        AcquisitionService, '__init__', lambda self, *a, **kw: None,
    ):
        svc = AcquisitionService.__new__(AcquisitionService)

    svc._config = mock_config
    svc.config = mock_config
    svc.logger = logger
    svc.project_dir = Path('/tmp/test/domain_test_domain')
    svc.domain_name = 'test_domain'
    svc.reporting_manager = None
    svc._config_dict_override = overrides or {}
    return svc


class TestAcquireProfileDatasets:
    """Unit tests for _acquire_profile_datasets()."""

    def test_core_profile_returns_immediately(self):
        svc = _make_service(profile='core')
        svc._acquire_profile_datasets()

    @patch(
        'symfluence.data.acquisition.acquisition_service.AcquisitionRegistry'
    )
    @patch(
        'symfluence.data.acquisition.acquisition_service.resolve_data_subdir'
    )
    def test_camels_spat_creates_tasks(
        self, mock_resolve, mock_registry
    ):
        svc = _make_service(profile='camels_spat')

        mock_resolve.return_value = Path('/tmp/test/attributes')
        mock_handler = MagicMock()
        mock_handler.download.return_value = Path('/tmp/test/out')
        mock_registry.get_handler.return_value = mock_handler

        svc._acquire_profile_datasets()

        assert mock_registry.get_handler.call_count == 8

    @patch(
        'symfluence.data.acquisition.acquisition_service.AcquisitionRegistry'
    )
    @patch(
        'symfluence.data.acquisition.acquisition_service.resolve_data_subdir'
    )
    def test_override_skips_dataset(self, mock_resolve, mock_registry):
        svc = _make_service(
            profile='camels_spat',
            overrides={'DOWNLOAD_GLHYMPS': False},
        )

        mock_resolve.return_value = Path('/tmp/test/attributes')
        mock_handler = MagicMock()
        mock_handler.download.return_value = Path('/tmp/test/out')
        mock_registry.get_handler.return_value = mock_handler

        svc._acquire_profile_datasets()

        handler_names = [
            call.args[0]
            for call in mock_registry.get_handler.call_args_list
        ]
        assert 'GLHYMPS' not in handler_names
        assert mock_registry.get_handler.call_count == 7

    @patch(
        'symfluence.data.acquisition.acquisition_service.AcquisitionRegistry'
    )
    @patch(
        'symfluence.data.acquisition.acquisition_service.resolve_data_subdir'
    )
    def test_nonfatal_failure_warns(self, mock_resolve, mock_registry):
        svc = _make_service(profile='camels_spat')

        mock_resolve.return_value = Path('/tmp/test/attributes')
        mock_handler = MagicMock()
        mock_handler.download.side_effect = RuntimeError('network error')
        mock_registry.get_handler.return_value = mock_handler

        svc._acquire_profile_datasets()
