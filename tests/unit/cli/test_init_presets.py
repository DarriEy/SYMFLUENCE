"""Unit tests for init_presets module."""

import pytest

from symfluence.cli.init_presets import (
    PRESETS,
    load_presets,
    get_preset,
    list_preset_names,
    validate_preset
)

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestPresetLoading:
    """Test preset loading functionality."""

    def test_load_presets_returns_dict(self):
        """Test load_presets returns a dictionary."""
        presets = load_presets()
        assert isinstance(presets, dict)
        assert len(presets) > 0

    def test_load_presets_contains_expected_presets(self):
        """Test load_presets contains expected initial presets."""
        presets = load_presets()
        assert 'fuse-provo' in presets
        assert 'summa-basic' in presets
        assert 'fuse-basic' in presets

    def test_presets_have_required_keys(self):
        """Test all presets have required keys."""
        presets = load_presets()
        required_keys = {'description', 'base_template', 'settings'}

        for preset_name, preset in presets.items():
            assert all(key in preset for key in required_keys), \
                f"Preset {preset_name} missing required keys"


class TestGetPreset:
    """Test get_preset functionality."""

    def test_get_preset_returns_valid_preset(self):
        """Test get_preset returns a valid preset."""
        preset = get_preset('fuse-provo')
        assert isinstance(preset, dict)
        assert 'description' in preset
        assert 'settings' in preset

    def test_get_preset_raises_on_invalid_name(self):
        """Test get_preset raises ValueError for invalid preset name."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset('nonexistent-preset')

    def test_get_preset_error_message_includes_available_presets(self):
        """Test error message includes available preset names."""
        try:
            get_preset('invalid-name')
        except ValueError as e:
            error_msg = str(e)
            assert 'fuse-provo' in error_msg
            assert 'summa-basic' in error_msg


class TestListPresetNames:
    """Test list_preset_names functionality."""

    def test_list_preset_names_returns_list(self):
        """Test list_preset_names returns a list."""
        names = list_preset_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_list_preset_names_contains_expected_names(self):
        """Test list_preset_names contains expected preset names."""
        names = list_preset_names()
        assert 'fuse-provo' in names
        assert 'summa-basic' in names
        assert 'fuse-basic' in names


class TestValidatePreset:
    """Test validate_preset functionality."""

    def test_validate_valid_preset(self):
        """Test validate_preset returns True for valid presets."""
        preset = get_preset('fuse-provo')
        is_valid, errors = validate_preset(preset)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_preset_missing_description(self):
        """Test validate_preset detects missing description."""
        invalid_preset = {
            'base_template': 'template.yaml',
            'settings': {}
        }
        is_valid, errors = validate_preset(invalid_preset)
        assert is_valid is False
        assert any('description' in error for error in errors)

    def test_validate_preset_missing_settings(self):
        """Test validate_preset detects missing settings."""
        invalid_preset = {
            'description': 'Test preset',
            'base_template': 'template.yaml'
        }
        is_valid, errors = validate_preset(invalid_preset)
        assert is_valid is False
        assert any('settings' in error for error in errors)

    def test_validate_preset_invalid_settings_type(self):
        """Test validate_preset detects invalid settings type."""
        invalid_preset = {
            'description': 'Test preset',
            'base_template': 'template.yaml',
            'settings': 'not a dict'
        }
        is_valid, errors = validate_preset(invalid_preset)
        assert is_valid is False
        assert any('settings' in error and 'dictionary' in error for error in errors)

    def test_validate_preset_invalid_fuse_decisions_type(self):
        """Test validate_preset detects invalid fuse_decisions type."""
        invalid_preset = {
            'description': 'Test preset',
            'base_template': 'template.yaml',
            'settings': {},
            'fuse_decisions': 'not a dict'
        }
        is_valid, errors = validate_preset(invalid_preset)
        assert is_valid is False
        assert any('fuse_decisions' in error for error in errors)


class TestPresetContent:
    """Test preset content for correctness."""

    def test_fuse_provo_preset_content(self):
        """Test fuse-provo preset has expected content."""
        preset = get_preset('fuse-provo')
        settings = preset['settings']

        # Check key settings
        assert settings['DOMAIN_NAME'] == 'provo_river'
        assert settings['HYDROLOGICAL_MODEL'] == 'FUSE'
        assert settings['FORCING_DATASET'] == 'ERA5'
        assert settings['DOMAIN_DISCRETIZATION'] == 'lumped'

        # Check FUSE decisions exist
        assert 'fuse_decisions' in preset
        assert 'RFERR' in preset['fuse_decisions']
        assert 'SNOWM' in preset['fuse_decisions']

    def test_summa_basic_preset_content(self):
        """Test summa-basic preset has expected content."""
        preset = get_preset('summa-basic')
        settings = preset['settings']

        # Check key settings
        assert settings['HYDROLOGICAL_MODEL'] == 'SUMMA'
        assert settings['ROUTING_MODEL'] == 'mizuRoute'
        assert settings['DOMAIN_DISCRETIZATION'] == 'GRUs'

        # Check SUMMA decisions exist
        assert 'summa_decisions' in preset

    def test_fuse_basic_preset_content(self):
        """Test fuse-basic preset has expected content."""
        preset = get_preset('fuse-basic')
        settings = preset['settings']

        # Check key settings
        assert settings['HYDROLOGICAL_MODEL'] == 'FUSE'
        assert settings['FUSE_SPATIAL_MODE'] == 'lumped'
        assert settings['DOMAIN_DISCRETIZATION'] == 'lumped'

        # Check FUSE decisions exist
        assert 'fuse_decisions' in preset

    def test_all_presets_have_calibration_params(self):
        """Test all presets define calibration parameters."""
        presets = load_presets()

        for preset_name, preset in presets.items():
            settings = preset['settings']
            # Should have either FUSE or SUMMA calibration params
            has_fuse_params = 'SETTINGS_FUSE_PARAMS_TO_CALIBRATE' in settings
            has_summa_params = 'PARAMS_TO_CALIBRATE' in settings

            assert has_fuse_params or has_summa_params, \
                f"Preset {preset_name} missing calibration parameters"

    def test_all_presets_have_experiment_id(self):
        """Test all presets define an experiment ID."""
        presets = load_presets()

        for preset_name, preset in presets.items():
            settings = preset['settings']
            assert 'EXPERIMENT_ID' in settings, \
                f"Preset {preset_name} missing EXPERIMENT_ID"

    def test_all_presets_have_forcing_dataset(self):
        """Test all presets define a forcing dataset."""
        presets = load_presets()

        for preset_name, preset in presets.items():
            settings = preset['settings']
            assert 'FORCING_DATASET' in settings, \
                f"Preset {preset_name} missing FORCING_DATASET"


class TestPresetConsistency:
    """Test preset internal consistency."""

    def test_fuse_presets_have_fuse_settings(self):
        """Test FUSE presets have appropriate FUSE settings."""
        fuse_presets = ['fuse-provo', 'fuse-basic']

        for preset_name in fuse_presets:
            preset = get_preset(preset_name)
            settings = preset['settings']

            assert settings['HYDROLOGICAL_MODEL'] == 'FUSE'
            assert 'FUSE_SPATIAL_MODE' in settings
            assert 'fuse_decisions' in preset

    def test_summa_presets_have_summa_settings(self):
        """Test SUMMA presets have appropriate SUMMA settings."""
        summa_presets = ['summa-basic']

        for preset_name in summa_presets:
            preset = get_preset(preset_name)
            settings = preset['settings']

            assert settings['HYDROLOGICAL_MODEL'] == 'SUMMA'
            assert settings['ROUTING_MODEL'] == 'mizuRoute'

    def test_preset_templates_exist(self):
        """Test all presets reference valid template names."""
        from symfluence.resources import get_config_template

        presets = load_presets()

        for preset_name, preset in presets.items():
            template_name = preset['base_template']
            # Check it's a valid template name
            assert isinstance(template_name, str)
            assert template_name.endswith('.yaml')
            # Verify template actually exists in package data
            template_path = get_config_template(template_name)
            assert template_path.exists()
