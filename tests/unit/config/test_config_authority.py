"""
Comprehensive configuration validation tests for SYMFLUENCE.

These tests ensure that:
1. config_template_comprehensive.yaml is the authoritative source
2. All configuration options in code are documented in the template
3. No undocumented settings can be used
4. Configuration stays synchronized with Pydantic models
5. All Pydantic model fields have corresponding YAML entries
"""

import re
from pathlib import Path
import pytest
import yaml
from typing import Set, Dict, List, Tuple


class TestConfigAuthority:
    """Tests for configuration template authority and completeness."""

    @pytest.fixture
    def config_template_path(self):
        """Path to the comprehensive config template."""
        return Path(__file__).parent.parent.parent.parent.parent / 'src' / 'symfluence' / 'resources' / \
               'config_templates' / 'config_template_comprehensive.yaml'

    @pytest.fixture
    def pydantic_models_dir(self):
        """Path to Pydantic model definitions."""
        return Path(__file__).parent.parent.parent.parent.parent / 'src' / 'symfluence' / 'core' / 'config' / 'models'

    @pytest.fixture
    def documented_settings(self, config_template_path):
        """Extract all documented configuration settings from YAML template."""
        with open(config_template_path, 'r') as f:
            content = f.read()

        # Extract all YAML keys (lines starting with uppercase/lowercase alphanumeric followed by :)
        settings = set(re.findall(r'^([a-zA-Z0-9_]+):', content, re.MULTILINE))
        return settings

    @pytest.fixture
    def pydantic_aliases(self, pydantic_models_dir):
        """Extract all field aliases from Pydantic models."""
        aliases = set()

        for model_file in pydantic_models_dir.glob('*.py'):
            if model_file.name.startswith('__'):
                continue

            with open(model_file, 'r') as f:
                content = f.read()

            # Find all alias definitions in Field()
            matches = re.findall(r"alias='([^']+)'", content)
            aliases.update(matches)

        return aliases

    def test_config_template_exists(self, config_template_path):
        """Test that the comprehensive config template file exists."""
        assert config_template_path.exists(), \
            f"Config template not found at {config_template_path}"
        assert config_template_path.stat().st_size > 0, \
            "Config template is empty"

    def test_config_template_is_valid_yaml(self, config_template_path):
        """Test that the comprehensive config template is valid YAML."""
        try:
            with open(config_template_path, 'r') as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            pytest.fail(f"Config template is not valid YAML: {e}")

    def test_no_duplicate_keys_in_template(self, config_template_path):
        """Test that there are no duplicate configuration keys."""
        with open(config_template_path, 'r') as f:
            content = f.read()

        # Find all YAML keys
        keys = re.findall(r'^([a-zA-Z0-9_]+):', content, re.MULTILINE)
        
        # Check for duplicates
        seen = set()
        duplicates = set()
        for key in keys:
            if key in seen:
                duplicates.add(key)
            seen.add(key)

        assert not duplicates, \
            f"Duplicate configuration keys found in template: {duplicates}"

    def test_minimum_documented_settings(self, documented_settings):
        """Test that minimum expected number of settings are documented."""
        # We expect at least 360 settings (accounting for nested/aliased entries)
        assert len(documented_settings) >= 360, \
            f"Template should document at least 360 settings, found {len(documented_settings)}"

    def test_all_pydantic_aliases_in_template(self, documented_settings, pydantic_aliases):
        """Test that all Pydantic model field aliases are documented in template.
        
        This ensures no new configuration options can be added to Pydantic models
        without being documented in the template.
        """
        missing = pydantic_aliases - documented_settings

        assert not missing, \
            f"The following Pydantic model aliases are not documented in the template:\n" \
            f"{sorted(missing)}\n\n" \
            f"You must add these to config_template_comprehensive.yaml"

    def test_all_template_settings_have_type_hint(self, config_template_path):
        """Test that all settings in template have type hint documentation."""
        with open(config_template_path, 'r') as f:
            lines = f.readlines()

        # Find all config entries (lines starting with alphanumeric)
        issues = []
        for i, line in enumerate(lines, 1):
            if re.match(r'^[a-zA-Z0-9_]+:', line):
                # Check if previous lines contain Type: comment
                has_type = False
                for j in range(max(0, i-5), i):
                    if '#   Type:' in lines[j]:
                        has_type = True
                        break

                if not has_type:
                    key = line.split(':')[0].strip()
                    issues.append(f"Line {i}: {key} missing Type: documentation")

        assert not issues, \
            f"Found settings without type documentation:\n" + "\n".join(issues)

    def test_all_template_settings_have_default(self, config_template_path):
        """Test that all settings in template have default value documentation."""
        with open(config_template_path, 'r') as f:
            lines = f.readlines()

        issues = []
        for i, line in enumerate(lines, 1):
            if re.match(r'^[a-zA-Z0-9_]+:', line):
                # Check if previous lines contain Default: comment
                has_default = False
                for j in range(max(0, i-5), i):
                    if '#   Default:' in lines[j]:
                        has_default = True
                        break

                if not has_default:
                    key = line.split(':')[0].strip()
                    issues.append(f"Line {i}: {key} missing Default: documentation")

        assert not issues, \
            f"Found settings without default value documentation:\n" + "\n".join(issues)

    def test_all_template_settings_have_source_reference(self, config_template_path):
        """Test that all settings reference their source Pydantic model."""
        with open(config_template_path, 'r') as f:
            lines = f.readlines()

        issues = []
        for i, line in enumerate(lines, 1):
            if re.match(r'^[a-zA-Z0-9_]+:', line):
                # Check if previous lines contain Source: comment
                has_source = False
                for j in range(max(0, i-5), i):
                    if '#   Source:' in lines[j]:
                        has_source = True
                        break

                if not has_source:
                    key = line.split(':')[0].strip()
                    issues.append(f"Line {i}: {key} missing Source: documentation")

        assert not issues, \
            f"Found settings without source code reference:\n" + "\n".join(issues)

    def test_config_sections_are_organized(self, config_template_path):
        """Test that settings are organized into logical sections."""
        with open(config_template_path, 'r') as f:
            content = f.read()

        # Count section headers
        sections = re.findall(r'^# \d+\. .+$', content, re.MULTILINE)
        
        assert len(sections) >= 15, \
            f"Expected at least 15 organized sections, found {len(sections)}"

    def test_required_sections_exist(self, config_template_path):
        """Test that all essential sections are present."""
        with open(config_template_path, 'r') as f:
            content = f.read()

        required_sections = [
            'System',
            'Domain',
            'Forcing',
            'Model',
            'Optimization',
            'Evaluation',
            'Paths',
        ]

        for section in required_sections:
            assert section in content, \
                f"Required section '{section}' not found in template"


class TestConfigConsistency:
    """Tests to ensure configuration consistency between template and code."""

    @pytest.fixture
    def config_template_path(self):
        """Path to the comprehensive config template."""
        return Path(__file__).parent.parent.parent.parent.parent / 'src' / 'symfluence' / 'resources' / \
               'config_templates' / 'config_template_comprehensive.yaml'

    @pytest.fixture
    def pydantic_models_dir(self):
        """Path to Pydantic model definitions."""
        return Path(__file__).parent.parent.parent.parent.parent / 'src' / 'symfluence' / 'core' / 'config' / 'models'

    @pytest.fixture
    def template_settings(self, config_template_path):
        """Extract settings from template with their metadata."""
        with open(config_template_path, 'r') as f:
            content = f.read()

        settings = {}
        lines = content.split('\n')

        for i, line in enumerate(lines):
            if re.match(r'^[a-zA-Z0-9_]+:', line):
                key = line.split(':')[0].strip()
                
                # Look backwards for metadata
                type_hint = None
                default = None
                source = None
                
                for j in range(max(0, i-5), i):
                    if '#   Type:' in lines[j]:
                        type_hint = lines[j].split('Type:')[1].strip()
                    if '#   Default:' in lines[j]:
                        default = lines[j].split('Default:')[1].strip()
                    if '#   Source:' in lines[j]:
                        source = lines[j].split('Source:')[1].strip()
                
                settings[key] = {
                    'type': type_hint,
                    'default': default,
                    'source': source
                }

        return settings

    def test_all_aliases_have_metadata(self, template_settings):
        """Test that all documented settings have complete metadata."""
        incomplete = {}
        
        for key, metadata in template_settings.items():
            missing_fields = []
            if not metadata['type']:
                missing_fields.append('Type')
            if not metadata['default']:
                missing_fields.append('Default')
            if not metadata['source']:
                missing_fields.append('Source')
            
            if missing_fields:
                incomplete[key] = missing_fields

        assert not incomplete, \
            f"Settings with incomplete metadata:\n" + \
            "\n".join(f"  {k}: missing {', '.join(v)}" for k, v in incomplete.items())

    def test_pydantic_model_field_counts(self, pydantic_models_dir):
        """Test that Pydantic models have expected field counts."""
        model_field_counts = {}

        for model_file in pydantic_models_dir.glob('*.py'):
            if model_file.name.startswith('__'):
                continue

            with open(model_file, 'r') as f:
                content = f.read()

            # Find Field definitions
            fields = re.findall(r'(\w+):\s+(?:Optional\[)?(\w+)(?:\])?.*=\s*Field\(', content)
            
            if fields:
                model_field_counts[model_file.name] = len(fields)

        assert sum(model_field_counts.values()) >= 360, \
            f"Expected at least 360 total fields in models, found {sum(model_field_counts.values())}"

    def test_no_orphaned_settings_in_pydantic(self, 
                                             template_settings, 
                                             pydantic_models_dir):
        """Test that no new settings exist in Pydantic models without template docs."""
        all_aliases = set()

        for model_file in pydantic_models_dir.glob('*.py'):
            if model_file.name.startswith('__'):
                continue

            with open(model_file, 'r') as f:
                content = f.read()

            matches = re.findall(r"alias='([^']+)'", content)
            all_aliases.update(matches)

        template_keys = set(template_settings.keys())
        orphaned = all_aliases - template_keys

        assert not orphaned, \
            f"The following Pydantic aliases are not in the template:\n" \
            f"{sorted(orphaned)}\n\n" \
            f"Add these settings to config_template_comprehensive.yaml"


class TestQuickstartTemplates:
    """Tests for minimal quickstart templates."""

    @pytest.fixture
    def quickstart_minimal_path(self):
        """Path to flat-style quickstart template."""
        return Path(__file__).parent.parent.parent.parent.parent / 'src' / 'symfluence' / 'resources' / \
               'config_templates' / 'config_quickstart_minimal.yaml'

    @pytest.fixture
    def quickstart_nested_path(self):
        """Path to nested-style quickstart template."""
        return Path(__file__).parent.parent.parent.parent.parent / 'src' / 'symfluence' / 'resources' / \
               'config_templates' / 'config_quickstart_minimal_nested.yaml'

    def test_quickstart_minimal_exists(self, quickstart_minimal_path):
        """Test that flat-style quickstart template exists."""
        assert quickstart_minimal_path.exists(), \
            f"Quickstart minimal template not found at {quickstart_minimal_path}"

    def test_quickstart_nested_exists(self, quickstart_nested_path):
        """Test that nested-style quickstart template exists."""
        assert quickstart_nested_path.exists(), \
            f"Quickstart nested template not found at {quickstart_nested_path}"

    def test_quickstart_minimal_has_required_fields(self, quickstart_minimal_path):
        """Test that flat-style quickstart has all 10 required fields."""
        with open(quickstart_minimal_path, 'r') as f:
            content = f.read()

        required = [
            'SYMFLUENCE_DATA_DIR',
            'SYMFLUENCE_CODE_DIR',
            'DOMAIN_NAME',
            'EXPERIMENT_ID',
            'EXPERIMENT_TIME_START',
            'EXPERIMENT_TIME_END',
            'DOMAIN_DEFINITION_METHOD',
            'DOMAIN_DISCRETIZATION',
            'FORCING_DATASET',
            'HYDROLOGICAL_MODEL',
        ]

        for field in required:
            assert field in content, \
                f"Required field {field} not found in flat-style quickstart template"

    def test_quickstart_nested_has_required_fields(self, quickstart_nested_path):
        """Test that nested-style quickstart has all 10 required fields."""
        with open(quickstart_nested_path, 'r') as f:
            content = f.read()

        required = [
            'data_dir',      # system.data_dir
            'code_dir',      # system.code_dir
            'name',          # domain.name
            'experiment_id', # domain.experiment_id
            'time_start',    # domain.time_start
            'time_end',      # domain.time_end
            'definition_method',  # domain.definition_method
            'discretization',     # domain.discretization
            'dataset',       # forcing.dataset
            'hydrological_model', # model.hydrological_model
        ]

        for field in required:
            assert field in content, \
                f"Required field {field} not found in nested-style quickstart template"

    def test_quickstart_templates_are_valid_yaml(self, 
                                                quickstart_minimal_path,
                                                quickstart_nested_path):
        """Test that both quickstart templates are valid YAML."""
        for template_path in [quickstart_minimal_path, quickstart_nested_path]:
            try:
                with open(template_path, 'r') as f:
                    yaml.safe_load(f)
            except yaml.YAMLError as e:
                pytest.fail(f"Quickstart template {template_path.name} is not valid YAML: {e}")

    def test_quickstart_has_documentation(self, quickstart_minimal_path):
        """Test that quickstart template includes helpful documentation."""
        with open(quickstart_minimal_path, 'r') as f:
            content = f.read()

        # Should have comments explaining each required field
        assert '# Root directory' in content or \
               '# Unique identifier' in content or \
               'REQUIRED' in content, \
            "Quickstart template should have descriptive comments"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
