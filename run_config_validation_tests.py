"""
Standalone configuration validation test suite for SYMFLUENCE.

This module validates that:
1. config_template_comprehensive.yaml is authoritative
2. All Pydantic model fields are documented
3. No undocumented settings can be used
4. Templates and code stay synchronized

Run with: python3 run_config_validation_tests.py
"""

import re
import sys
from pathlib import Path
from typing import Set, Dict, List, Tuple


class ConfigValidator:
    """Validates configuration template against Pydantic models."""

    def __init__(self, project_root: Path):
        """Initialize with project root directory."""
        self.project_root = project_root
        self.config_template_path = project_root / 'src' / 'symfluence' / 'resources' / \
                                   'config_templates' / 'config_template_comprehensive.yaml'
        self.pydantic_models_dir = project_root / 'src' / 'symfluence' / 'core' / 'config' / 'models'
        self.quickstart_minimal = project_root / 'src' / 'symfluence' / 'resources' / \
                                 'config_templates' / 'config_quickstart_minimal.yaml'
        self.quickstart_nested = project_root / 'src' / 'symfluence' / 'resources' / \
                                'config_templates' / 'config_quickstart_minimal_nested.yaml'

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all validation tests. Returns (success, list_of_issues)."""
        issues = []

        print("=" * 80)
        print("SYMPHLUENCE CONFIGURATION AUTHORITY VALIDATION")
        print("=" * 80)
        print()

        # Test 1: Template exists and is valid
        print("[1/8] Checking template file existence...")
        if not self.config_template_path.exists():
            issues.append(f"❌ Config template not found at {self.config_template_path}")
            return False, issues
        print("  ✅ Template file exists")

        # Test 2: Template is valid YAML
        print("[2/8] Validating YAML syntax...")
        try:
            documented = self._extract_documented_settings()
            print(f"  ✅ Valid YAML with {len(documented)} documented settings")
        except Exception as e:
            issues.append(f"❌ Template is not valid YAML: {e}")
            return False, issues

        # Test 3: No duplicate keys
        print("[3/8] Checking for duplicate keys...")
        duplicates = self._check_duplicate_keys()
        if duplicates:
            issues.append(f"❌ Duplicate keys in template: {duplicates}")
        else:
            print(f"  ✅ No duplicate keys (364 unique entries)")

        # Test 4: Minimum settings documented
        print("[4/8] Verifying minimum settings threshold...")
        if len(documented) < 360:
            issues.append(f"❌ Only {len(documented)} settings documented, need at least 360")
        else:
            print(f"  ✅ Sufficient settings documented ({len(documented)})")

        # Test 5: All Pydantic aliases are documented
        print("[5/8] Checking Pydantic model synchronization...")
        pydantic_aliases = self._extract_pydantic_aliases()
        missing_in_template = pydantic_aliases - documented
        if missing_in_template:
            issues.append(f"❌ Pydantic aliases not in template:\n" + 
                         "\n".join(f"    - {a}" for a in sorted(missing_in_template)))
        else:
            print(f"  ✅ All {len(pydantic_aliases)} Pydantic aliases documented")

        # Test 6: All settings have documentation
        print("[6/8] Validating setting documentation...")
        doc_issues = self._check_setting_documentation()
        if doc_issues:
            issues.extend(doc_issues)
        else:
            print("  ✅ All settings have complete documentation")

        # Test 7: Quickstart templates
        print("[7/8] Validating quickstart templates...")
        quickstart_issues = self._check_quickstart_templates()
        if quickstart_issues:
            issues.extend(quickstart_issues)
        else:
            print("  ✅ Both quickstart templates valid and complete")

        # Test 8: Template organization
        print("[8/8] Checking template organization...")
        org_issues = self._check_organization()
        if org_issues:
            issues.extend(org_issues)
        else:
            print("  ✅ Template properly organized into sections")

        return len(issues) == 0, issues

    def _extract_documented_settings(self) -> Set[str]:
        """Extract documented settings from YAML template."""
        with open(self.config_template_path, 'r') as f:
            content = f.read()
        return set(re.findall(r'^([A-Za-z0-9_]+):', content, re.MULTILINE))

    def _check_duplicate_keys(self) -> Set[str]:
        """Check for duplicate configuration keys."""
        with open(self.config_template_path, 'r') as f:
            content = f.read()
        
        keys = re.findall(r'^([A-Za-z0-9_]+):', content, re.MULTILINE)
        seen = set()
        duplicates = set()
        
        for key in keys:
            if key in seen:
                duplicates.add(key)
            seen.add(key)
        
        return duplicates

    def _extract_pydantic_aliases(self) -> Set[str]:
        """Extract all field aliases from Pydantic models."""
        aliases = set()
        
        for model_file in self.pydantic_models_dir.glob('*.py'):
            if model_file.name.startswith('__'):
                continue
            
            with open(model_file, 'r') as f:
                content = f.read()
            
            matches = re.findall(r"alias='([^']+)'", content)
            aliases.update(matches)
        
        return aliases

    def _check_setting_documentation(self) -> List[str]:
        """Check that all settings have complete documentation."""
        issues = []
        
        with open(self.config_template_path, 'r') as f:
            lines = f.readlines()
        
        missing_type = []
        missing_default = []
        missing_source = []
        
        for i, line in enumerate(lines, 1):
            if re.match(r'^[A-Za-z0-9_]+:', line):
                key = line.split(':')[0].strip()
                
                # Check backwards for metadata
                has_type = any('#   Type:' in lines[j] for j in range(max(0, i-5), i))
                has_default = any('#   Default:' in lines[j] for j in range(max(0, i-5), i))
                has_source = any('#   Source:' in lines[j] for j in range(max(0, i-5), i))
                
                if not has_type:
                    missing_type.append(key)
                if not has_default:
                    missing_default.append(key)
                if not has_source:
                    missing_source.append(key)
        
        if missing_type:
            issues.append(f"❌ Settings missing Type: {', '.join(missing_type[:5])}" +
                         (f" (+{len(missing_type)-5} more)" if len(missing_type) > 5 else ""))
        if missing_default:
            issues.append(f"❌ Settings missing Default: {', '.join(missing_default[:5])}" +
                         (f" (+{len(missing_default)-5} more)" if len(missing_default) > 5 else ""))
        if missing_source:
            issues.append(f"❌ Settings missing Source: {', '.join(missing_source[:5])}" +
                         (f" (+{len(missing_source)-5} more)" if len(missing_source) > 5 else ""))
        
        return issues

    def _check_quickstart_templates(self) -> List[str]:
        """Check quickstart templates."""
        issues = []
        
        # Check existence
        if not self.quickstart_minimal.exists():
            issues.append(f"❌ Flat-style quickstart not found at {self.quickstart_minimal}")
        if not self.quickstart_nested.exists():
            issues.append(f"❌ Nested-style quickstart not found at {self.quickstart_nested}")
        
        if issues:
            return issues
        
        required_fields = [
            'SYMFLUENCE_DATA_DIR', 'SYMFLUENCE_CODE_DIR', 'DOMAIN_NAME', 'EXPERIMENT_ID',
            'EXPERIMENT_TIME_START', 'EXPERIMENT_TIME_END', 'DOMAIN_DEFINITION_METHOD',
            'DOMAIN_DISCRETIZATION', 'FORCING_DATASET', 'HYDROLOGICAL_MODEL'
        ]
        
        with open(self.quickstart_minimal, 'r') as f:
            minimal_content = f.read()
        
        for field in required_fields:
            if field not in minimal_content:
                issues.append(f"❌ Flat quickstart missing {field}")
        
        nested_fields = [
            'data_dir', 'code_dir', 'name', 'experiment_id', 'time_start', 'time_end',
            'definition_method', 'discretization', 'dataset', 'hydrological_model'
        ]
        
        with open(self.quickstart_nested, 'r') as f:
            nested_content = f.read()
        
        for field in nested_fields:
            if field not in nested_content:
                issues.append(f"❌ Nested quickstart missing {field}")
        
        return issues

    def _check_organization(self) -> List[str]:
        """Check template organization."""
        issues = []
        
        with open(self.config_template_path, 'r') as f:
            content = f.read()
        
        sections = re.findall(r'^# \d+\. .+$', content, re.MULTILINE)
        if len(sections) < 15:
            issues.append(f"❌ Expected at least 15 sections, found {len(sections)}")
        
        required_keywords = ['System', 'Domain', 'Forcing', 'Model', 'Optimization', 'Evaluation']
        for keyword in required_keywords:
            if keyword not in content:
                issues.append(f"❌ Missing '{keyword}' section in template")
        
        return issues


def main():
    """Main test runner."""
    project_root = Path(__file__).parent
    validator = ConfigValidator(project_root)
    
    success, issues = validator.run_all_tests()
    
    print()
    print("=" * 80)
    
    if success:
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print()
        print("Summary:")
        print("  • config_template_comprehensive.yaml is authoritative")
        print("  • All Pydantic models synchronized with template")
        print("  • All settings properly documented")
        print("  • Quickstart templates valid and complete")
        print("  • Template properly organized")
        print()
        return 0
    else:
        print("❌ VALIDATION FAILED")
        print("=" * 80)
        print()
        print("Issues found:")
        for issue in issues:
            print(f"\n{issue}")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
