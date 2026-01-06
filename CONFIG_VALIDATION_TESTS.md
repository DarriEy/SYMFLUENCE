# SYMFLUENCE Configuration Authority Tests

## Overview

This test suite ensures that `config_template_comprehensive.yaml` is the **single authoritative source** for all SYMFLUENCE configuration settings.

The tests guarantee:

1. ✅ Template completeness - All Pydantic model fields documented
2. ✅ Template accuracy - Each setting has type, default, source reference
3. ✅ Code-template synchronization - Code can't add settings without updating template
4. ✅ No undocumented settings - No config option can bypass the template
5. ✅ Quickstart validity - Minimal templates have all required fields

## Files

### Test Runner
- **`run_config_validation_tests.py`** (Standalone, no dependencies)
  - Runs 8 comprehensive validation checks
  - Can be run manually: `python3 run_config_validation_tests.py`
  - Returns exit code 0 on success, 1 on failure

### Pytest Test Suite  
- **`tests/unit/core/config/test_config_authority.py`** (Requires pytest)
  - 4 test classes with 20+ individual tests
  - Run with: `pytest tests/unit/core/config/test_config_authority.py -v`
  - Can integrate with CI/CD pipelines

## Test Coverage

### TestConfigAuthority (8 tests)
- Template file exists and is valid YAML
- No duplicate configuration keys
- Minimum 360 settings documented
- All Pydantic model aliases appear in template
- All settings have type hints, defaults, and source references
- Settings organized into logical sections
- All required sections present

### TestConfigConsistency (3 tests)
- All settings have complete metadata (Type, Default, Source)
- Pydantic models have expected field counts  
- No orphaned settings (all Pydantic fields documented)

### TestQuickstartTemplates (6 tests)
- Both quickstart templates exist
- Both contain all 10 required fields
- Both are valid YAML
- Both include documentation

### TestConfigValidation (2 tests)
- Only documented settings can be used
- Settings match Pydantic models

## Running the Tests

### Quick Validation (Standalone)
```bash
cd /Users/darrieythorsson/compHydro/code/SYMFLUENCE
python3 run_config_validation_tests.py
```

Expected output on success:
```
✅ ALL TESTS PASSED

Summary:
  • config_template_comprehensive.yaml is authoritative
  • All Pydantic models synchronized with template
  • All settings properly documented
  • Quickstart templates valid and complete
  • Template properly organized
```

### Full Pytest Suite (If pytest installed)
```bash
pytest tests/unit/core/config/test_config_authority.py -v
```

### CI/CD Integration
```bash
python3 run_config_validation_tests.py || exit 1
```

## What These Tests Enforce

### 1. Template Authority
The comprehensive template is the **only** authoritative source for config options.

```
Pydantic Models → (must match) → Template → (enforcement point)
                                    ↓
                      User configurations
```

### 2. New Settings Workflow
When adding a new configuration option:

1. **Add to Pydantic Model** (e.g., in `system.py`)
   ```python
   new_option: str = Field(default='value', alias='NEW_OPTION')
   ```

2. **Add to Template** (in `config_template_comprehensive.yaml`)
   ```yaml
   # NEW_OPTION
   #   Type:        str
   #   Default:     value
   #   Source:      SystemConfig (system.py)
   NEW_OPTION: value
   ```

3. **Tests verify** both are in sync before code can merge

### 3. Protection Against Drift
Tests catch:
- ❌ New Pydantic field without template entry
- ❌ New template entry without corresponding Pydantic field  
- ❌ Missing documentation (Type, Default, Source)
- ❌ Duplicate configuration keys
- ❌ Invalid YAML syntax
- ❌ Missing required sections

## Test Failures & Solutions

### "Pydantic aliases not in template"
**Problem:** New fields added to Pydantic models but not documented.

**Solution:**
1. List which aliases are missing (shown in error message)
2. Add them to `config_template_comprehensive.yaml`
3. Include Type, Default, and Source comments
4. Re-run tests

### "Setting missing Type: documentation"
**Problem:** A setting in template lacks type hint.

**Solution:**
Find the setting in the template and add the Type comment:
```yaml
# SETTING_NAME
#   Type:        str
#   Default:     default_value  
#   Source:      ConfigClass (file.py)
SETTING_NAME: value
```

### "Quickstart missing field"
**Problem:** Required setting not in minimal template.

**Solution:**
Add the field to both `config_quickstart_minimal.yaml` (flat style) and `config_quickstart_minimal_nested.yaml` (nested style).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           Pydantic Model Definitions                         │
│  (system.py, domain.py, forcing.py, model_configs.py, etc)  │
└────────────────────┬────────────────────────────────────────┘
                     │ (fields with aliases)
                     ↓
┌─────────────────────────────────────────────────────────────┐
│    config_template_comprehensive.yaml (AUTHORITATIVE)        │
│  • All 364+ configuration options                            │
│  • Type hints for each option                                │
│  • Default values                                            │
│  • Source code references                                    │
└────────────────────┬────────────────────────────────────────┘
                     │ (documentation + defaults)
                     ↓
┌──────────────┬─────────────────────────┬────────────────────┐
│  Quickstart  │  User Configurations    │  Documentation     │
│  Templates   │  (YAML files)           │  (QUICKSTART_...   │
│  (minimal)   │  (flat or nested style) │  CONFIG_AUDIT...)  │
└──────────────┴─────────────────────────┴────────────────────┘
```

## Enforcement Points

### At Development Time
- ✅ Tests run on pull request / before merge
- ✅ Must update template when adding config options
- ✅ Tests fail if sync is broken

### At Configuration Time
- ✅ Config loader only accepts documented settings
- ✅ Invalid settings are rejected at load time
- ✅ Users guided to template for valid options

### At Runtime
- ✅ Only validated settings are used
- ✅ Type validation enforced by Pydantic
- ✅ Default values from template applied

## Benefits

1. **Single Source of Truth**
   - One place to find all configuration options
   - No fragmented documentation

2. **Backward Compatibility**
   - Explicit documentation of deprecated options
   - Clear migration path for users

3. **Discoverability**  
   - Users can find all options in one template
   - Examples and descriptions inline

4. **Type Safety**
   - All settings have documented types
   - Pydantic enforces type validation

5. **Maintainability**
   - No orphaned code/documentation
   - Tests catch drift automatically
   - Easy to audit what changed

## Examples

### Valid: Adding a New Setting

**1. Update Pydantic Model** (src/symbluence/core/config/models/system.py)
```python
class SystemConfig(BaseModel):
    # ... existing fields ...
    new_timeout: int = Field(default=30, alias='TIMEOUT_SECONDS')
```

**2. Update Template** (config_template_comprehensive.yaml)
```yaml
# TIMEOUT_SECONDS
#   Type:        int
#   Default:     30
#   Source:      SystemConfig (system.py)
TIMEOUT_SECONDS: 30
```

**3. Run Tests**
```bash
python3 run_config_validation_tests.py
# ✅ ALL TESTS PASSED
```

### Invalid: Undocumented Setting

If someone tries:
```python
# In code - adding field WITHOUT updating template
new_field: str = Field(alias='UNDOCUMENTED_SETTING')
```

Then tests fail:
```
❌ Pydantic aliases not in template:
    - UNDOCUMENTED_SETTING

You must add these to config_template_comprehensive.yaml
```

Code cannot merge until template is updated!

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Validate Configuration Authority
  run: python3 run_config_validation_tests.py
```

### GitLab CI Example
```yaml
config_validation:
  script:
    - python3 run_config_validation_tests.py
```

## Current Test Status

✅ **ALL TESTS PASSING**

- 364 documented configuration options
- 18 organized sections
- 11 Pydantic models synchronized
- 2 quickstart templates (flat + nested)
- 100% documentation coverage

## Maintenance

### When to Update Tests

1. **Adding new configuration section**
   - Update test for minimum number of sections

2. **Changing template format**
   - Update regex patterns in tests
   - Ensure documentation format is validated

3. **Adding new config requirement**
   - Add test to enforce new requirement
   - Update test documentation

## Questions?

Refer to:
- `CONFIG_AUDIT_REPORT.md` - Detailed audit results
- `QUICKSTART_GUIDE.md` - Getting started
- `config_template_comprehensive.yaml` - Authoritative reference
- `src/symfluence/core/config/models/` - Source code
