# User Configuration Workspace

This directory is your personal workspace for SYMFLUENCE configuration files.

## Purpose

- Store your custom configuration files here
- Files in this directory are gitignored (except this README)
- This is NOT part of the Python package distribution

## Configuration Templates

As of SYMFLUENCE v0.6.0, configuration templates have been moved into the package for better distribution.

### Accessing Templates

**Via Python:**
```python
from symfluence.utils.resources import get_config_template, list_config_templates

# Get default template
template = get_config_template()

# Get specific template
template = get_config_template('config_template_comprehensive.yaml')

# List all available templates
templates = list_config_templates()
for t in templates:
    print(t.name)
```

**Via CLI:**
```bash
# List available templates
symfluence config list

# Initialize a new project (creates config from template)
symfluence project init --output-dir ./0_config_files/
```

## Creating New Configurations

```bash
# Using the CLI (recommended)
symfluence project init --output-dir ./0_config_files/

# Or manually copy from package templates
python -c "from symfluence.utils.resources import get_config_template; \
           import shutil; \
           shutil.copy(get_config_template(), './0_config_files/my_config.yaml')"
```

## Directory Structure Changes (v0.8.0)

Prior to v0.8.0, this directory contained:
- ✅ **Templates** → Now in `src/symfluence/data/config_templates/` (distributed with package)
- ✅ **Tutorial configs** → Now in `examples/configs/` (for documentation)
- ✅ **User configs** → Remain here (your personal workspace)

## See Also

- **Base model settings**: `src/symfluence/data/base_settings/` (FUSE, SUMMA, etc.)
- **Tutorial examples**: `examples/configs/` (iceland_tutorial.yaml, etc.)
- **Migration guide**: `docs/source/migration_guide_v0.8.md`
