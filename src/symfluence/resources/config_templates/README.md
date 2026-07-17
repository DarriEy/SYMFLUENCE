# SYMFLUENCE Configuration Templates

This directory contains configuration templates for SYMFLUENCE workflows.
Choose the appropriate template based on your use case.

## Template Hierarchy

```
config_templates/
├── config_template_comprehensive.yaml        # Reference: ALL options (flat style)
├── config_template_comprehensive_nested.yaml # Reference: ALL options (nested style)
├── config_template.yaml                      # Standard: Common options with docs
└── config_quickstart_minimal_nested.yaml     # Quickstart: Nested style
```

## Which Template Should I Use?

### Starting a New Project

The easiest way to generate a starting configuration is the CLI:

```bash
symfluence project init --domain my_domain --model SUMMA --minimal
```

| Template | Use When |
|----------|----------|
| `config_quickstart_minimal_nested.yaml` | You prefer organized nested structure (domain, forcing, model sections) |
| `config_template.yaml` | You want common options with inline documentation |

### Reference and Documentation

| Template | Style | Use When |
|----------|-------|----------|
| `config_template_comprehensive.yaml` | Flat | Looking up ALL options with UPPERCASE keys |
| `config_template_comprehensive_nested.yaml` | Nested | Looking up ALL options organized by hydrological workflow |

These are the **authoritative references** - every configuration option is documented
with type hints, defaults, and descriptions. The nested version follows hydrological
modeling best practices:

1. **System** - Infrastructure (paths, logging, execution)
2. **Domain** - Watershed definition (extent, timing, discretization)
3. **Data** - Geospatial inputs (DEM, soils, landcover)
4. **Forcing** - Meteorological drivers (ERA5, RDRS, etc.)
5. **Model** - Process representation (SUMMA, FUSE, GR, etc.)
6. **Optimization** - Parameter calibration (PSO, DE, DDS, etc.)
7. **Evaluation** - Model assessment (streamflow, snow, ET, etc.)
8. **Paths** - File/directory locations

## Configuration Styles

SYMFLUENCE supports two configuration styles that can be mixed:

### Flat Style (Traditional)
```yaml
DOMAIN_NAME: my_domain
EXPERIMENT_TIME_START: "2000-01-01"
FORCING_DATASET: ERA5
```

### Nested Style (Recommended)
```yaml
domain:
  name: my_domain
  time_start: "2000-01-01"
forcing:
  dataset: ERA5
```

Both styles are fully supported. The nested style provides better organization
for complex configurations.

## Key Configuration Sections

| Section | Description |
|---------|-------------|
| `domain` | Domain name, coordinates, time period, discretization |
| `forcing` | Forcing dataset, variables, time step |
| `model` | Hydrological model, routing model, model-specific settings |
| `optimization` | Calibration algorithm, parameters, metrics |
| `paths` | Data directories, shapefile paths, output locations |
| `system` | Code/data directories, parallelization settings |

## Creating Your Configuration

1. **Generate** a config with `symfluence project init` (or copy the quickstart template)
2. **Modify** the required fields (domain name, coordinates, time period)
3. **Reference** `config_template_comprehensive.yaml` for additional options
4. **Validate** using `symfluence config validate your_config.yaml`

## Validation

SYMFLUENCE validates configurations using Pydantic models. Common validation includes:

- Required fields are present
- Time periods are valid (start before end)
- Coordinates are within valid ranges
- File paths exist (when required)

Run validation with:
```bash
symfluence config validate path/to/config.yaml
```
