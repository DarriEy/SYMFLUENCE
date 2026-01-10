# SYMFLUENCE Configuration Templates

This directory contains configuration templates for SYMFLUENCE workflows.
Choose the appropriate template based on your use case.

## Template Hierarchy

```
config_templates/
├── config_template_comprehensive.yaml   # Reference: ALL options documented
├── config_template.yaml                 # Standard: Common options with docs
├── config_quickstart_minimal.yaml       # Quickstart: Flat key style
├── config_quickstart_minimal_nested.yaml # Quickstart: Nested style
├── camelsspat_template.yaml             # CAMELS-SPAT dataset preset
├── fluxnet_template.yaml                # FLUXNET dataset preset
├── norswe_template.yaml                 # NorSWE dataset preset
└── examples/                            # Tutorial-specific configs
```

## Which Template Should I Use?

### Starting a New Project

| Template | Use When |
|----------|----------|
| `config_quickstart_minimal.yaml` | You want the simplest starting point with flat UPPERCASE keys |
| `config_quickstart_minimal_nested.yaml` | You prefer organized nested structure (domain, forcing, model sections) |
| `config_template.yaml` | You want common options with inline documentation |

### Reference and Documentation

| Template | Use When |
|----------|----------|
| `config_template_comprehensive.yaml` | Looking up ALL available options with full documentation |

This is the **authoritative reference** - every configuration option is documented here
with type hints, defaults, and descriptions.

### Dataset-Specific Presets

| Template | Dataset | Use When |
|----------|---------|----------|
| `camelsspat_template.yaml` | CAMELS-SPAT | Working with CAMELS-SPAT catchments |
| `fluxnet_template.yaml` | FLUXNET | Working with FLUXNET tower sites |
| `norswe_template.yaml` | NorSWE | Working with Norwegian SWE data |

These templates have pre-configured paths and settings optimized for their respective datasets.

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

## Examples Directory

The `examples/` subdirectory contains configs used by tutorial notebooks:

- `config_basin_notebook.yaml` - Basin-scale modeling tutorial
- `config_continental_tutorial.yaml` - Continental-scale workflow
- `config_fluxnet_notebook.yaml` - FLUXNET site modeling
- `config_iceland_tutorial.yaml` - Iceland case study

These are maintained for reproducibility and should not be modified.

## Creating Your Configuration

1. **Copy** a quickstart template to your project directory
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
